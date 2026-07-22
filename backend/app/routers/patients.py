
from __future__ import annotations

import json
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..db import get_session
from ..models_db import Patient, Prediction, Study, StudyImage, User
from ..schemas import PatientCreate, PatientRead, PredictionRead, StudyRead

router = APIRouter(prefix="/api/patients", tags=["patients"])


@router.post("", response_model=PatientRead)
def create_patient(body: PatientCreate, session: Session = Depends(get_session)) -> Patient:
    patient = Patient(**body.model_dump())
    session.add(patient)
    session.commit()
    session.refresh(patient)
    return patient


def _patients_to_read_batch(patients: List[Patient], session: Session) -> List[PatientRead]:
    """Batch version of _patient_to_read -- O(1) queries instead of O(N).
    N+1 queries here were the dominant cost of /api/patients (6.9s for 26
    rows): every extra query pays a full round trip, which is expensive
    regardless of whether the database is local or remote."""
    if not patients:
        return []
    patient_ids = [p.id for p in patients]

    studies_by_patient: Dict[int, List[tuple]] = {}
    for study_id, patient_id, uploaded_at in session.exec(
        select(Study.id, Study.patient_id, Study.uploaded_at)
        .where(Study.patient_id.in_(patient_ids))
        .order_by(Study.uploaded_at.desc())
    ).all():
        studies_by_patient.setdefault(patient_id, []).append((study_id, uploaded_at))

    latest_study_ids = [studies[0][0] for studies in studies_by_patient.values() if studies]
    label_by_study: Dict[int, str] = {}
    if latest_study_ids:
        for study_id, label in session.exec(
            select(Prediction.study_id, Prediction.label).where(Prediction.study_id.in_(latest_study_ids))
        ).all():
            label_by_study[study_id] = label

    result = []
    for p in patients:
        studies = studies_by_patient.get(p.id, [])
        result.append(PatientRead(
            id=p.id,
            name=p.name,
            sex=p.sex,
            birth_year=p.birth_year,
            created_at=p.created_at,
            study_count=len(studies),
            last_study_at=studies[0][1] if studies else None,
            last_label=label_by_study.get(studies[0][0]) if studies else None,
        ))
    return result


def _patient_to_read(patient: Patient, session: Session) -> PatientRead:
    return _patients_to_read_batch([patient], session)[0]


@router.get("", response_model=List[PatientRead])
def list_patients(session: Session = Depends(get_session)):
    patients = session.exec(
        select(Patient).where(Patient.org_id.is_(None)).order_by(Patient.created_at.desc())
    ).all()
    return _patients_to_read_batch(patients, session)


def _get_public_patient(patient_id: int, session: Session) -> Patient:
    patient = session.get(Patient, patient_id)
    if not patient or patient.org_id is not None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.get("/{patient_id}", response_model=PatientRead)
def get_patient(patient_id: int, session: Session = Depends(get_session)):
    patient = _get_public_patient(patient_id, session)
    return _patient_to_read(patient, session)


def _studies_to_read_batch(studies: List[Study], session: Session) -> List[StudyRead]:
    """Batch version of _study_to_read -- O(1) queries instead of O(4N).
    This was the dominant cost of /api/studies (23.6s for 76 rows): the old
    per-study StudyImage query selected the FULL row including .data (the
    actual image bytes, a LargeBinary column) just to read a filename, on
    top of one query each for prediction/patient/reviewer per study."""
    if not studies:
        return []
    study_ids = [s.id for s in studies]

    images_by_study: Dict[int, Dict[str, int]] = {}
    for study_id, image_id, name in session.exec(
        select(StudyImage.study_id, StudyImage.id, StudyImage.name).where(StudyImage.study_id.in_(study_ids))
    ).all():
        images_by_study.setdefault(study_id, {})[name] = image_id

    preds_by_study: Dict[int, Prediction] = {
        pred.study_id: pred
        for pred in session.exec(select(Prediction).where(Prediction.study_id.in_(study_ids))).all()
    }

    patient_ids = {s.patient_id for s in studies if s.patient_id}
    name_by_patient: Dict[int, str] = {}
    if patient_ids:
        name_by_patient = dict(
            session.exec(select(Patient.id, Patient.name).where(Patient.id.in_(patient_ids))).all()
        )

    reviewer_ids = {s.reviewed_by_user_id for s in studies if s.reviewed_by_user_id}
    email_by_reviewer: Dict[int, str] = {}
    if reviewer_ids:
        email_by_reviewer = dict(
            session.exec(select(User.id, User.email).where(User.id.in_(reviewer_ids))).all()
        )

    result = []
    for study in studies:
        imgs = images_by_study.get(study.id, {})

        def url(name: str, _imgs=imgs) -> Optional[str]:
            return f"/api/image/{_imgs[name]}" if name in _imgs else None

        pred = preds_by_study.get(study.id)
        prediction = None
        if pred:
            prediction = PredictionRead(
                label=pred.label,
                confidence=pred.confidence,
                probabilities=json.loads(pred.probabilities),
                backbone=pred.backbone,
                heatmap_url=url("gradcam"),
                per_model=json.loads(pred.per_model) if pred.per_model else None,
            )
        result.append(StudyRead(
            id=study.id,
            patient_id=study.patient_id,
            patient_name=name_by_patient.get(study.patient_id) if study.patient_id else None,
            modality=study.modality,
            uploaded_at=study.uploaded_at,
            quality_passed=study.quality_passed,
            quality_score=study.quality_score,
            analysis_stopped=bool(study.analysis_stopped),
            model_version=study.model_version,
            num_rois=study.num_rois,
            image_url=url("original") or url("rois") or "",
            annotated_url=url("rois"),
            prediction=prediction,
            org_id=study.org_id,
            review_status=study.review_status,
            reviewed_by=email_by_reviewer.get(study.reviewed_by_user_id) if study.reviewed_by_user_id else None,
            review_note=study.review_note,
            reviewed_at=study.reviewed_at,
        ))
    return result


def _study_to_read(study: Study, session: Session) -> StudyRead:
    return _studies_to_read_batch([study], session)[0]


@router.get("/{patient_id}/timeline", response_model=List[StudyRead])
def patient_timeline(patient_id: int, session: Session = Depends(get_session)):
    _get_public_patient(patient_id, session)
    studies = session.exec(
        select(Study)
        .where(Study.patient_id == patient_id, Study.org_id.is_(None))
        .order_by(Study.uploaded_at)
    ).all()
    return _studies_to_read_batch(studies, session)
