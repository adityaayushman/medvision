
from __future__ import annotations

import json
from typing import List

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


def _patient_to_read(patient: Patient, session: Session) -> PatientRead:
    studies = session.exec(
        select(Study).where(Study.patient_id == patient.id).order_by(Study.uploaded_at.desc())
    ).all()
    last_label = None
    if studies:
        pred = session.exec(select(Prediction).where(Prediction.study_id == studies[0].id)).first()
        last_label = pred.label if pred else None
    return PatientRead(
        id=patient.id,
        name=patient.name,
        sex=patient.sex,
        birth_year=patient.birth_year,
        created_at=patient.created_at,
        study_count=len(studies),
        last_study_at=studies[0].uploaded_at if studies else None,
        last_label=last_label,
    )


@router.get("", response_model=List[PatientRead])
def list_patients(session: Session = Depends(get_session)):
    patients = session.exec(
        select(Patient).where(Patient.org_id.is_(None)).order_by(Patient.created_at.desc())
    ).all()
    return [_patient_to_read(p, session) for p in patients]


def _get_public_patient(patient_id: int, session: Session) -> Patient:
    patient = session.get(Patient, patient_id)
    if not patient or patient.org_id is not None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.get("/{patient_id}", response_model=PatientRead)
def get_patient(patient_id: int, session: Session = Depends(get_session)):
    patient = _get_public_patient(patient_id, session)
    return _patient_to_read(patient, session)


def _study_to_read(study: Study, session: Session) -> StudyRead:
    imgs = {
        i.name: i.id
        for i in session.exec(select(StudyImage).where(StudyImage.study_id == study.id)).all()
    }

    def url(name: str):
        return f"/api/image/{imgs[name]}" if name in imgs else None

    pred = session.exec(select(Prediction).where(Prediction.study_id == study.id)).first()
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
    patient_name = None
    if study.patient_id:
        patient = session.get(Patient, study.patient_id)
        patient_name = patient.name if patient else None
    reviewed_by = None
    if study.reviewed_by_user_id:
        reviewer = session.get(User, study.reviewed_by_user_id)
        reviewed_by = reviewer.email if reviewer else None
    return StudyRead(
        id=study.id,
        patient_id=study.patient_id,
        patient_name=patient_name,
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
        reviewed_by=reviewed_by,
        review_note=study.review_note,
        reviewed_at=study.reviewed_at,
    )


@router.get("/{patient_id}/timeline", response_model=List[StudyRead])
def patient_timeline(patient_id: int, session: Session = Depends(get_session)):
    _get_public_patient(patient_id, session)
    studies = session.exec(
        select(Study)
        .where(Study.patient_id == patient_id, Study.org_id.is_(None))
        .order_by(Study.uploaded_at)
    ).all()
    return [_study_to_read(s, session) for s in studies]
