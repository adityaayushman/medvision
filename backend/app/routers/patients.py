"""Patient records + per-patient study timeline (the Digital Twin surface)."""

from __future__ import annotations

import json
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..db import get_session
from ..models_db import Patient, Prediction, Study, StudyImage
from ..schemas import PatientCreate, PatientRead, PredictionRead, StudyRead

router = APIRouter(prefix="/api/patients", tags=["patients"])


@router.post("", response_model=PatientRead)
def create_patient(body: PatientCreate, session: Session = Depends(get_session)) -> Patient:
    patient = Patient(**body.model_dump())
    session.add(patient)
    session.commit()
    session.refresh(patient)
    return patient


@router.get("", response_model=List[PatientRead])
def list_patients(session: Session = Depends(get_session)):
    return session.exec(select(Patient).order_by(Patient.created_at.desc())).all()


@router.get("/{patient_id}", response_model=PatientRead)
def get_patient(patient_id: int, session: Session = Depends(get_session)) -> Patient:
    patient = session.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


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
        )
    return StudyRead(
        id=study.id,
        patient_id=study.patient_id,
        modality=study.modality,
        uploaded_at=study.uploaded_at,
        quality_passed=study.quality_passed,
        num_rois=study.num_rois,
        image_url=url("original") or url("rois") or "",
        annotated_url=url("rois"),
        prediction=prediction,
    )


@router.get("/{patient_id}/timeline", response_model=List[StudyRead])
def patient_timeline(patient_id: int, session: Session = Depends(get_session)):
    """Chronological studies for a patient — the disease-monitoring view."""
    if not session.get(Patient, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    studies = session.exec(
        select(Study).where(Study.patient_id == patient_id).order_by(Study.uploaded_at)
    ).all()
    return [_study_to_read(s, session) for s in studies]
