"""All analyzed studies — the global record of every scan run through /analyze,
whether or not it was attached to a patient. This is what the frontend's
Records page lists."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from ..db import get_session
from ..models_db import Patient, Study
from ..schemas import StudyRead
from .patients import _study_to_read

router = APIRouter(prefix="/api/studies", tags=["studies"])


class AssignPatient(BaseModel):
    patient_id: int


@router.get("", response_model=List[StudyRead])
def list_studies(limit: int = 200, session: Session = Depends(get_session)):
    """Every analyzed scan, newest first (across all patients + unassigned)."""
    studies = session.exec(
        select(Study).order_by(Study.uploaded_at.desc()).limit(limit)
    ).all()
    return [_study_to_read(s, session) for s in studies]


@router.get("/count")
def study_count(session: Session = Depends(get_session)) -> dict:
    return {"count": len(session.exec(select(Study)).all())}


@router.patch("/{study_id}/patient", response_model=StudyRead)
def assign_patient(study_id: int, body: AssignPatient, session: Session = Depends(get_session)):
    """Attach an existing (e.g. previously unassigned) study to a patient."""
    study = session.get(Study, study_id)
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")
    if not session.get(Patient, body.patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    study.patient_id = body.patient_id
    session.add(study)
    session.commit()
    session.refresh(study)
    return _study_to_read(study, session)
