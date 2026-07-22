
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import Session, select

from ..audit import log_action
from ..auth import get_current_user, require_role
from ..db import get_session
from ..ml import AnalyzerService, get_analyzer
from ..models_db import AuditLog, Patient, Study, User
from ..routers.inference import _analyze_and_persist
from ..routers.patients import _patient_to_read, _patients_to_read_batch, _studies_to_read_batch, _study_to_read
from ..schemas import (
    AnalyzeResponse,
    AuditLogRead,
    PatientCreate,
    PatientRead,
    ReviewStatusUpdate,
    StudyRead,
)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


class AssignPatient(BaseModel):
    patient_id: int


def _get_org_patient(patient_id: int, org_id: int, session: Session) -> Patient:
    patient = session.get(Patient, patient_id)
    if not patient or patient.org_id != org_id:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


def _get_org_study(study_id: int, org_id: int, session: Session) -> Study:
    study = session.get(Study, study_id)
    if not study or study.org_id != org_id:
        raise HTTPException(status_code=404, detail="Study not found")
    return study


@router.post("/patients", response_model=PatientRead)
def create_patient(
    body: PatientCreate,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    patient = Patient(**body.model_dump(), org_id=user.org_id)
    session.add(patient)
    session.commit()
    session.refresh(patient)
    log_action(session, user.org_id, user.id, "patient.created", "patient", patient.id)
    return _patient_to_read(patient, session)


@router.get("/patients", response_model=List[PatientRead])
def list_patients(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    patients = session.exec(
        select(Patient).where(Patient.org_id == user.org_id).order_by(Patient.created_at.desc())
    ).all()
    return _patients_to_read_batch(patients, session)


@router.get("/patients/{patient_id}", response_model=PatientRead)
def get_patient(
    patient_id: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    patient = _get_org_patient(patient_id, user.org_id, session)
    return _patient_to_read(patient, session)


@router.get("/patients/{patient_id}/timeline", response_model=List[StudyRead])
def patient_timeline(
    patient_id: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    _get_org_patient(patient_id, user.org_id, session)
    studies = session.exec(
        select(Study).where(Study.patient_id == patient_id, Study.org_id == user.org_id)
        .order_by(Study.uploaded_at)
    ).all()
    return _studies_to_read_batch(studies, session)


@router.post("/studies/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    patient_id: Optional[int] = Form(None),
    modality: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
    analyzer: AnalyzerService = Depends(get_analyzer),
    session: Session = Depends(get_session),
) -> AnalyzeResponse:
    if patient_id is not None:
        _get_org_patient(patient_id, user.org_id, session)
    result = _analyze_and_persist(await file.read(), patient_id, modality, user.org_id, analyzer, session)
    log_action(session, user.org_id, user.id, "study.created", "study", result.study_id)
    return result


@router.get("/studies", response_model=List[StudyRead])
def list_studies(
    review_status: Optional[str] = None,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    stmt = select(Study).where(Study.org_id == user.org_id)
    if review_status:
        stmt = stmt.where(Study.review_status == review_status)
    studies = session.exec(stmt.order_by(Study.uploaded_at.desc())).all()
    return _studies_to_read_batch(studies, session)


@router.get("/studies/{study_id}", response_model=StudyRead)
def get_study(
    study_id: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    study = _get_org_study(study_id, user.org_id, session)
    log_action(session, user.org_id, user.id, "study.viewed", "study", study.id)
    return _study_to_read(study, session)


@router.patch("/studies/{study_id}/patient", response_model=StudyRead)
def assign_patient(
    study_id: int,
    body: AssignPatient,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    study = _get_org_study(study_id, user.org_id, session)
    _get_org_patient(body.patient_id, user.org_id, session)
    study.patient_id = body.patient_id
    session.add(study)
    session.commit()
    session.refresh(study)
    log_action(session, user.org_id, user.id, "study.patient_assigned", "study", study.id)
    return _study_to_read(study, session)


@router.patch("/studies/{study_id}/review", response_model=StudyRead)
def update_review_status(
    study_id: int,
    body: ReviewStatusUpdate,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    study = _get_org_study(study_id, user.org_id, session)
    from_status = study.review_status
    study.review_status = body.review_status
    study.reviewed_by_user_id = user.id
    study.review_note = body.note
    study.reviewed_at = datetime.now(timezone.utc)
    session.add(study)
    session.commit()
    session.refresh(study)
    log_action(
        session,
        user.org_id,
        user.id,
        "study.review_status_changed",
        "study",
        study.id,
        meta={"from": from_status, "to": body.review_status, "note": body.note},
    )
    return _study_to_read(study, session)


@router.get("/audit", response_model=List[AuditLogRead])
def list_audit_log(
    admin: User = Depends(require_role("admin")),
    session: Session = Depends(get_session),
):
    entries = session.exec(
        select(AuditLog).where(AuditLog.org_id == admin.org_id).order_by(AuditLog.created_at.desc())
    ).all()
    users = {u.id: u.email for u in session.exec(select(User).where(User.org_id == admin.org_id)).all()}
    return [
        AuditLogRead(
            id=e.id,
            actor_user_id=e.actor_user_id,
            actor_email=users.get(e.actor_user_id),
            action=e.action,
            target_type=e.target_type,
            target_id=e.target_id,
            meta=json.loads(e.meta) if e.meta else None,
            created_at=e.created_at,
        )
        for e in entries
    ]
