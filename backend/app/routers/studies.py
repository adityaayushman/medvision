
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlmodel import Session, select

from ..db import get_session
from ..models_db import Patient, Study
from ..reports import build_report_data, render_report_pdf
from ..schemas import ReportRead, StudyRead
from .patients import _study_to_read

router = APIRouter(prefix="/api/studies", tags=["studies"])


class AssignPatient(BaseModel):
    patient_id: int


@router.get("", response_model=List[StudyRead])
def list_studies(limit: int = 200, session: Session = Depends(get_session)):
    studies = session.exec(
        select(Study).where(Study.org_id.is_(None)).order_by(Study.uploaded_at.desc()).limit(limit)
    ).all()
    return [_study_to_read(s, session) for s in studies]


@router.get("/count")
def study_count(session: Session = Depends(get_session)) -> dict:
    return {"count": len(session.exec(select(Study).where(Study.org_id.is_(None))).all())}


def _get_public_study(study_id: int, session: Session) -> Study:
    study = session.get(Study, study_id)
    if not study or study.org_id is not None:
        raise HTTPException(status_code=404, detail="Study not found")
    return study


@router.get("/{study_id}", response_model=StudyRead)
def get_study(study_id: int, session: Session = Depends(get_session)):
    study = _get_public_study(study_id, session)
    return _study_to_read(study, session)


@router.patch("/{study_id}/patient", response_model=StudyRead)
def assign_patient(study_id: int, body: AssignPatient, session: Session = Depends(get_session)):
    study = _get_public_study(study_id, session)
    patient = session.get(Patient, body.patient_id)
    if not patient or patient.org_id is not None:
        raise HTTPException(status_code=404, detail="Patient not found")
    study.patient_id = body.patient_id
    session.add(study)
    session.commit()
    session.refresh(study)
    return _study_to_read(study, session)


@router.get("/{study_id}/report", response_model=ReportRead)
def get_report(study_id: int, session: Session = Depends(get_session)):
    study = _get_public_study(study_id, session)
    return build_report_data(study, session)


@router.get("/{study_id}/report.pdf")
def get_report_pdf(study_id: int, session: Session = Depends(get_session)) -> Response:
    study = _get_public_study(study_id, session)
    report = build_report_data(study, session)
    pdf_bytes = render_report_pdf(report, session)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="medchron_report_study_{study_id}.pdf"'},
    )
