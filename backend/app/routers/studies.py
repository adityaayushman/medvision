"""All analyzed studies — the global record of every scan run through /analyze,
whether or not it was attached to a patient. This is what the frontend's
Records page lists."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from ..db import get_session
from ..models_db import Study
from ..schemas import StudyRead
from .patients import _study_to_read

router = APIRouter(prefix="/api/studies", tags=["studies"])


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
