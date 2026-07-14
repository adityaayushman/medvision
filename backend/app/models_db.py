"""Database tables — the longitudinal patient record ("Digital Twin").

A Patient has many Studies (scans over time); each Study has one Prediction.
This is what turns MedChron from a one-shot classifier into a monitoring tool.

NOTE: no ``from __future__ import annotations`` here — it stringifies the
Relationship type hints and breaks SQLModel's forward-reference resolution.
"""

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import Column, LargeBinary
from sqlmodel import Field, Relationship, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Patient(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    sex: Optional[str] = None
    birth_year: Optional[int] = None
    created_at: datetime = Field(default_factory=_utcnow)

    studies: List["Study"] = Relationship(back_populates="patient")


class Study(SQLModel, table=True):
    """One uploaded scan and its preprocessing outcome."""

    id: Optional[int] = Field(default=None, primary_key=True)
    patient_id: Optional[int] = Field(default=None, foreign_key="patient.id", index=True)
    modality: str = "chest_xray"
    image_path: str
    uploaded_at: datetime = Field(default_factory=_utcnow)

    quality_passed: bool = True
    quality_reasons: str = ""     # semicolon-joined
    num_rois: int = 0

    # research-grade processing metadata (nullable: absent on studies analyzed
    # before these fields existed — every field here must be Optional, since the
    # live column was added via ALTER TABLE without NOT NULL and can hold NULL
    # on pre-existing rows)
    quality_score: Optional[int] = None
    analysis_stopped: Optional[bool] = False
    model_version: Optional[str] = None
    processing_time_ms: Optional[float] = None
    inference_time_ms: Optional[float] = None
    segmentation_success: Optional[bool] = None

    patient: Optional[Patient] = Relationship(back_populates="studies")
    prediction: Optional["Prediction"] = Relationship(back_populates="study")


class StudyImage(SQLModel, table=True):
    """A pipeline-stage image stored in the DB (bytes), so it survives restarts
    on hosts with an ephemeral disk. Served via GET /api/image/{id}."""

    id: Optional[int] = Field(default=None, primary_key=True)
    study_id: Optional[int] = Field(default=None, foreign_key="study.id", index=True)
    name: str = Field(index=True)  # original | enhanced | segmentation | rois | gradcam
    data: bytes = Field(sa_column=Column(LargeBinary))
    created_at: datetime = Field(default_factory=_utcnow)


class Prediction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    study_id: Optional[int] = Field(default=None, foreign_key="study.id", index=True)
    label: str
    confidence: float
    probabilities: str = "{}"     # JSON string
    backbone: str = ""
    heatmap_path: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)

    study: Optional[Study] = Relationship(back_populates="prediction")
