
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import Column, LargeBinary
from sqlmodel import Field, Relationship, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Organization(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    created_at: datetime = Field(default_factory=_utcnow)


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    org_id: int = Field(foreign_key="organization.id", index=True)
    email: str = Field(index=True, unique=True)
    password_hash: str
    role: str = "radiologist"
    name: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)


class AuditLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    org_id: int = Field(index=True)
    actor_user_id: int = Field(index=True)
    action: str
    target_type: str
    target_id: int
    meta: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow, index=True)


class Patient(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    sex: Optional[str] = None
    birth_year: Optional[int] = None
    created_at: datetime = Field(default_factory=_utcnow)

    org_id: Optional[int] = Field(default=None, foreign_key="organization.id", index=True)

    studies: List["Study"] = Relationship(back_populates="patient")


class Study(SQLModel, table=True):

    id: Optional[int] = Field(default=None, primary_key=True)
    patient_id: Optional[int] = Field(default=None, foreign_key="patient.id", index=True)
    modality: str = "chest_xray"
    image_path: str
    uploaded_at: datetime = Field(default_factory=_utcnow)

    quality_passed: bool = True
    quality_reasons: str = ""
    num_rois: int = 0

    quality_score: Optional[int] = None
    analysis_stopped: Optional[bool] = False
    model_version: Optional[str] = None
    processing_time_ms: Optional[float] = None
    inference_time_ms: Optional[float] = None
    segmentation_success: Optional[bool] = None

    org_id: Optional[int] = Field(default=None, foreign_key="organization.id", index=True)
    review_status: Optional[str] = None
    reviewed_by_user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    review_note: Optional[str] = None
    reviewed_at: Optional[datetime] = None

    patient: Optional[Patient] = Relationship(back_populates="studies")
    prediction: Optional["Prediction"] = Relationship(back_populates="study")


class StudyImage(SQLModel, table=True):

    id: Optional[int] = Field(default=None, primary_key=True)
    study_id: Optional[int] = Field(default=None, foreign_key="study.id", index=True)
    name: str = Field(index=True)
    data: bytes = Field(sa_column=Column(LargeBinary))
    created_at: datetime = Field(default_factory=_utcnow)


class Prediction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    study_id: Optional[int] = Field(default=None, foreign_key="study.id", index=True)
    label: str
    confidence: float
    probabilities: str = "{}"
    backbone: str = ""
    heatmap_path: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)

    study: Optional[Study] = Relationship(back_populates="prediction")
