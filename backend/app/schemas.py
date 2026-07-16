
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel

ReviewStatus = Literal["pending", "in-review", "reviewed", "flagged"]


class PatientCreate(BaseModel):
    name: str
    sex: Optional[str] = None
    birth_year: Optional[int] = None


class PatientRead(BaseModel):
    id: int
    name: str
    sex: Optional[str] = None
    birth_year: Optional[int] = None
    created_at: datetime
    study_count: int = 0
    last_study_at: Optional[datetime] = None
    last_label: Optional[str] = None


class PredictionRead(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]
    backbone: str = ""
    heatmap_url: Optional[str] = None
    per_model: Optional[List[Dict]] = None


class StudyRead(BaseModel):
    id: int
    patient_id: Optional[int]
    patient_name: Optional[str] = None
    modality: str
    uploaded_at: datetime
    quality_passed: bool
    quality_score: Optional[int] = None
    analysis_stopped: bool = False
    model_version: Optional[str] = None
    num_rois: int
    image_url: str
    annotated_url: Optional[str] = None
    prediction: Optional[PredictionRead] = None
    org_id: Optional[int] = None
    review_status: Optional[str] = None
    reviewed_by: Optional[str] = None
    review_note: Optional[str] = None
    reviewed_at: Optional[datetime] = None


class AnalyzeResponse(BaseModel):
    study_id: Optional[int] = None
    modality: str
    model_loaded: bool
    quality: Dict
    num_rois: int
    rois: List[Dict]
    prediction: Optional[Dict] = None
    image_url: str
    annotated_url: str
    heatmap_url: Optional[str] = None
    stages: List[Dict] = []
    analysis_stopped: bool = False
    pipeline_steps: List[Dict] = []
    processing_metadata: Dict = {}


class ReportQuality(BaseModel):
    passed: bool
    score: Optional[int] = None
    reasons: List[str] = []


class ReportPrediction(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]
    backbone: str = ""
    per_model: Optional[List[Dict]] = None


class ReportPatient(BaseModel):
    id: int
    name: str
    sex: Optional[str] = None
    birth_year: Optional[int] = None


class ReportRead(BaseModel):
    study_id: int
    generated_at: datetime
    modality: str
    modality_label: str
    uploaded_at: datetime
    patient: Optional[ReportPatient] = None
    quality: ReportQuality
    num_rois: int
    analysis_stopped: bool
    model_version: Optional[str] = None
    processing_time_ms: Optional[float] = None
    inference_time_ms: Optional[float] = None
    prediction: Optional[ReportPrediction] = None
    disclaimer: str


class OrgSignup(BaseModel):
    org_name: str
    email: str
    password: str
    name: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class UserCreate(BaseModel):
    email: str
    password: str
    role: str = "radiologist"
    name: Optional[str] = None


class UserRead(BaseModel):
    id: int
    org_id: int
    email: str
    role: str
    name: Optional[str] = None
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserRead


class ReviewStatusUpdate(BaseModel):
    review_status: ReviewStatus
    note: Optional[str] = None


class AuditLogRead(BaseModel):
    id: int
    actor_user_id: int
    actor_email: Optional[str] = None
    action: str
    target_type: str
    target_id: int
    meta: Optional[Dict] = None
    created_at: datetime


class DatasetSpecRead(BaseModel):
    key: str
    name: str
    modality: str
    task: str
    access: str
    roi_support: bool
    approx_images: str
    url: str
    notes: str = ""
    recommended_for: List[str] = []
