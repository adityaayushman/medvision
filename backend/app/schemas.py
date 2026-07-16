
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel


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
