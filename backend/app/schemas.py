"""Pydantic request/response models for the API."""

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


class PredictionRead(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]
    backbone: str = ""
    heatmap_url: Optional[str] = None


class StudyRead(BaseModel):
    id: int
    patient_id: Optional[int]
    modality: str
    uploaded_at: datetime
    quality_passed: bool
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
