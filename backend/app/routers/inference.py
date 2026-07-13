"""/api/analyze — upload a scan, run the pipeline (+model), persist a Study."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlmodel import Session

from ..config import settings
from ..db import get_session
from ..ml import AnalyzerService, get_analyzer
from ..models_db import Prediction, Study
from ..schemas import AnalyzeResponse

router = APIRouter(prefix="/api", tags=["inference"])


def _decode(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image file.")
    return image


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    patient_id: Optional[int] = Form(None),
    analyzer: AnalyzerService = Depends(get_analyzer),
    session: Session = Depends(get_session),
) -> AnalyzeResponse:
    image = _decode(await file.read())
    payload, annotated, overlay = analyzer.analyze(image)

    # persist images to static storage
    uid = uuid.uuid4().hex[:12]
    upload_path = settings.storage_dir / "uploads" / f"{uid}.png"
    annotated_path = settings.storage_dir / "overlays" / f"{uid}_annotated.png"
    cv2.imwrite(str(upload_path), image)
    cv2.imwrite(str(annotated_path), annotated)

    heatmap_url = None
    heatmap_path = None
    if overlay is not None:
        heatmap_path = settings.storage_dir / "overlays" / f"{uid}_gradcam.png"
        cv2.imwrite(str(heatmap_path), overlay)
        heatmap_url = f"/static/overlays/{heatmap_path.name}"

    # persist the Study (+ Prediction) — the longitudinal record
    study = Study(
        patient_id=patient_id,
        modality=payload["modality"],
        image_path=str(upload_path),
        quality_passed=payload["quality"]["passed"],
        quality_reasons=";".join(payload["quality"]["reasons"]),
        num_rois=payload["num_rois"],
    )
    session.add(study)
    session.commit()
    session.refresh(study)

    pred = payload.get("prediction")
    if pred:
        session.add(Prediction(
            study_id=study.id,
            label=pred["label"],
            confidence=pred["confidence"],
            probabilities=json.dumps(pred["probabilities"]),
            backbone=pred.get("backbone", ""),
            heatmap_path=str(heatmap_path) if heatmap_path else None,
        ))
        session.commit()

    return AnalyzeResponse(
        study_id=study.id,
        modality=payload["modality"],
        model_loaded=payload["model_loaded"],
        quality=payload["quality"],
        num_rois=payload["num_rois"],
        rois=payload["rois"],
        prediction=pred,
        image_url=f"/static/uploads/{upload_path.name}",
        annotated_url=f"/static/overlays/{annotated_path.name}",
        heatmap_url=heatmap_url,
    )
