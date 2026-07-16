
from __future__ import annotations

import json
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlmodel import Session

from ..db import get_session
from ..ml import AnalyzerService, get_analyzer
from ..models_db import Prediction, Study, StudyImage
from ..schemas import AnalyzeResponse

router = APIRouter(prefix="/api", tags=["inference"])

STAGE_LABELS = {
    "original": "Original",
    "enhanced": "Enhanced (CLAHE)",
    "segmentation": "Segmentation",
    "rois": "ROIs",
    "gradcam": "Grad-CAM",
}


def _decode(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image file.")
    return image


def _png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image.")
    return buf.tobytes()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    patient_id: Optional[int] = Form(None),
    modality: Optional[str] = Form(None),
    analyzer: AnalyzerService = Depends(get_analyzer),
    session: Session = Depends(get_session),
) -> AnalyzeResponse:
    image = _decode(await file.read())
    payload, result, overlay = analyzer.analyze(image, modality=modality)

    meta = payload["processing_metadata"]

    study = Study(
        patient_id=patient_id,
        modality=payload["modality"],
        image_path="",
        quality_passed=payload["quality"]["passed"],
        quality_reasons=";".join(payload["quality"]["reasons"]),
        num_rois=payload["num_rois"],
        quality_score=payload["quality"]["overall_score"],
        analysis_stopped=payload["analysis_stopped"],
        model_version=meta.get("model_version"),
        processing_time_ms=meta.get("processing_time_ms"),
        inference_time_ms=meta.get("inference_time_ms"),
        segmentation_success=meta.get("segmentation_success"),
    )
    session.add(study)
    session.commit()
    session.refresh(study)

    stage_arrays = [
        ("original", result.original),
        ("enhanced", result.enhanced),
        ("segmentation", result.cleaned_mask),
        ("rois", result.annotated),
    ]
    if overlay is not None:
        stage_arrays.append(("gradcam", overlay))

    ids: dict[str, int] = {}
    for name, arr in stage_arrays:
        si = StudyImage(study_id=study.id, name=name, data=_png(arr))
        session.add(si)
        session.commit()
        session.refresh(si)
        ids[name] = si.id

    def url(name: str) -> Optional[str]:
        return f"/api/image/{ids[name]}" if name in ids else None

    stages = [{"name": STAGE_LABELS[name], "url": url(name)} for name, _ in stage_arrays]
    heatmap_url = url("gradcam")

    pred = payload.get("prediction")
    if pred:
        session.add(Prediction(
            study_id=study.id,
            label=pred["label"],
            confidence=pred["confidence"],
            probabilities=json.dumps(pred["probabilities"]),
            backbone=pred.get("backbone", ""),
            heatmap_path=heatmap_url,
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
        image_url=url("original") or "",
        annotated_url=url("rois") or "",
        heatmap_url=heatmap_url,
        stages=stages,
        analysis_stopped=payload["analysis_stopped"],
        pipeline_steps=payload["pipeline_steps"],
        processing_metadata=meta,
    )
