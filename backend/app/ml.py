"""Bridge between the FastAPI app and the medchron ML core.

Runs the DIP pipeline on every image (quality + ROI), and — only if a trained
checkpoint exists — adds classification + Grad-CAM. Torch is imported lazily, so
the API still serves the preprocessing demo before any model is trained.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from medchron import MedicalImagePipeline, get_config

from .config import settings


class AnalyzerService:
    def __init__(self) -> None:
        self.pipeline = MedicalImagePipeline(get_config(settings.modality))
        self.predictor = self._maybe_load_predictor()

    @staticmethod
    def _maybe_load_predictor():
        ckpt = Path(settings.model_checkpoint)
        if not ckpt.exists():
            print(f"[ml] No checkpoint at {ckpt} — running in preprocess-only mode.")
            return None
        try:
            from medchron.models import Predictor  # imports torch lazily
            print(f"[ml] Loading model checkpoint {ckpt}")
            return Predictor(str(ckpt))
        except Exception as exc:  # torch missing / bad checkpoint
            print(f"[ml] Could not load model ({exc}); preprocess-only mode.")
            return None

    @property
    def model_loaded(self) -> bool:
        return self.predictor is not None

    def analyze(
        self, image_bgr: np.ndarray
    ) -> Tuple[Dict, np.ndarray, Optional[np.ndarray]]:
        """Return (payload, annotated_image, gradcam_overlay_or_None)."""
        result = self.pipeline.run(image_bgr)
        payload: Dict = {
            "modality": settings.modality,
            "quality": result.quality.to_dict(),
            "num_rois": len(result.rois),
            "rois": [r.to_dict() for r in result.rois],
            "prediction": None,
            "model_loaded": self.model_loaded,
        }

        overlay = None
        if self.predictor is not None:
            overlay, prediction = self.predictor.explain(image_bgr)
            prediction["backbone"] = self.predictor.model_config.backbone
            payload["prediction"] = prediction

        return payload, result.annotated, overlay


@lru_cache(maxsize=1)
def get_analyzer() -> AnalyzerService:
    """Process-wide singleton (model loaded once)."""
    return AnalyzerService()
