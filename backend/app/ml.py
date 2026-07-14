"""Bridge between the FastAPI app and the medchron ML core.

Runs the DIP pipeline on every image (quality + ROI). Classification + Grad-CAM
only run if a trained checkpoint exists for the requested modality *and* the
image clears the quality gate — a scan that's too blurry/dark/washed-out is not
reliable to classify, so analysis stops there rather than returning a
prediction nobody should trust. Torch is imported lazily, so the API still
serves the preprocessing demo before any model is trained.

Multiple modalities (chest X-ray, brain MRI, ...) each get their own pipeline
+ optional predictor, loaded once at startup. A modality with no configured
checkpoint simply runs preprocess-only — exactly how the single-modality
service always behaved, so existing chest-X-ray callers are unaffected.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from medchron import MedicalImagePipeline, get_config
from medchron.config import PreprocessConfig
from medchron.imaging.pipeline import PipelineResult
from medchron.imaging.roi import ROI

from .config import settings

# modality -> checkpoint path. A modality with an empty/missing path just runs
# preprocess-only. Add a new modality here + a PRESETS entry in medchron.config
# to onboard it — nothing else in this file changes.
MODALITY_CHECKPOINTS: Dict[str, str] = {
    "chest_xray": settings.model_checkpoint,
    "brain_mri": settings.model_checkpoint_brain_mri,
    "mammography": settings.model_checkpoint_mammography,
}


def _preprocessing_ops(cfg: PreprocessConfig) -> List[str]:
    """Human-readable list of the DIP operations actually applied, from the
    same config object the pipeline ran with — never hand-typed / guessed."""
    ops = [f"Denoise: {cfg.denoise_method} ({cfg.denoise_ksize}x{cfg.denoise_ksize})"]
    if cfg.use_hist_eq:
        ops.append(f"CLAHE: clip={cfg.clahe_clip}, tile={cfg.clahe_tile}x{cfg.clahe_tile}")
    ops.append(f"Segmentation: {cfg.threshold_method} thresholding")
    if cfg.morph_op != "none":
        ops.append(f"Morphology: {cfg.morph_op} ({cfg.morph_ksize}x{cfg.morph_ksize}, x{cfg.morph_iterations})")
    ops.append(f"ROI extraction: contour-based, min area {cfg.min_roi_area_ratio * 100:.1f}% of image")
    return ops


def _roi_confidence(rois: List[ROI], cfg: PreprocessConfig) -> Optional[Dict]:
    """Heuristic confidence per ROI, based on how far its area is above the
    minimum-area threshold. This is explicitly NOT a learned/calibrated
    confidence (contour-based ROI extraction has none) — it's a transparent,
    documented proxy so the UI never implies more certainty than exists."""
    if not rois:
        return None
    img_area = cfg.target_size[0] * cfg.target_size[1]
    min_area = cfg.min_roi_area_ratio * img_area
    per_roi = []
    for r in rois:
        ratio = (r.area / min_area) if min_area > 0 else 1.0
        conf = max(0.3, min(1.0, 0.3 + 0.7 * min(1.0, (ratio - 1.0) / 4.0)))
        per_roi.append(round(conf, 2))
    overall = round(sum(per_roi) / len(per_roi), 2)
    label = "High" if overall >= 0.75 else ("Medium" if overall >= 0.5 else "Low")
    return {
        "overall_score": overall,
        "overall_label": label,
        "per_roi": per_roi,
        "note": "Heuristic — area relative to the minimum-ROI-size threshold, not a learned confidence.",
    }


def _pipeline_steps(quality_passed: bool, model_loaded: bool) -> List[Dict]:
    def step(name: str, status: str, detail: str = "") -> Dict:
        return {"name": name, "status": status, "detail": detail}

    steps = [
        step("Upload", "done"),
        step("Quality Assessment", "done"),
        step("CLAHE Enhancement", "done"),
        step("Morphological Cleanup", "done"),
        step("Segmentation", "done"),
        step("ROI Extraction", "done"),
    ]
    if not quality_passed:
        steps.append(step("Classification", "stopped", "Quality gate: image quality too low"))
        steps.append(step("Grad-CAM", "stopped", "No prediction to explain"))
    elif not model_loaded:
        steps.append(step("Classification", "skipped", "No trained model loaded"))
        steps.append(step("Grad-CAM", "skipped", "No trained model loaded"))
    else:
        steps.append(step("Classification", "done"))
        steps.append(step("Grad-CAM", "done"))
    steps.append(step("Report", "done"))
    return steps


class AnalyzerService:
    def __init__(self) -> None:
        self.configs: Dict[str, PreprocessConfig] = {}
        self.pipelines: Dict[str, MedicalImagePipeline] = {}
        self.predictors: Dict[str, Optional[object]] = {}
        for modality, ckpt_path in MODALITY_CHECKPOINTS.items():
            cfg = get_config(modality)
            self.configs[modality] = cfg
            self.pipelines[modality] = MedicalImagePipeline(cfg)
            self.predictors[modality] = self._maybe_load_predictor(ckpt_path, modality)

    @staticmethod
    def _maybe_load_predictor(ckpt_path: str, modality: str):
        if not ckpt_path:
            return None
        ckpt = Path(ckpt_path)
        if not ckpt.exists():
            print(f"[ml] No checkpoint at {ckpt} for modality={modality!r} — preprocess-only for this modality.")
            return None
        try:
            from medchron.models import Predictor  # imports torch lazily
            print(f"[ml] Loading {modality} checkpoint {ckpt}")
            return Predictor(str(ckpt))
        except Exception as exc:  # torch missing / bad checkpoint
            print(f"[ml] Could not load {modality} model ({exc}); preprocess-only for this modality.")
            return None

    def available_modalities(self) -> Dict[str, bool]:
        """modality -> whether a trained model is loaded for it."""
        return {m: (p is not None) for m, p in self.predictors.items()}

    @property
    def model_loaded(self) -> bool:
        """Backward-compat single flag: is the *default* modality's model loaded."""
        return self.predictors.get(settings.modality) is not None

    def analyze(self, image_bgr: np.ndarray, modality: Optional[str] = None):
        """Return (payload, PipelineResult, gradcam_overlay_or_None).

        The PipelineResult carries every DIP stage (enhanced, segmentation mask,
        annotated ROIs, ...) so the API can expose the full processing gallery.
        Unknown/omitted modality falls back to the configured default, so every
        caller from before v2 (which never sent a modality) behaves identically.
        """
        modality = modality if modality in self.pipelines else settings.modality
        pipeline = self.pipelines[modality]
        predictor = self.predictors[modality]
        config = self.configs[modality]

        t0 = time.perf_counter()
        result: PipelineResult = pipeline.run(image_bgr)
        processing_time_ms = round((time.perf_counter() - t0) * 1000, 1)

        fg_ratio = float(np.count_nonzero(result.cleaned_mask)) / result.cleaned_mask.size
        segmentation_success = 0.0 < fg_ratio < 0.98
        model_loaded = predictor is not None

        payload: Dict = {
            "modality": modality,
            "quality": result.quality.to_dict(),
            "num_rois": len(result.rois),
            "rois": [r.to_dict() for r in result.rois],
            "prediction": None,
            "model_loaded": model_loaded,
            "analysis_stopped": not result.quality.passed,
            "pipeline_steps": _pipeline_steps(result.quality.passed, model_loaded),
            "processing_metadata": {
                "preprocessing_ops": _preprocessing_ops(config),
                "segmentation_success": segmentation_success,
                "foreground_ratio": round(fg_ratio, 4),
                "roi_confidence": _roi_confidence(result.rois, config),
                "processing_time_ms": processing_time_ms,
                "model_version": predictor.model_version if predictor else None,
                "inference_time_ms": None,  # filled in below if inference actually runs
            },
        }

        overlay = None
        if predictor is not None and result.quality.passed:
            overlay, prediction = predictor.explain(image_bgr)
            prediction["backbone"] = predictor.model_config.backbone
            payload["prediction"] = prediction
            payload["processing_metadata"]["inference_time_ms"] = prediction.get("inference_time_ms")

        return payload, result, overlay


@lru_cache(maxsize=1)
def get_analyzer() -> AnalyzerService:
    """Process-wide singleton (models loaded once)."""
    return AnalyzerService()
