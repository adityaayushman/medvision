
from __future__ import annotations

import gc
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

MODALITY_CHECKPOINTS: Dict[str, str] = {
    "chest_xray": settings.model_checkpoint,
    "brain_mri": settings.model_checkpoint_brain_mri,
    "mammography": settings.model_checkpoint_mammography,
}


def _preprocessing_ops(cfg: PreprocessConfig) -> List[str]:
    ops = [f"Denoise: {cfg.denoise_method} ({cfg.denoise_ksize}x{cfg.denoise_ksize})"]
    if cfg.use_hist_eq:
        ops.append(f"CLAHE: clip={cfg.clahe_clip}, tile={cfg.clahe_tile}x{cfg.clahe_tile}")
    ops.append(f"Segmentation: {cfg.threshold_method} thresholding")
    if cfg.morph_op != "none":
        ops.append(f"Morphology: {cfg.morph_op} ({cfg.morph_ksize}x{cfg.morph_ksize}, x{cfg.morph_iterations})")
    ops.append(f"ROI extraction: contour-based, min area {cfg.min_roi_area_ratio * 100:.1f}% of image")
    return ops


def _roi_confidence(rois: List[ROI], cfg: PreprocessConfig) -> Optional[Dict]:
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
        # Ensemble checkpoints (2+ paths) for a modality are NOT loaded here.
        # This service loads every configured modality eagerly at startup, so
        # holding a 2-3 model ensemble resident permanently means it's stacked
        # in memory on top of every other modality for the process's whole
        # lifetime -- confirmed empirically to OOM/crash-loop this
        # deployment's 512MB free-tier ceiling (see git log: the brain MRI
        # ensemble). Modalities listed here instead load fresh per request in
        # analyze() and are discarded immediately after -- see _lazy_ckpts.
        self._lazy_ckpts: Dict[str, List[str]] = {}
        for modality, ckpt_path in MODALITY_CHECKPOINTS.items():
            cfg = get_config(modality)
            self.configs[modality] = cfg
            self.pipelines[modality] = MedicalImagePipeline(cfg)
            predictor, lazy_paths = self._resolve_predictor(ckpt_path, modality)
            self.predictors[modality] = predictor
            if lazy_paths:
                self._lazy_ckpts[modality] = lazy_paths

    @staticmethod
    def _resolve_predictor(ckpt_spec: str, modality: str):
        """Returns (predictor_or_None, lazy_ckpt_paths_or_None). Exactly one
        of the two is non-None when a usable checkpoint exists."""
        if not ckpt_spec:
            return None, None
        candidates = [p.strip() for p in ckpt_spec.split(",") if p.strip()]
        existing = [p for p in candidates if Path(p).exists()]
        if not existing:
            print(f"[ml] No checkpoint(s) at {candidates} for modality={modality!r} — preprocess-only for this modality.")
            return None, None
        if len(existing) > 1:
            print(f"[ml] {modality} ensemble ({len(existing)} checkpoints) will load per-request, not at startup: {existing}")
            return None, existing
        try:
            from medchron.models import Predictor
            print(f"[ml] Loading {modality} checkpoint {existing[0]}")
            return Predictor(existing[0]), None
        except Exception as exc:
            print(f"[ml] Could not load {modality} model ({exc}); preprocess-only for this modality.")
            return None, None

    @staticmethod
    def _load_ensemble(paths: List[str], modality: str):
        try:
            from medchron.models import EnsemblePredictor
            print(f"[ml] Loading {modality} ensemble ({len(paths)} checkpoints) for this request")
            return EnsemblePredictor(paths)
        except Exception as exc:
            print(f"[ml] Could not load {modality} ensemble ({exc}); preprocess-only for this request.")
            return None

    def available_modalities(self) -> Dict[str, bool]:
        return {
            m: (self.predictors.get(m) is not None or m in self._lazy_ckpts)
            for m in self.pipelines
        }

    @property
    def model_loaded(self) -> bool:
        m = settings.modality
        return self.predictors.get(m) is not None or m in self._lazy_ckpts

    def analyze(self, image_bgr: np.ndarray, modality: Optional[str] = None):
        modality = modality if modality in self.pipelines else settings.modality
        pipeline = self.pipelines[modality]
        config = self.configs[modality]

        predictor = self.predictors[modality]
        is_lazy = predictor is None and modality in self._lazy_ckpts
        if is_lazy:
            predictor = self._load_ensemble(self._lazy_ckpts[modality], modality)

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
                "inference_time_ms": None,
            },
        }

        overlay = None
        if predictor is not None and result.quality.passed:
            overlay, prediction = predictor.explain(image_bgr)
            prediction["backbone"] = predictor.backbone
            payload["prediction"] = prediction
            payload["processing_metadata"]["inference_time_ms"] = prediction.get("inference_time_ms")

        if is_lazy:
            del predictor
            gc.collect()

        return payload, result, overlay


@lru_cache(maxsize=1)
def get_analyzer() -> AnalyzerService:
    return AnalyzerService()
