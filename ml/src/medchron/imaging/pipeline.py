
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from ..config import PreprocessConfig
from .enhancement import denoise, enhance_contrast, to_grayscale
from .quality import QualityReport, assess_quality
from .roi import ROI, draw_rois, extract_rois
from .segmentation import clean_mask, segment


@dataclass
class PipelineResult:

    original: np.ndarray
    gray: np.ndarray
    denoised: np.ndarray
    enhanced: np.ndarray
    mask: np.ndarray
    cleaned_mask: np.ndarray
    quality: QualityReport
    rois: List[ROI]
    annotated: np.ndarray

    @property
    def stages(self) -> Dict[str, np.ndarray]:
        return {
            "original": self.original,
            "gray": self.gray,
            "denoised": self.denoised,
            "enhanced": self.enhanced,
            "mask": self.mask,
            "cleaned_mask": self.cleaned_mask,
            "annotated": self.annotated,
        }

    def summary(self) -> dict:
        return {
            "quality": self.quality.to_dict(),
            "num_rois": len(self.rois),
            "rois": [r.to_dict() for r in self.rois],
        }


class MedicalImagePipeline:

    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:
        self.config = config or PreprocessConfig()

    def run(self, image: np.ndarray) -> PipelineResult:
        cfg = self.config

        original = cv2.resize(image, cfg.target_size, interpolation=cv2.INTER_AREA)
        gray = to_grayscale(original)
        quality = assess_quality(
            gray,
            min_focus=cfg.min_focus,
            brightness_range=(cfg.brightness_lo, cfg.brightness_hi),
            min_contrast=cfg.min_contrast,
        )
        denoised = denoise(gray, cfg)
        enhanced = enhance_contrast(denoised, cfg)
        mask = segment(enhanced, cfg)
        cleaned = clean_mask(mask, cfg)
        rois = extract_rois(cleaned, original, cfg)
        annotated = draw_rois(original, rois)

        return PipelineResult(
            original=original,
            gray=gray,
            denoised=denoised,
            enhanced=enhanced,
            mask=mask,
            cleaned_mask=cleaned,
            quality=quality,
            rois=rois,
            annotated=annotated,
        )

    def run_path(self, path: Union[str, Path]) -> PipelineResult:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return self.run(image)

    def model_input(self, result: PipelineResult) -> np.ndarray:
        cfg = self.config
        resized = cv2.resize(
            result.enhanced, cfg.model_input_size, interpolation=cv2.INTER_AREA
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        return rgb.astype(np.float32)


def model_image(image: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    gray = to_grayscale(image)
    small = cv2.resize(gray, cfg.model_input_size, interpolation=cv2.INTER_AREA)
    denoised = denoise(small, cfg)
    enhanced = enhance_contrast(denoised, cfg)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
