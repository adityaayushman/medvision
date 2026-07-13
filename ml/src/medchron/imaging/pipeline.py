"""The end-to-end MedChron preprocessing pipeline.

This ties the individual DIP steps together into one auditable pass and returns
*every* intermediate stage, so the same result object can drive:
  * the training data preparation (``model_input``),
  * the explainability overlays (ROIs, enhanced image),
  * and the UI's "show your work" stage gallery.
"""

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
    """Container for every stage of one image's journey through the pipeline."""

    original: np.ndarray      # BGR, resized to target_size
    gray: np.ndarray
    denoised: np.ndarray
    enhanced: np.ndarray
    mask: np.ndarray
    cleaned_mask: np.ndarray
    quality: QualityReport
    rois: List[ROI]
    annotated: np.ndarray     # BGR original with ROI boxes drawn

    @property
    def stages(self) -> Dict[str, np.ndarray]:
        """Named stages, in pipeline order — ready for a montage or UI gallery."""
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
        """Compact, JSON-friendly summary for logging / API responses."""
        return {
            "quality": self.quality.to_dict(),
            "num_rois": len(self.rois),
            "rois": [r.to_dict() for r in self.rois],
        }


class MedicalImagePipeline:
    """Config-driven preprocessing pipeline for a single modality.

    Example
    -------
    >>> from medchron import MedicalImagePipeline
    >>> pipe = MedicalImagePipeline()
    >>> result = pipe.run_path("assets/samples/chest_xray_sample.jpg")
    >>> result.quality.passed, len(result.rois)
    """

    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:
        self.config = config or PreprocessConfig()

    def run(self, image: np.ndarray) -> PipelineResult:
        """Run the full pipeline on an in-memory BGR (or grayscale) image."""
        cfg = self.config

        original = cv2.resize(image, cfg.target_size, interpolation=cv2.INTER_AREA)
        gray = to_grayscale(original)
        quality = assess_quality(gray)               # gate on the raw image
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
        """Read an image from disk and run the pipeline."""
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return self.run(image)

    def model_input(self, result: PipelineResult) -> np.ndarray:
        """Build a 3-channel float32 tensor for an ImageNet backbone (VGG16).

        We feed the *enhanced* grayscale replicated across RGB channels and
        resized to ``model_input_size``. The caller is responsible for applying
        the backbone-specific ``preprocess_input`` (e.g. VGG16 mean subtraction).
        """
        cfg = self.config
        resized = cv2.resize(
            result.enhanced, cfg.model_input_size, interpolation=cv2.INTER_AREA
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        return rgb.astype(np.float32)


def model_image(image: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Fast enhancement-only path to a model-ready ``HxWx3`` uint8 RGB image.

    Used by the training data loader, where running full segmentation + contour
    extraction per sample would be wasteful. It still applies the DIP enhancement
    (denoise + CLAHE) that distinguishes MedChron from a raw-pixel classifier —
    it simply skips the segmentation/ROI stages the classifier doesn't consume.

    Resizes to ``model_input_size`` *before* denoise/CLAHE, not after: the model
    never sees more than that many pixels, so enhancing at the source resolution
    (e.g. 1024x1024 DICOM) wastes ~20x the CPU for no benefit downstream.
    """
    gray = to_grayscale(image)
    small = cv2.resize(gray, cfg.model_input_size, interpolation=cv2.INTER_AREA)
    denoised = denoise(small, cfg)
    enhanced = enhance_contrast(denoised, cfg)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
