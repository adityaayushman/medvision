"""Image Quality Assessment — the first gate in the pipeline.

A clinical tool must refuse to reason about a scan that is too blurry, too dark
or blown out. This module produces a small, explainable verdict that the API and
UI can surface to the user ("rejected: image is blurry").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class QualityReport:
    """Explainable quality verdict for a single grayscale image."""

    focus: float          # variance of the Laplacian — higher means sharper
    brightness: float     # mean intensity, 0-255
    contrast: float       # standard deviation of intensity
    passed: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "focus": round(self.focus, 2),
            "brightness": round(self.brightness, 2),
            "contrast": round(self.contrast, 2),
            "passed": self.passed,
            "reasons": self.reasons,
        }


def assess_quality(
    gray: np.ndarray,
    *,
    min_focus: float = 80.0,
    brightness_range: Tuple[float, float] = (25.0, 235.0),
    min_contrast: float = 12.0,
) -> QualityReport:
    """Score sharpness, exposure and contrast, and explain any failure.

    Measured on the *raw* grayscale image (before denoising, which would
    artificially lower the focus score).
    """
    focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    reasons: List[str] = []
    if focus < min_focus:
        reasons.append(f"blurry / out of focus (Laplacian variance {focus:.1f} < {min_focus})")
    lo, hi = brightness_range
    if brightness < lo:
        reasons.append(f"under-exposed (mean brightness {brightness:.1f} < {lo})")
    elif brightness > hi:
        reasons.append(f"over-exposed (mean brightness {brightness:.1f} > {hi})")
    if contrast < min_contrast:
        reasons.append(f"low contrast (intensity std {contrast:.1f} < {min_contrast})")

    return QualityReport(focus, brightness, contrast, len(reasons) == 0, reasons)
