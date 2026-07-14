"""Image Quality Assessment — the first gate in the pipeline.

A clinical tool must refuse to reason about a scan that is too blurry, too dark
or blown out. This module produces an explainable, *scored* verdict (not just
pass/fail) that the API and UI can surface directly as a quality report, and
that the analyzer uses to decide whether to run the model at all.

Scoring is a deliberate, documented design choice, not a learned metric: each
raw signal (focus / brightness / contrast) is mapped through a piecewise-linear
ramp anchored on the same thresholds used for the pass/fail gate, so a value
that just clears the gate scores ~55 (borderline) and a value well past it
scores 100. This keeps the score and the gate decision consistent with each
other instead of being two unrelated numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np


def _piecewise(value: float, points: List[Tuple[float, float]]) -> float:
    """Linear interpolation through sorted (x, y) anchors; clamps outside the range."""
    pts = sorted(points)
    if value <= pts[0][0]:
        return pts[0][1]
    if value >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= value <= x1:
            t = (value - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return pts[-1][1]  # pragma: no cover - unreachable given the checks above


def _status(score: float) -> str:
    if score >= 70:
        return "ok"
    if score >= 45:
        return "warn"
    return "fail"


@dataclass
class QualityCheck:
    name: str
    status: str   # "ok" | "warn" | "fail"
    detail: str

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass
class QualityReport:
    """Explainable, scored quality verdict for a single grayscale image."""

    focus: float          # variance of the Laplacian — higher means sharper
    brightness: float     # mean intensity, 0-255
    contrast: float       # standard deviation of intensity
    passed: bool
    reasons: List[str] = field(default_factory=list)

    overall_score: int = 100
    focus_score: int = 100
    brightness_score: int = 100
    contrast_score: int = 100
    motion_blur_ratio: float = 1.0     # gradient-anisotropy; 1.0 = isotropic, ->0 = directional (motion) blur
    motion_blur_detected: bool = False
    checks: List[QualityCheck] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "focus": round(self.focus, 2),
            "brightness": round(self.brightness, 2),
            "contrast": round(self.contrast, 2),
            "passed": self.passed,
            "reasons": self.reasons,
            "overall_score": self.overall_score,
            "focus_score": self.focus_score,
            "brightness_score": self.brightness_score,
            "contrast_score": self.contrast_score,
            "motion_blur_ratio": round(self.motion_blur_ratio, 3),
            "motion_blur_detected": self.motion_blur_detected,
            "checks": [c.to_dict() for c in self.checks],
            "recommendation": self.recommendation,
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
    passed = len(reasons) == 0

    # --- per-metric 0-100 scores, anchored on the same thresholds above ---
    focus_score = round(_piecewise(focus, [(0, 0), (min_focus, 55), (min_focus * 3, 100)]))
    contrast_score = round(_piecewise(contrast, [(0, 0), (min_contrast, 55), (min_contrast * 3, 100)]))
    center = (lo + hi) / 2
    half = (hi - lo) / 2
    dist = abs(brightness - center)
    brightness_score = round(_piecewise(dist, [(0, 100), (half, 55), (half * 2, 0)]))

    # --- motion-blur: gradient anisotropy (directional blur smears one axis more than the other) ---
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    ex, ey = float(np.mean(gx ** 2)), float(np.mean(gy ** 2))
    motion_blur_ratio = min(ex, ey) / max(ex, ey) if max(ex, ey) > 0 else 1.0
    # only flag when the image is *also* borderline-sharp, to avoid false positives on
    # legitimately directional content (e.g. rib structure) that is otherwise crisp
    motion_blur_detected = motion_blur_ratio < 0.35 and focus < min_focus * 1.5

    overall_score = round(0.45 * focus_score + 0.30 * brightness_score + 0.25 * contrast_score)
    if motion_blur_detected:
        overall_score = max(0, overall_score - 10)
    overall_score = max(0, min(100, overall_score))

    checks = [
        QualityCheck("Focus Quality", _status(focus_score),
                     f"Laplacian variance {focus:.1f} (min {min_focus:.0f})"),
        QualityCheck("Brightness", _status(brightness_score),
                     f"mean intensity {brightness:.1f} (target {lo:.0f}-{hi:.0f})"),
        QualityCheck("Contrast", _status(contrast_score),
                     f"intensity std {contrast:.1f} (min {min_contrast:.0f})"),
    ]
    if motion_blur_detected:
        checks.append(QualityCheck(
            "Motion Blur", "warn",
            f"directional gradient ratio {motion_blur_ratio:.2f} (1.00 = isotropic)",
        ))

    if passed:
        recommendation = (
            "Image quality is excellent. Suitable for AI analysis."
            if overall_score >= 85 else
            "Image quality is acceptable. Suitable for AI analysis, though a sharper "
            "or better-exposed scan would improve reliability."
        )
    else:
        recommendation = (
            "Image quality is insufficient for reliable AI analysis: "
            + "; ".join(reasons) + ". Recommendation: retake the scan."
        )

    return QualityReport(
        focus=focus, brightness=brightness, contrast=contrast,
        passed=passed, reasons=reasons,
        overall_score=overall_score, focus_score=focus_score,
        brightness_score=brightness_score, contrast_score=contrast_score,
        motion_blur_ratio=motion_blur_ratio, motion_blur_detected=motion_blur_detected,
        checks=checks, recommendation=recommendation,
    )
