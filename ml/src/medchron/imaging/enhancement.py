"""Grayscale conversion, denoising and contrast enhancement.

Each function does exactly one Digital Image Processing step and takes the
:class:`~medchron.config.PreprocessConfig` so the choices stay auditable.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import PreprocessConfig


def _odd(value: int) -> int:
    """OpenCV kernels must be odd; nudge even values up by one."""
    return value if value % 2 == 1 else value + 1


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 image, whatever the input channel count."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Suppress acquisition noise while preserving edges where possible."""
    k = _odd(cfg.denoise_ksize)
    if cfg.denoise_method == "gaussian":
        return cv2.GaussianBlur(gray, (k, k), 0)
    if cfg.denoise_method == "median":
        return cv2.medianBlur(gray, k)
    if cfg.denoise_method == "bilateral":
        return cv2.bilateralFilter(gray, cfg.bilateral_d, 75, 75)
    return gray  # "none"


def enhance_contrast(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Global histogram equalisation (optional) followed by CLAHE.

    CLAHE is the workhorse for medical images: it lifts local contrast in soft
    tissue without blowing out already-bright regions.
    """
    out = gray
    if cfg.use_hist_eq:
        out = cv2.equalizeHist(out)
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip,
        tileGridSize=(cfg.clahe_tile, cfg.clahe_tile),
    )
    return clahe.apply(out)


def normalize01(gray: np.ndarray) -> np.ndarray:
    """Scale a uint8 image to float32 in [0, 1] (model-friendly)."""
    return gray.astype(np.float32) / 255.0
