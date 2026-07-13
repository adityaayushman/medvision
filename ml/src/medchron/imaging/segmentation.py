"""Thresholding-based segmentation and morphological mask cleanup."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import PreprocessConfig

_MORPH_TYPES = {"open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE}


def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def segment(enhanced: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Produce a binary foreground mask from the enhanced grayscale image."""
    if cfg.threshold_method == "otsu":
        _, mask = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return mask
    if cfg.threshold_method == "adaptive_mean":
        method = cv2.ADAPTIVE_THRESH_MEAN_C
    else:  # adaptive_gaussian
        method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    return cv2.adaptiveThreshold(
        enhanced,
        255,
        method,
        cv2.THRESH_BINARY,
        _odd(cfg.adaptive_block_size),
        cfg.adaptive_c,
    )


def clean_mask(mask: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Remove speckle / close gaps in the mask via morphology."""
    if cfg.morph_op == "none":
        return mask
    kernel = np.ones((cfg.morph_ksize, cfg.morph_ksize), np.uint8)
    if cfg.morph_op in _MORPH_TYPES:
        return cv2.morphologyEx(
            mask, _MORPH_TYPES[cfg.morph_op], kernel, iterations=cfg.morph_iterations
        )
    if cfg.morph_op == "erode":
        return cv2.erode(mask, kernel, iterations=cfg.morph_iterations)
    if cfg.morph_op == "dilate":
        return cv2.dilate(mask, kernel, iterations=cfg.morph_iterations)
    return mask
