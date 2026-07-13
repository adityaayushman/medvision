"""Tests for the medchron.imaging pipeline.

These use a synthetic image so they need no dataset and run in milliseconds.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from medchron import MedicalImagePipeline, PreprocessConfig, get_config
from medchron.imaging import assess_quality


@pytest.fixture
def synthetic_scan() -> np.ndarray:
    """A dark background with two bright blobs — stand-in for lesions/regions."""
    img = np.full((400, 400, 3), 20, dtype=np.uint8)
    cv2.circle(img, (120, 120), 45, (200, 200, 200), -1)
    cv2.circle(img, (280, 260), 60, (230, 230, 230), -1)
    # a little texture so it isn't perfectly flat (raises focus/contrast)
    noise = np.random.default_rng(0).integers(0, 25, img.shape, dtype=np.uint8)
    return cv2.add(img, noise)


def test_pipeline_runs_and_returns_all_stages(synthetic_scan):
    result = MedicalImagePipeline().run(synthetic_scan)
    expected = {"original", "gray", "denoised", "enhanced", "mask", "cleaned_mask", "annotated"}
    assert expected <= set(result.stages)
    # working resolution honoured (width, height)
    assert result.original.shape[:2] == (512, 512)
    assert result.gray.ndim == 2


def test_rois_are_detected_and_bounded(synthetic_scan):
    result = MedicalImagePipeline().run(synthetic_scan)
    assert len(result.rois) >= 2  # both blobs found
    assert len(result.rois) <= PreprocessConfig().max_rois
    for roi in result.rois:
        assert roi.w > 0 and roi.h > 0
        assert roi.crop.size > 0
        assert roi.area > 0
    # sorted largest-first
    areas = [r.area for r in result.rois]
    assert areas == sorted(areas, reverse=True)


def test_max_rois_is_respected(synthetic_scan):
    cfg = PreprocessConfig(max_rois=1)
    result = MedicalImagePipeline(cfg).run(synthetic_scan)
    assert len(result.rois) == 1


def test_model_input_shape_and_dtype(synthetic_scan):
    pipe = MedicalImagePipeline()
    result = pipe.run(synthetic_scan)
    tensor = pipe.model_input(result)
    assert tensor.shape == (224, 224, 3)
    assert tensor.dtype == np.float32


def test_quality_flags_blur():
    flat = np.full((256, 256), 128, dtype=np.uint8)  # zero focus, zero contrast
    report = assess_quality(flat)
    assert report.passed is False
    assert any("blur" in r or "contrast" in r for r in report.reasons)


def test_presets_exist():
    for name in ("chest_xray", "brain_mri", "mammography"):
        assert isinstance(get_config(name), PreprocessConfig)
    with pytest.raises(KeyError):
        get_config("does_not_exist")
