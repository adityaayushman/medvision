"""Tests for the U-Net lesion segmenter (models.segment) and the
task-marker dispatch in LocalizedPredictor that lets it accept either a
bbox regressor or a segmentation model as its localizer.

Follows test_detect.py's pattern — synthetic (pretrained=False) checkpoints
built directly via torch.save where possible, so these stay fast and
memory-light; one tiny real end-to-end training run proves the pieces wire
up correctly.
"""

from __future__ import annotations

import gc
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from medchron.config import PreprocessConfig
from medchron.models import ModelConfig, create_model
from medchron.models.detect import SpatialBBoxNet
from medchron.models.segment import (
    SegmentationSample,
    SegmentationTrainConfig,
    UNetSegmenter,
    dice_loss,
    evaluate_segmenter,
    mask_to_bbox,
    train_segmenter,
)
from medchron.models.inference import LocalizedPredictor

CPU = torch.device("cpu")
LIGHT = "efficientnet_b0"
CLASSES = {"Benign": 0, "Malignant": 1}


def _write_synthetic_segmentation_checkpoint(tmp_path, name: str = "seg.pt") -> str:
    model = UNetSegmenter(pretrained=False)
    path = tmp_path / name
    torch.save(
        {
            "state_dict": model.state_dict(),
            "preprocess": PreprocessConfig().to_dict(),
            "train_config": {},
            "history": [],
            "task": "segmentation",
        },
        path,
    )
    del model
    gc.collect()
    return str(path)


def _write_synthetic_bbox_checkpoint(tmp_path, name: str = "bbox.pt") -> str:
    model = SpatialBBoxNet(pretrained=False)
    path = tmp_path / name
    torch.save(
        {
            "state_dict": model.state_dict(),
            "preprocess": PreprocessConfig().to_dict(),
            "train_config": {},
            "history": [],
            "task": "bbox_regression",
        },
        path,
    )
    del model
    gc.collect()
    return str(path)


def _write_synthetic_classifier_checkpoint(tmp_path, name: str = "clf.pt") -> str:
    cfg = ModelConfig(backbone=LIGHT, num_classes=len(CLASSES), pretrained=False)
    model = create_model(cfg)
    path = tmp_path / name
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_to_idx": CLASSES,
            "model_config": asdict(cfg),
            "train_config": {},
            "preprocess": PreprocessConfig().to_dict(),
            "history": [],
        },
        path,
    )
    del model
    gc.collect()
    return str(path)


def test_dice_loss_perfect_overlap_and_disjoint():
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1.0

    perfect_logits = torch.full((1, 1, 8, 8), -10.0)
    perfect_logits[:, :, 2:6, 2:6] = 10.0
    assert dice_loss(perfect_logits, target).item() == pytest.approx(0.0, abs=1e-3)

    disjoint_logits = torch.full((1, 1, 8, 8), 10.0)
    disjoint_logits[:, :, 2:6, 2:6] = -10.0
    assert dice_loss(disjoint_logits, target).item() > 0.9


def test_mask_to_bbox_recovers_known_region():
    mask = np.zeros((100, 200), dtype=np.float32)
    mask[20:40, 60:100] = 1.0  # rows(y) 20-39, cols(x) 60-99
    cx, cy, w, h = mask_to_bbox(mask)
    assert cx == pytest.approx((60 + 99) / 2 / 200, abs=1e-3)
    assert cy == pytest.approx((20 + 39) / 2 / 100, abs=1e-3)
    assert w == pytest.approx((99 - 60) / 200, abs=1e-3)
    assert h == pytest.approx((39 - 20) / 100, abs=1e-3)


def test_mask_to_bbox_empty_returns_none():
    mask = np.zeros((50, 50), dtype=np.float32)
    assert mask_to_bbox(mask) is None


def _write_synthetic_images_and_masks(tmp_path, n=16):
    import cv2
    rng = np.random.default_rng(0)
    samples = []
    for i in range(n):
        img_path = tmp_path / f"img{i}.png"
        cv2.imwrite(str(img_path), rng.integers(0, 255, (48, 48), dtype=np.uint8))
        mask_path = tmp_path / f"mask{i}.png"
        mask = np.zeros((48, 48), dtype=np.uint8)
        mask[15:30, 15:30] = 255
        cv2.imwrite(str(mask_path), mask)
        samples.append((str(img_path), str(mask_path)))
    return samples


def test_end_to_end_segmentation_train_and_evaluate(tmp_path):
    pairs = _write_synthetic_images_and_masks(tmp_path)
    splits = ["train"] * 10 + ["val"] * 3 + ["test"] * 3
    samples = [
        SegmentationSample(img_path, [mask_path], split=s)
        for (img_path, mask_path), s in zip(pairs, splits)
    ]

    tcfg = SegmentationTrainConfig(
        epochs=1, batch_size=4, device="cpu", out_dir=str(tmp_path / "artifacts"),
    )
    result = train_segmenter(samples, tcfg)
    assert "checkpoint" in result
    assert len(result["history"]) == 1

    metrics = evaluate_segmenter(result["checkpoint"], samples, split="test", device="cpu")
    assert "mean_dice" in metrics
    assert 0.0 <= metrics["mean_dice"] <= 1.0
    gc.collect()


def test_localized_predictor_dispatches_segmentation_checkpoint(tmp_path):
    seg_ckpt = _write_synthetic_segmentation_checkpoint(tmp_path)
    clf_ckpt = _write_synthetic_classifier_checkpoint(tmp_path)
    predictor = LocalizedPredictor(seg_ckpt, clf_ckpt, device=CPU)

    assert predictor.model_version.startswith("localized(segmentation+")

    image = np.random.randint(0, 255, (400, 300, 3), np.uint8)
    pred = predictor.predict(image)
    assert pred["label"] in CLASSES
    box = pred["detected_bbox"]
    assert set(box) == {"cx", "cy", "w", "h"}
    gc.collect()


def test_localized_predictor_still_dispatches_bbox_checkpoint(tmp_path):
    """Regression check: adding the segmentation dispatch branch must not
    break the original bbox-regressor path."""
    bbox_ckpt = _write_synthetic_bbox_checkpoint(tmp_path)
    clf_ckpt = _write_synthetic_classifier_checkpoint(tmp_path)
    predictor = LocalizedPredictor(bbox_ckpt, clf_ckpt, device=CPU)

    assert predictor.model_version.startswith("localized(bbox+")

    image = np.random.randint(0, 255, (400, 300, 3), np.uint8)
    pred = predictor.predict(image)
    assert pred["label"] in CLASSES
    gc.collect()
