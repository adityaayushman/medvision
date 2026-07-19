"""Tests for the bbox-regression lesion localizer (models.detect) and the
composed LocalizedPredictor (bbox -> crop -> classify) in models.inference.

Follows test_ensemble.py's pattern — synthetic (pretrained=False) checkpoints
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
from medchron.models.detect import (
    BBoxSample,
    BBoxTrainConfig,
    SpatialBBoxNet,
    box_iou,
    evaluate_bbox_checkpoint,
    train_bbox_regressor,
)
from medchron.models.inference import LocalizedPredictor

CPU = torch.device("cpu")
LIGHT = "efficientnet_b0"
CLASSES = {"Benign": 0, "Malignant": 1}


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


def test_box_iou_perfect_overlap_and_disjoint():
    same = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    assert box_iou(same, same).item() == pytest.approx(1.0, abs=1e-5)

    disjoint = torch.tensor([[0.1, 0.1, 0.05, 0.05]])
    other = torch.tensor([[0.9, 0.9, 0.05, 0.05]])
    assert box_iou(disjoint, other).item() == pytest.approx(0.0, abs=1e-5)


def _write_synthetic_images(tmp_path, n=16):
    import cv2
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        p = tmp_path / f"img{i}.png"
        cv2.imwrite(str(p), rng.integers(0, 255, (48, 48), dtype=np.uint8))
        paths.append(str(p))
    return paths


def test_end_to_end_bbox_train_and_evaluate(tmp_path):
    paths = _write_synthetic_images(tmp_path)
    splits = ["train"] * 10 + ["val"] * 3 + ["test"] * 3
    samples = [
        BBoxSample(p, cx=0.5, cy=0.5, w=0.3, h=0.3, split=s)
        for p, s in zip(paths, splits)
    ]

    tcfg = BBoxTrainConfig(
        backbone=LIGHT, epochs=1, batch_size=4, device="cpu", out_dir=str(tmp_path / "artifacts"),
    )
    result = train_bbox_regressor(samples, tcfg)
    assert "checkpoint" in result
    assert len(result["history"]) == 1

    metrics = evaluate_bbox_checkpoint(result["checkpoint"], samples, split="test", device="cpu")
    assert "mean_iou" in metrics
    assert 0.0 <= metrics["mean_iou"] <= 1.0
    gc.collect()


def test_localized_predictor_predict_and_explain(tmp_path):
    bbox_ckpt = _write_synthetic_bbox_checkpoint(tmp_path)
    clf_ckpt = _write_synthetic_classifier_checkpoint(tmp_path)
    predictor = LocalizedPredictor(bbox_ckpt, clf_ckpt, device=CPU)

    image = np.random.randint(0, 255, (400, 300, 3), np.uint8)

    pred = predictor.predict(image)
    assert pred["label"] in CLASSES
    assert 0.0 <= pred["confidence"] <= 1.0
    assert abs(sum(pred["probabilities"].values()) - 1.0) < 1e-4
    box = pred["detected_bbox"]
    assert set(box) == {"cx", "cy", "w", "h"}
    assert pred["model_version"].startswith("localized(")

    overlay, explained = predictor.explain(image)
    assert overlay.shape[2] == 3
    assert explained["explained_class"] in CLASSES
    assert "detected_bbox" in explained
    gc.collect()
