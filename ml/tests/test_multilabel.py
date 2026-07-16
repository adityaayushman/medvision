"""Tests for multi-label training support (e.g. NIH ChestX-ray14: several
simultaneous findings per image, unlike RSNA's single label per image).

Uses EfficientNet-B0 (~20 MB) to stay fast and memory-safe, consistent with
the rest of the model-layer tests.
"""

from __future__ import annotations

import gc

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from medchron.data import Sample
from medchron.models import ModelConfig, TrainConfig, create_model, evaluate_checkpoint, train
from medchron.models.dataset import MedChronDataset, build_dataloaders, multilabel_pos_weights
from medchron.models.evaluate import compute_multilabel_metrics

LIGHT = "efficientnet_b0"


def _write_images(tmp_path, n=8):
    paths = []
    rng = np.random.default_rng(0)
    for i in range(n):
        p = tmp_path / f"img{i}.png"
        cv2.imwrite(str(p), rng.integers(0, 255, (32, 32), dtype=np.uint8))
        paths.append(str(p))
    return paths


def test_dataset_produces_multihot_targets(tmp_path):
    paths = _write_images(tmp_path, 2)
    samples = [
        Sample(paths[0], "Cardiomegaly|Effusion", split="train"),
        Sample(paths[1], "No Finding", split="train"),
    ]
    class_to_idx = {"Cardiomegaly": 0, "Effusion": 1, "No Finding": 2}
    ds = MedChronDataset(samples, class_to_idx, task="multilabel")

    _, target0 = ds[0]
    assert target0.dtype == torch.float32
    assert target0.tolist() == [1.0, 1.0, 0.0]

    _, target1 = ds[1]
    assert target1.tolist() == [0.0, 0.0, 1.0]


def test_multilabel_pos_weights_reflect_rarity():
    samples = [
        Sample("a.png", "Rare"),
        Sample("b.png", "Common"),
        Sample("c.png", "Common"),
        Sample("d.png", "Common"),
    ]
    class_to_idx = {"Rare": 0, "Common": 1}
    w = multilabel_pos_weights(samples, class_to_idx)
   
    assert w[0].item() == pytest.approx(3.0)
    assert w[1].item() == pytest.approx(1 / 3)


def test_build_dataloaders_multilabel_class_index(tmp_path):
    paths = _write_images(tmp_path, 4)
    samples = [
        Sample(paths[0], "A|B", split="train"),
        Sample(paths[1], "B", split="train"),
        Sample(paths[2], "A", split="val"),
        Sample(paths[3], "B", split="test"),
    ]
    bundle = build_dataloaders(samples, batch_size=2, task="multilabel")
    assert set(bundle.class_to_idx) == {"A", "B"}
    assert set(bundle.loaders) == {"train", "val", "test"}
    xb, yb = next(iter(bundle.loaders["train"]))
    assert yb.shape == (2, 2)        
    assert yb.dtype == torch.float32


def test_compute_multilabel_metrics_perfect_predictions():
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32)
    y_prob = y_true.copy()
    metrics = compute_multilabel_metrics(y_true, y_prob, ["A", "B"])
    assert metrics["exact_match_accuracy"] == 1.0
    assert metrics["hamming_accuracy"] == 1.0
    assert metrics["per_class"]["A"]["precision"] == 1.0
    assert metrics["per_class"]["B"]["recall"] == 1.0


def test_end_to_end_multilabel_train_and_evaluate(tmp_path):
    """Tiny real training run (1 epoch/phase) proving the multilabel path
    connects end to end: dataset -> BCE loss -> checkpoint -> evaluation.
    """
    paths = _write_images(tmp_path, 12)
    labels = ["A|B", "A", "B", "A|B", "A", "B", "A|B", "A", "B", "A|B", "A", "B"]
    samples = [
        Sample(p, lbl, split=("train" if i < 8 else "val" if i < 10 else "test"))
        for i, (p, lbl) in enumerate(zip(paths, labels))
    ]

    tcfg = TrainConfig(
        backbone=LIGHT, task="multilabel", epochs_head=1, epochs_finetune=1,
        batch_size=4, device="cpu", out_dir=str(tmp_path / "artifacts"),
    )
    result = train(samples, tcfg)
    assert set(result["class_to_idx"]) == {"A", "B"}

    metrics = evaluate_checkpoint(result["checkpoint"], samples, split="test", device="cpu")
    assert "mean_auc" in metrics
    assert "A" in metrics["per_class"] and "B" in metrics["per_class"]
    gc.collect()
