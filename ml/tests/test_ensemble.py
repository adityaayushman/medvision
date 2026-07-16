"""Tests for the multi-model ensemble (v3): soft-voting across several
Predictors trained on the same manifest/split.

Follows test_models.py's pattern — synthetic (pretrained=False) checkpoints
built directly via torch.save, no real training/download needed, so these
stay fast and memory-light.
"""

from __future__ import annotations

import gc
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from medchron.config import PreprocessConfig
from medchron.data import Sample
from medchron.models import EnsemblePredictor, ModelConfig, TrainConfig, create_model, evaluate_ensemble, train
from medchron.models.evaluate import evaluate_checkpoint

CPU = torch.device("cpu")
LIGHT = "efficientnet_b0"
CLASSES = {"glioma_tumor": 0, "meningioma_tumor": 1, "no_tumor": 2, "pituitary_tumor": 3}


def _write_synthetic_checkpoint(tmp_path, name: str, class_to_idx=None) -> str:
    class_to_idx = class_to_idx or CLASSES
    cfg = ModelConfig(backbone=LIGHT, num_classes=len(class_to_idx), pretrained=False)
    model = create_model(cfg)
    path = tmp_path / name
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
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


def test_ensemble_needs_at_least_two_checkpoints(tmp_path):
    ckpt = _write_synthetic_checkpoint(tmp_path, "a.pt")
    with pytest.raises(ValueError):
        EnsemblePredictor([ckpt], device=CPU)


def test_ensemble_mismatched_class_to_idx_raises(tmp_path):
    ckpt_a = _write_synthetic_checkpoint(tmp_path, "a.pt", CLASSES)
    other_classes = {"normal": 0, "pneumonia": 1}
    ckpt_b = _write_synthetic_checkpoint(tmp_path, "b.pt", other_classes)
    with pytest.raises(ValueError):
        EnsemblePredictor([ckpt_a, ckpt_b], device=CPU)


def test_ensemble_predict_averages_and_reports_per_model(tmp_path):
    ckpts = [_write_synthetic_checkpoint(tmp_path, f"m{i}.pt") for i in range(2)]
    ensemble = EnsemblePredictor(ckpts, device=CPU)
    image = np.random.randint(0, 255, (300, 300, 3), np.uint8)

    pred = ensemble.predict(image)
    assert pred["label"] in CLASSES
    assert 0.0 <= pred["confidence"] <= 1.0
    assert abs(sum(pred["probabilities"].values()) - 1.0) < 1e-4
    assert len(pred["per_model"]) == 2
    for member in pred["per_model"]:
        assert member["backbone"] == LIGHT
        assert member["label"] in CLASSES
        assert 0.0 <= member["confidence"] <= 1.0


def test_ensemble_explain_matches_reported_label(tmp_path):
    ckpts = [_write_synthetic_checkpoint(tmp_path, f"m{i}.pt") for i in range(3)]
    ensemble = EnsemblePredictor(ckpts, device=CPU)
    image = np.random.randint(0, 255, (300, 300, 3), np.uint8)

    overlay, pred = ensemble.explain(image)
    assert overlay.shape[2] == 3
    # The Grad-CAM overlay must explain the class actually reported as the
    # ensemble's prediction, not whichever class the primary member alone
    # would have argmax'd to.
    assert pred["explained_class"] == pred["label"]
    assert pred["model_version"].startswith("ensemble(")
    assert len(pred["per_model"]) == 3
    gc.collect()


def test_end_to_end_ensemble_train_and_evaluate(tmp_path):
    """Tiny real training run proving evaluate_ensemble connects end to end:
    two trained checkpoints -> averaged probabilities -> the same metrics
    shape as a single-model evaluate_checkpoint (directly comparable)."""
    rng = np.random.default_rng(0)
    import cv2
    paths = []
    for i in range(16):
        p = tmp_path / f"img{i}.png"
        cv2.imwrite(str(p), rng.integers(0, 255, (32, 32), dtype=np.uint8))
        paths.append(str(p))

    labels = ["a", "b"] * 8
    samples = [
        Sample(p, lbl, split=("train" if i < 10 else "val" if i < 13 else "test"))
        for i, (p, lbl) in enumerate(zip(paths, labels))
    ]

    ckpts = []
    for seed in (0, 1):
        tcfg = TrainConfig(
            backbone=LIGHT, epochs_head=1, epochs_finetune=1, batch_size=4,
            device="cpu", seed=seed, out_dir=str(tmp_path / f"artifacts_{seed}"),
        )
        result = train(samples, tcfg)
        ckpts.append(result["checkpoint"])

    single_metrics = evaluate_checkpoint(ckpts[0], samples, split="test", device="cpu")
    ensemble_metrics = evaluate_ensemble(ckpts, samples, split="test", device="cpu")

    assert set(ensemble_metrics) == set(single_metrics)
    assert "accuracy" in ensemble_metrics and "roc_auc" in ensemble_metrics
    gc.collect()
