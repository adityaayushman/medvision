"""Tests for the dataset layer: manifests, splitting, and leakage protection."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from medchron.data import (
    REGISTRY,
    Sample,
    build_manifest_from_csv,
    build_manifest_from_folders,
    build_multilabel_class_index,
    class_distribution,
    get_spec,
    read_manifest,
    recommended_v1,
    split_distribution,
    stratified_split,
    write_manifest,
)
from medchron.data.split import assert_no_patient_leakage


def _make_dataset(tmp_path, per_class=20):
    """Create an ImageFolder-style dataset of tiny images on disk."""
    for label in ("normal", "pneumonia"):
        d = tmp_path / label
        d.mkdir()
        for i in range(per_class):
            img = np.random.default_rng(i).integers(0, 255, (16, 16), dtype=np.uint8)
            cv2.imwrite(str(d / f"patient{i:03d}.png"), img)
    return tmp_path


def test_build_and_roundtrip_manifest(tmp_path):
    root = _make_dataset(tmp_path)
    samples = build_manifest_from_folders(root)
    assert len(samples) == 40
    assert class_distribution(samples) == {"normal": 20, "pneumonia": 20}

    out = write_manifest(samples, tmp_path / "manifest.csv")
    reloaded = read_manifest(out)
    assert len(reloaded) == 40
    assert reloaded[0].label in {"normal", "pneumonia"}


def test_sample_level_stratified_split_keeps_all_samples():
    samples = [Sample(f"a{i}.png", "normal") for i in range(50)] + [
        Sample(f"b{i}.png", "pneumonia") for i in range(50)
    ]
    result = stratified_split(samples, val_size=0.2, test_size=0.2, seed=1)
    dist = split_distribution(result)
    assert set(dist) == {"train", "val", "test"}
    assert sum(sum(c.values()) for c in dist.values()) == 100
    # both classes represented in every split (stratification worked)
    for split in ("train", "val", "test"):
        assert set(dist[split]) == {"normal", "pneumonia"}


def test_patient_grouping_prevents_leakage():
    # 10 patients, 5 images each; label is per-patient so grouping is meaningful
    samples = []
    for p in range(10):
        label = "normal" if p < 5 else "pneumonia"
        for k in range(5):
            samples.append(Sample(f"p{p}_{k}.png", label, patient_id=f"P{p}"))
    result = stratified_split(samples, val_size=0.2, test_size=0.2, seed=7)
    assert_no_patient_leakage(result)  # raises if a patient crosses splits

    # every image of a patient shares one split
    by_patient = {}
    for s in result:
        by_patient.setdefault(s.patient_id, set()).add(s.split)
    assert all(len(splits) == 1 for splits in by_patient.values())


def test_registry_shape():
    assert "rsna_pneumonia" in REGISTRY
    spec = get_spec("rsna_pneumonia")
    assert spec.roi_support is True
    assert spec.modality == "Chest X-ray"
    assert len(recommended_v1()) == 2
    with pytest.raises(KeyError):
        get_spec("nope")


def test_multilabel_class_index_builds_vocab_from_delimited_labels():
    samples = [
        Sample("a.png", "Cardiomegaly|Effusion"),
        Sample("b.png", "Effusion"),
        Sample("c.png", "No Finding"),
        Sample("d.png", "Atelectasis|Cardiomegaly"),
    ]
    idx = build_multilabel_class_index(samples)
    assert set(idx) == {"Cardiomegaly", "Effusion", "No Finding", "Atelectasis"}
    # deterministic (sorted) regardless of input order
    assert idx == build_multilabel_class_index(list(reversed(samples)))


def test_build_manifest_from_csv_with_path_prefix(tmp_path):
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "Image Index,Finding Labels\n"
        "img1.png,Cardiomegaly|Effusion\n"
        "img2.png,No Finding\n"
    )
    samples = build_manifest_from_csv(
        csv_path,
        path_col="Image Index",
        label_col="Finding Labels",
        path_prefix=tmp_path / "images",
    )
    assert len(samples) == 2
    assert samples[0].path == str(tmp_path / "images" / "img1.png")
    assert samples[0].label == "Cardiomegaly|Effusion"
