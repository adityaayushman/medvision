"""Verifies Prediction.per_model round-trips: analyzer -> _analyze_and_persist
-> DB -> _study_to_read -> StudyRead.prediction.per_model.

Runs in the same hermetic preprocess-only setup as test_api.py, but overrides
the get_analyzer dependency so the real preprocessing pipeline runs while the
prediction itself is a fabricated ensemble result — no real trained
checkpoint needed to test the persistence path.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

_TMP = Path(tempfile.mkdtemp(prefix="medchron_test_ensemble_"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP / 'test.db'}")
os.environ.setdefault("STORAGE_DIR", str(_TMP / "storage"))
os.environ.setdefault("MODEL_CHECKPOINT", str(_TMP / "no_such_model.pt"))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from app.main import app
from app.ml import get_analyzer


def _png_bytes() -> bytes:
    img = np.full((256, 256, 3), 30, np.uint8)
    cv2.circle(img, (128, 128), 50, (220, 220, 220), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class _FakeEnsembleAnalyzer:
    """Wraps the real analyzer's preprocessing, but injects a fabricated
    ensemble prediction (per_model breakdown) — exercises the persistence
    path without needing a real trained checkpoint."""

    def __init__(self, real_analyzer):
        self._real = real_analyzer

    @property
    def model_loaded(self):
        return True

    def available_modalities(self):
        return self._real.available_modalities()

    def analyze(self, image_bgr, modality=None):
        payload, result, _overlay = self._real.analyze(image_bgr, modality=modality)
        payload["model_loaded"] = True
        payload["prediction"] = {
            "label": "glioma_tumor",
            "confidence": 0.61,
            "probabilities": {
                "glioma_tumor": 0.61, "meningioma_tumor": 0.20,
                "no_tumor": 0.11, "pituitary_tumor": 0.08,
            },
            "backbone": "ensemble",
            "per_model": [
                {"backbone": "efficientnet_b0", "label": "glioma_tumor", "confidence": 0.58},
                {"backbone": "resnet50", "label": "glioma_tumor", "confidence": 0.70},
                {"backbone": "densenet121", "label": "meningioma_tumor", "confidence": 0.55},
            ],
        }
        overlay = result.annotated  # any real array works as the "gradcam" stage stand-in
        return payload, result, overlay


@pytest.fixture(scope="module")
def client():
    real_analyzer = get_analyzer()
    app.dependency_overrides[get_analyzer] = lambda: _FakeEnsembleAnalyzer(real_analyzer)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_analyzer, None)


def test_per_model_round_trips_through_persistence_and_read(client):
    files = {"file": ("scan.png", _png_bytes(), "image/png")}
    r = client.post("/api/analyze", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"]["per_model"] == [
        {"backbone": "efficientnet_b0", "label": "glioma_tumor", "confidence": 0.58},
        {"backbone": "resnet50", "label": "glioma_tumor", "confidence": 0.70},
        {"backbone": "densenet121", "label": "meningioma_tumor", "confidence": 0.55},
    ]
    study_id = body["study_id"]

    r = client.get(f"/api/studies/{study_id}")
    assert r.status_code == 200
    read_body = r.json()
    assert read_body["prediction"]["per_model"] == body["prediction"]["per_model"]
    assert read_body["prediction"]["backbone"] == "ensemble"


def test_report_includes_per_model_breakdown(client):
    files = {"file": ("scan.png", _png_bytes(), "image/png")}
    r = client.post("/api/analyze", files=files)
    study_id = r.json()["study_id"]

    r = client.get(f"/api/studies/{study_id}/report")
    assert r.status_code == 200
    report = r.json()
    assert len(report["prediction"]["per_model"]) == 3

    r = client.get(f"/api/studies/{study_id}/report.pdf")
    assert r.status_code == 200
    assert r.content[:5] == b"%PDF-"
