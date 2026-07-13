"""Hermetic API tests.

Runs in preprocess-only mode (no trained model required), against a temp DB and
temp storage, so it always works in CI. Validates the full request path:
analyze -> persisted Study -> patient timeline.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Configure the app BEFORE importing it (settings is read at import time).
_TMP = Path(tempfile.mkdtemp(prefix="medchron_test_"))
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"
os.environ["STORAGE_DIR"] = str(_TMP / "storage")
os.environ["MODEL_CHECKPOINT"] = str(_TMP / "no_such_model.pt")  # force preprocess-only

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # context manager fires the lifespan handler -> tables created, model loaded
    with TestClient(app) as c:
        yield c


def _png_bytes() -> bytes:
    img = np.full((256, 256, 3), 30, np.uint8)
    cv2.circle(img, (128, 128), 50, (220, 220, 220), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False  # preprocess-only


def test_analyze_and_timeline(client):
    # create a patient
    r = client.post("/api/patients", json={"name": "Test Patient", "sex": "F", "birth_year": 1990})
    assert r.status_code == 200
    patient_id = r.json()["id"]

    # analyze a scan attached to that patient
    files = {"file": ("scan.png", _png_bytes(), "image/png")}
    r = client.post("/api/analyze", files=files, data={"patient_id": str(patient_id)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "quality" in body and "num_rois" in body
    assert body["prediction"] is None          # no model loaded
    assert body["image_url"].startswith("/api/image/")   # images stored in the DB
    assert body["study_id"] is not None
    # the full DIP pipeline gallery is returned (no model -> no Grad-CAM stage)
    stage_names = [s["name"] for s in body["stages"]]
    assert stage_names == ["Original", "Enhanced (CLAHE)", "Segmentation", "ROIs"]

    # each stage image is actually retrievable from the DB
    stage_url = body["stages"][0]["url"]
    r = client.get(stage_url)
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 100

    # the study shows up on the patient's timeline
    r = client.get(f"/api/patients/{patient_id}/timeline")
    assert r.status_code == 200
    timeline = r.json()
    assert len(timeline) == 1
    assert timeline[0]["id"] == body["study_id"]


def test_analyze_rejects_non_image(client):
    files = {"file": ("bad.txt", b"not an image", "text/plain")}
    r = client.post("/api/analyze", files=files)
    assert r.status_code == 400


def test_studies_lists_all_analyzed(client):
    # an analyzed scan (no patient attached) must still show up in /api/studies
    files = {"file": ("scan.png", _png_bytes(), "image/png")}
    r = client.post("/api/analyze", files=files)
    assert r.status_code == 200
    study_id = r.json()["study_id"]

    r = client.get("/api/studies")
    assert r.status_code == 200
    studies = r.json()
    assert any(s["id"] == study_id for s in studies)


def test_datasets_registry(client):
    r = client.get("/api/datasets")
    assert r.status_code == 200
    specs = r.json()
    assert any(s["key"] == "rsna_pneumonia" for s in specs)

    r = client.get("/api/datasets/recommended")
    assert r.status_code == 200
    assert len(r.json()) == 2
