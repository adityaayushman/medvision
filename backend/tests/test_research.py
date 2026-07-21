"""Research Workspace tests: role gating (radiologist excluded, admin and
researcher allowed), create/list round-trip, metrics JSON reshape, and that
runs are platform-wide (not org-scoped) unlike Patient/Study.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="medchron_test_research_"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP / 'test.db'}")
os.environ.setdefault("STORAGE_DIR", str(_TMP / "storage"))
os.environ.setdefault("MODEL_CHECKPOINT", str(_TMP / "no_such_model.pt"))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _signup(client, email: str, org_name: str) -> dict:
    r = client.post("/api/auth/signup", json={"org_name": org_name, "email": email, "password": "hunter2"})
    assert r.status_code == 200, r.text
    return r.json()


def _add_teammate(client, admin_token: str, email: str, role: str) -> dict:
    r = client.post(
        "/api/auth/users",
        json={"email": email, "password": "hunter2", "role": role},
        headers=_auth(admin_token),
    )
    assert r.status_code == 200, r.text
    r = client.post("/api/auth/login", json={"email": email, "password": "hunter2"})
    assert r.status_code == 200
    return r.json()


_SAMPLE_METRICS = {
    "accuracy": 0.711,
    "roc_auc": 0.785,
    "per_class": {"Benign": {"precision": 0.8, "recall": 0.7}},
}


def test_researcher_role_accepted_on_signup_teammate(client):
    admin = _signup(client, "admin1@researchorg.test", "Research Org")
    r = client.post(
        "/api/auth/users",
        json={"email": "researcher1@researchorg.test", "password": "hunter2", "role": "researcher"},
        headers=_auth(admin["access_token"]),
    )
    assert r.status_code == 200, r.text
    assert r.json()["role"] == "researcher"


def test_radiologist_cannot_create_or_list_runs(client):
    admin = _signup(client, "admin2@researchorg2.test", "Research Org 2")
    rad = _add_teammate(client, admin["access_token"], "rad@researchorg2.test", "radiologist")

    r = client.post(
        "/api/research/runs",
        json={"kind": "classification", "modality": "mammography", "backbone": "efficientnet_b0",
              "label": "test run", "metrics": _SAMPLE_METRICS},
        headers=_auth(rad["access_token"]),
    )
    assert r.status_code == 403

    r = client.get("/api/research/runs", headers=_auth(rad["access_token"]))
    assert r.status_code == 403


def test_researcher_can_create_and_list_runs(client):
    admin = _signup(client, "admin3@researchorg3.test", "Research Org 3")
    researcher = _add_teammate(client, admin["access_token"], "researcher@researchorg3.test", "researcher")

    r = client.post(
        "/api/research/runs",
        json={
            "kind": "classification", "modality": "mammography", "backbone": "efficientnet_b0",
            "label": "CBIS-DDSM cropped-patch", "metrics": _SAMPLE_METRICS,
            "notes": "real win, needs a cropped input",
        },
        headers=_auth(researcher["access_token"]),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] == "CBIS-DDSM cropped-patch"
    assert body["metrics"] == _SAMPLE_METRICS
    assert body["created_by_email"] == "researcher@researchorg3.test"

    r = client.get("/api/research/runs", headers=_auth(admin["access_token"]))
    assert r.status_code == 200
    labels = [run["label"] for run in r.json()]
    assert "CBIS-DDSM cropped-patch" in labels


def test_runs_are_platform_wide_not_org_scoped(client):
    """Unlike Patient/Study, a run created in one org must be visible to an
    admin in a completely different org -- experiments are about the
    platform's models, not a hospital's patient data."""
    admin_a = _signup(client, "admin4@researchorg4.test", "Research Org 4")
    client.post(
        "/api/research/runs",
        json={"kind": "segmentation", "modality": "mammography", "backbone": "efficientnet_b0",
              "label": "cross-org visible run", "metrics": {"mean_dice": 0.254}},
        headers=_auth(admin_a["access_token"]),
    )

    admin_b = _signup(client, "admin5@researchorg5.test", "Research Org 5")
    r = client.get("/api/research/runs", headers=_auth(admin_b["access_token"]))
    assert r.status_code == 200
    labels = [run["label"] for run in r.json()]
    assert "cross-org visible run" in labels


def test_filter_by_kind_and_modality(client):
    admin = _signup(client, "admin6@researchorg6.test", "Research Org 6")
    headers = _auth(admin["access_token"])
    client.post(
        "/api/research/runs",
        json={"kind": "bbox_regression", "modality": "mammography", "backbone": "efficientnet_b0",
              "label": "bbox attempt 3", "metrics": {"mean_iou": 0.068}},
        headers=headers,
    )
    client.post(
        "/api/research/runs",
        json={"kind": "classification", "modality": "brain_mri", "backbone": "efficientnet_b0",
              "label": "brain mri v1", "metrics": {"accuracy": 0.82}},
        headers=headers,
    )

    r = client.get("/api/research/runs?kind=bbox_regression", headers=headers)
    assert all(run["kind"] == "bbox_regression" for run in r.json())

    r = client.get("/api/research/runs?modality=brain_mri", headers=headers)
    assert all(run["modality"] == "brain_mri" for run in r.json())
