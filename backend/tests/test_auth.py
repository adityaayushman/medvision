"""V4 auth/dashboard subsystem tests: org signup, JWT login, role gating,
org isolation, case-review transitions, audit trail, and the "separate pool"
guarantee that dashboard data never leaks into the public endpoints.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

_TMP = Path(tempfile.mkdtemp(prefix="medchron_test_auth_"))
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


def _png_bytes() -> bytes:
    img = np.full((256, 256, 3), 30, np.uint8)
    cv2.circle(img, (128, 128), 50, (220, 220, 220), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _signup(client, email: str, org_name: str = "Test Org", password: str = "hunter2") -> dict:
    r = client.post("/api/auth/signup", json={"org_name": org_name, "email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_signup_creates_org_and_admin(client):
    body = _signup(client, "admin1@orga.test", org_name="Org A")
    assert body["user"]["role"] == "admin"
    assert body["user"]["email"] == "admin1@orga.test"
    assert body["access_token"]


def test_signup_duplicate_email_rejected(client):
    _signup(client, "dupe@orga.test")
    r = client.post(
        "/api/auth/signup",
        json={"org_name": "Org A", "email": "dupe@orga.test", "password": "hunter2"},
    )
    assert r.status_code == 409


def test_login_success(client):
    _signup(client, "loginok@orga.test")
    r = client.post("/api/auth/login", json={"email": "loginok@orga.test", "password": "hunter2"})
    assert r.status_code == 200
    assert r.json()["user"]["email"] == "loginok@orga.test"


def test_login_wrong_password_rejected(client):
    _signup(client, "loginbad@orga.test")
    r = client.post("/api/auth/login", json={"email": "loginbad@orga.test", "password": "wrong"})
    assert r.status_code == 401


def test_login_unknown_email_rejected(client):
    r = client.post("/api/auth/login", json={"email": "nobody@nowhere.test", "password": "x"})
    assert r.status_code == 401


def test_me_requires_token(client):
    r = client.get("/api/auth/me")
    assert r.status_code == 401


def test_me_returns_current_user(client):
    body = _signup(client, "me@orga.test")
    r = client.get("/api/auth/me", headers=_auth(body["access_token"]))
    assert r.status_code == 200
    assert r.json()["email"] == "me@orga.test"


def test_admin_can_create_teammate(client):
    admin = _signup(client, "admin2@orgb.test", org_name="Org B")
    r = client.post(
        "/api/auth/users",
        json={"email": "radiologist@orgb.test", "password": "hunter2", "role": "radiologist"},
        headers=_auth(admin["access_token"]),
    )
    assert r.status_code == 200, r.text
    assert r.json()["role"] == "radiologist"
    assert r.json()["org_id"] == admin["user"]["org_id"]


def test_radiologist_cannot_create_teammate(client):
    admin = _signup(client, "admin3@orgc.test", org_name="Org C")
    client.post(
        "/api/auth/users",
        json={"email": "rad3@orgc.test", "password": "hunter2", "role": "radiologist"},
        headers=_auth(admin["access_token"]),
    )
    r = client.post("/api/auth/login", json={"email": "rad3@orgc.test", "password": "hunter2"})
    rad_headers = _auth(r.json()["access_token"])
    r = client.post(
        "/api/auth/users",
        json={"email": "another@orgc.test", "password": "hunter2", "role": "radiologist"},
        headers=rad_headers,
    )
    assert r.status_code == 403


def test_org_isolation_patients(client):
    orgA = _signup(client, "iso-a@orgd.test", org_name="Org D")
    orgB = _signup(client, "iso-b@orge.test", org_name="Org E")

    r = client.post("/api/dashboard/patients", json={"name": "Patient A"}, headers=_auth(orgA["access_token"]))
    assert r.status_code == 200
    patient_a_id = r.json()["id"]

    r = client.get("/api/dashboard/patients", headers=_auth(orgB["access_token"]))
    assert r.status_code == 200
    assert all(p["id"] != patient_a_id for p in r.json())

    r = client.get(f"/api/dashboard/patients/{patient_a_id}", headers=_auth(orgB["access_token"]))
    assert r.status_code == 404


def test_review_status_transitions(client):
    org = _signup(client, "reviewer@orgf.test", org_name="Org F")
    headers = _auth(org["access_token"])
    files = {"file": ("scan.png", _png_bytes(), "image/png")}
    r = client.post("/api/dashboard/studies/analyze", files=files, headers=headers)
    assert r.status_code == 200
    study_id = r.json()["study_id"]

    r = client.get(f"/api/dashboard/studies/{study_id}", headers=headers)
    assert r.json()["review_status"] == "pending"

    for status in ("in-review", "reviewed", "flagged"):
        r = client.patch(
            f"/api/dashboard/studies/{study_id}/review",
            json={"review_status": status, "note": f"moved to {status}"},
            headers=headers,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["review_status"] == status
        assert body["reviewed_by"] == "reviewer@orgf.test"
        assert body["reviewed_at"] is not None

    r = client.patch(
        f"/api/dashboard/studies/{study_id}/review",
        json={"review_status": "not-a-real-status"},
        headers=headers,
    )
    assert r.status_code == 422


def test_audit_log_entries_created(client):
    org = _signup(client, "auditor@orgg.test", org_name="Org G")
    headers = _auth(org["access_token"])

    r = client.post("/api/dashboard/patients", json={"name": "Audit Patient"}, headers=headers)
    patient_id = r.json()["id"]

    files = {"file": ("scan.png", _png_bytes(), "image/png")}
    r = client.post(
        "/api/dashboard/studies/analyze", files=files, data={"patient_id": str(patient_id)}, headers=headers
    )
    study_id = r.json()["study_id"]

    client.get(f"/api/dashboard/studies/{study_id}", headers=headers)
    client.patch(
        f"/api/dashboard/studies/{study_id}/review",
        json={"review_status": "in-review"},
        headers=headers,
    )

    r = client.get("/api/dashboard/audit", headers=headers)
    assert r.status_code == 200
    actions = [e["action"] for e in r.json()]
    assert "user.created" in actions
    assert "patient.created" in actions
    assert "study.created" in actions
    assert "study.viewed" in actions
    assert "study.review_status_changed" in actions
    assert all(e["actor_email"] == "auditor@orgg.test" for e in r.json())


def test_public_pool_unchanged(client):
    org = _signup(client, "pool@orgh.test", org_name="Org H")
    headers = _auth(org["access_token"])

    r = client.post("/api/dashboard/patients", json={"name": "Hidden Patient"}, headers=headers)
    patient_id = r.json()["id"]
    files = {"file": ("scan.png", _png_bytes(), "image/png")}
    r = client.post(
        "/api/dashboard/studies/analyze", files=files, data={"patient_id": str(patient_id)}, headers=headers
    )
    study_id = r.json()["study_id"]

    r = client.get("/api/patients")
    assert all(p["id"] != patient_id for p in r.json())

    r = client.get("/api/studies")
    assert all(s["id"] != study_id for s in r.json())

    r = client.get(f"/api/studies/{study_id}")
    assert r.status_code == 404
