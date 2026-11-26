"""The service must start, answer /health and fail fast when the database is unreachable.

A paused or unreachable hosted Postgres used to block every connection attempt for the operating
system's TCP timeout. Startup made fourteen attempts in a row, so the service never finished
starting and never answered a request. These tests simulate that failure without a real database.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="medchron_dbdown_"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP / 'test.db'}")
os.environ.setdefault("STORAGE_DIR", str(_TMP / "storage"))
os.environ.setdefault("MODEL_CHECKPOINT", str(_TMP / "no_such_model.pt"))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app import db, main


def _unreachable() -> OperationalError:
    return OperationalError("connect", {}, Exception("connection to server timed out"))


def test_hosted_databases_get_a_connect_timeout():
    assert db.connect_args_for("postgresql://u:p@host/db") == {"connect_timeout": db.CONNECT_TIMEOUT_S}
    assert db.connect_args_for("sqlite:///x.db") == {"check_same_thread": False}
    assert 0 < db.CONNECT_TIMEOUT_S <= 15


def test_init_db_reports_failure_instead_of_raising(monkeypatch):
    monkeypatch.setattr(db.SQLModel.metadata, "create_all", lambda engine: (_ for _ in ()).throw(_unreachable()))
    migrations = []
    for name in ("_migrate_study_columns", "_migrate_patient_columns", "_migrate_prediction_columns"):
        monkeypatch.setattr(db, name, lambda n=name: migrations.append(n))

    assert db.init_db() is False
    assert db.db_ready() is False
    # Each migration opens its own connection, so none may run against an unreachable host.
    assert migrations == []


def test_init_db_runs_the_migrations_once_the_tables_exist(monkeypatch):
    monkeypatch.setattr(db.SQLModel.metadata, "create_all", lambda engine: None)
    migrations = []
    for name in ("_migrate_study_columns", "_migrate_patient_columns", "_migrate_prediction_columns"):
        monkeypatch.setattr(db, name, lambda n=name: migrations.append(n))
    monkeypatch.setattr(db, "_db_ready", False)

    assert db.init_db() is True
    assert db.db_ready() is True
    assert len(migrations) == 3


def test_the_service_starts_and_answers_health_with_the_database_down(monkeypatch):
    retries = []
    monkeypatch.setattr(main, "init_db", lambda: False)
    monkeypatch.setattr(main, "retry_init_in_background", lambda: retries.append(True))
    monkeypatch.setattr(db, "_db_ready", False)

    with TestClient(main.app) as client:
        r = client.get("/health")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["database"] == "unavailable"
    # It keeps trying in the background, so it recovers without a restart.
    assert retries == [True]


def test_health_is_ok_when_the_database_is_ready(monkeypatch):
    monkeypatch.setattr(db, "_db_ready", True)
    with TestClient(main.app) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"


def test_database_errors_during_a_request_are_a_503_not_a_hang(monkeypatch):
    @main.app.get("/_test_database_down")
    def _boom():
        raise _unreachable()

    try:
        with TestClient(main.app, raise_server_exceptions=False) as client:
            r = client.get("/_test_database_down")
    finally:
        main.app.router.routes[:] = [
            route for route in main.app.router.routes if getattr(route, "path", "") != "/_test_database_down"
        ]

    assert r.status_code == 503
    assert "database unavailable" in r.json()["detail"]


def test_background_retry_stops_once_the_database_answers(monkeypatch):
    attempts = []

    def fake_init() -> bool:
        attempts.append(1)
        if len(attempts) >= 2:
            monkeypatch.setattr(db, "_db_ready", True)
            return True
        return False

    monkeypatch.setattr(db, "_db_ready", False)
    monkeypatch.setattr(db, "init_db", fake_init)
    thread = db.retry_init_in_background(interval_s=0.01)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(attempts) == 2
