"""The PostgreSQL driver must be one that is installed, whatever SQLAlchemy's default is.

SQLAlchemy 2.1 changed the default for postgresql:// from psycopg2 to psycopg 3. A deployed image
that only had psycopg2 then crashed on import ("No module named 'psycopg'") and Render restarted
it forever. These tests cover every combination of requested and installed drivers.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="medchron_dburl_"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP / 'test.db'}")
os.environ.setdefault("STORAGE_DIR", str(_TMP / "storage"))
os.environ.setdefault("MODEL_CHECKPOINT", str(_TMP / "no_such_model.pt"))

pytest.importorskip("sqlalchemy")
from app import db

PLAIN = "postgresql://user:pa%40ss@db.example.com:5432/postgres"
REST = "user:pa%40ss@db.example.com:5432/postgres"


def installed(monkeypatch, *names: str) -> None:
    monkeypatch.setattr(db, "_module_available", lambda name: name in names)


def test_plain_url_uses_psycopg2_when_only_psycopg2_is_installed(monkeypatch):
    installed(monkeypatch, "psycopg2")
    assert db.resolve_database_url(PLAIN) == f"postgresql+psycopg2://{REST}"


def test_plain_url_uses_psycopg_when_only_psycopg_is_installed(monkeypatch):
    installed(monkeypatch, "psycopg")
    assert db.resolve_database_url(PLAIN) == f"postgresql+psycopg://{REST}"


def test_plain_url_prefers_psycopg2_when_both_are_installed(monkeypatch):
    installed(monkeypatch, "psycopg2", "psycopg")
    assert db.resolve_database_url(PLAIN) == f"postgresql+psycopg2://{REST}"


def test_heroku_style_scheme_is_normalised(monkeypatch):
    installed(monkeypatch, "psycopg2")
    assert db.resolve_database_url("postgres://" + REST) == f"postgresql+psycopg2://{REST}"


def test_a_requested_driver_is_kept_when_installed(monkeypatch):
    installed(monkeypatch, "psycopg2", "psycopg")
    assert db.resolve_database_url(f"postgresql+psycopg://{REST}") == f"postgresql+psycopg://{REST}"
    assert db.resolve_database_url(f"postgresql+psycopg2://{REST}") == f"postgresql+psycopg2://{REST}"


def test_a_requested_driver_that_is_missing_falls_back_to_the_installed_one(monkeypatch):
    installed(monkeypatch, "psycopg2")
    assert db.resolve_database_url(f"postgresql+psycopg://{REST}") == f"postgresql+psycopg2://{REST}"
    installed(monkeypatch, "psycopg")
    assert db.resolve_database_url(f"postgresql+psycopg2://{REST}") == f"postgresql+psycopg://{REST}"


def test_the_password_and_host_are_never_altered(monkeypatch):
    installed(monkeypatch, "psycopg")
    out = db.resolve_database_url("postgresql://u:p%2Fq%3F@h-1.example.co:6543/db?sslmode=require")
    assert out == "postgresql+psycopg://u:p%2Fq%3F@h-1.example.co:6543/db?sslmode=require"


def test_other_drivers_and_databases_are_left_alone(monkeypatch):
    installed(monkeypatch, "psycopg2", "psycopg")
    for url in ("sqlite:///x.db", "sqlite://", "postgresql+asyncpg://u:p@h/db", "mysql+pymysql://u:p@h/db"):
        assert db.resolve_database_url(url) == url


def test_with_no_driver_installed_the_url_is_left_for_sqlalchemy_to_report(monkeypatch):
    installed(monkeypatch)
    assert db.resolve_database_url(PLAIN) == PLAIN


def test_the_resolved_url_selects_the_driver_it_names(monkeypatch):
    from sqlalchemy.engine.url import make_url

    installed(monkeypatch, "psycopg")
    assert make_url(db.resolve_database_url(PLAIN)).get_dialect().driver == "psycopg"
    installed(monkeypatch, "psycopg2")
    assert make_url(db.resolve_database_url(PLAIN)).get_dialect().driver == "psycopg2"
