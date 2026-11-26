
from __future__ import annotations

import importlib.util
import logging
import threading
import time

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

from .config import settings

log = logging.getLogger("medchron.db")


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def resolve_database_url(url: str) -> str:
    """Return a SQLAlchemy URL whose PostgreSQL driver is actually installed.

    SQLAlchemy 2.1 changed the default driver for ``postgresql://`` from psycopg2 to psycopg
    (version 3). An image that only has psycopg2 then crashes on import with
    ``No module named 'psycopg'`` before it serves a single request. Naming the driver
    explicitly, from whichever is installed, makes that independent of the SQLAlchemy default.
    Other drivers and non-PostgreSQL URLs are left exactly as given.
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    scheme, sep, rest = url.partition("://")
    if not sep or not scheme.startswith("postgresql"):
        return url
    requested = scheme.partition("+")[2]
    if requested and requested not in ("psycopg2", "psycopg"):
        return url
    order = ([requested] if requested else []) + [d for d in ("psycopg2", "psycopg") if d != requested]
    for driver in order:
        if _module_available(driver):
            if requested and driver != requested:
                log.warning("driver %s is not installed; using %s instead", requested, driver)
            return f"postgresql+{driver}://{rest}"
    return url  # no driver installed: let SQLAlchemy report it in its usual way


_url = resolve_database_url(settings.database_url)

_is_sqlite = _url.startswith("sqlite")

# Seconds to wait when opening a connection to a hosted database. Without a limit, a paused
# or unreachable Postgres makes every connection attempt block for the operating system's TCP
# timeout (minutes), so the service never finishes starting and never answers a request.
CONNECT_TIMEOUT_S = 8


def connect_args_for(url: str) -> dict:
    if url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {"connect_timeout": CONNECT_TIMEOUT_S}


engine = create_engine(
    _url,
    echo=False,
    pool_pre_ping=not _is_sqlite,
    connect_args=connect_args_for(_url),
)

_db_ready = False


def db_ready() -> bool:
    """True once the tables exist and the database has answered."""
    return _db_ready


_NEW_STUDY_COLUMNS = [
    ("quality_score", "INTEGER"),
    ("analysis_stopped", "BOOLEAN"),
    ("model_version", "VARCHAR"),
    ("processing_time_ms", "FLOAT"),
    ("inference_time_ms", "FLOAT"),
    ("segmentation_success", "BOOLEAN"),
    ("org_id", "INTEGER"),
    ("review_status", "VARCHAR"),
    ("reviewed_by_user_id", "INTEGER"),
    ("review_note", "VARCHAR"),
    ("reviewed_at", "TIMESTAMP"),
]

_NEW_PATIENT_COLUMNS = [
    ("org_id", "INTEGER"),
]

_NEW_PREDICTION_COLUMNS = [
    ("per_model", "VARCHAR"),
]


def _migrate_study_columns() -> None:
    for name, sql_type in _NEW_STUDY_COLUMNS:
        try:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE study ADD COLUMN {name} {sql_type}"))
        except Exception:
            pass


def _migrate_patient_columns() -> None:
    for name, sql_type in _NEW_PATIENT_COLUMNS:
        try:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE patient ADD COLUMN {name} {sql_type}"))
        except Exception:
            pass


def _migrate_prediction_columns() -> None:
    for name, sql_type in _NEW_PREDICTION_COLUMNS:
        try:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE prediction ADD COLUMN {name} {sql_type}"))
        except Exception:
            pass


def init_db() -> bool:
    """Create the tables and apply the small column migrations.

    Returns False, without raising, when the database cannot be reached. The service can then
    still start and answer /health, and database routes answer 503 until it is back. The column
    migrations are skipped in that case: each opens its own connection, so running them against
    an unreachable host would multiply the wait.
    """
    global _db_ready
    from . import models_db
    try:
        SQLModel.metadata.create_all(engine)
    except Exception as exc:  # an unreachable host is an OperationalError, but nothing may stop startup
        _db_ready = False
        detail = (str(exc).splitlines() or [""])[0][:200]
        log.warning("database unavailable, starting without it: %s: %s", type(exc).__name__, detail)
        return False
    _migrate_study_columns()
    _migrate_patient_columns()
    _migrate_prediction_columns()
    _db_ready = True
    return True


def retry_init_in_background(interval_s: float = 30.0) -> threading.Thread:
    """Keep trying to initialise the database until it succeeds, so the service recovers on its own."""

    def run() -> None:
        while not _db_ready:
            time.sleep(interval_s)
            if init_db():
                log.info("database reachable again, initialised")

    thread = threading.Thread(target=run, name="db-retry", daemon=True)
    thread.start()
    return thread


def get_session():
    with Session(engine) as session:
        yield session
