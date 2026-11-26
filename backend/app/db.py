
from __future__ import annotations

import logging
import threading
import time

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

from .config import settings

_url = settings.database_url
if _url.startswith("postgres://"):
    _url = _url.replace("postgres://", "postgresql://", 1)

_is_sqlite = _url.startswith("sqlite")

log = logging.getLogger("medchron.db")

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
