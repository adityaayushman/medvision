"""Database engine + session helpers (SQLModel over SQLite locally, Postgres in
production via DATABASE_URL — e.g. a Supabase connection string)."""

from __future__ import annotations

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

from .config import settings

# SQLAlchemy needs the `postgresql://` scheme; some providers hand out the
# deprecated `postgres://`. Normalise it so a pasted Supabase URL just works.
_url = settings.database_url
if _url.startswith("postgres://"):
    _url = _url.replace("postgres://", "postgresql://", 1)

_is_sqlite = _url.startswith("sqlite")

# SQLite needs check_same_thread=False; Postgres pooled connections benefit from
# pre-ping (drops dead connections after the host sleeps/restarts).
engine = create_engine(
    _url,
    echo=False,
    pool_pre_ping=not _is_sqlite,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)


# create_all() only creates *missing* tables — it never alters an existing
# table's columns. New nullable columns on an already-deployed table (e.g.
# Study gaining quality_score) need an explicit, idempotent ALTER TABLE.
_NEW_STUDY_COLUMNS = [
    ("quality_score", "INTEGER"),
    ("analysis_stopped", "BOOLEAN"),
    ("model_version", "VARCHAR"),
    ("processing_time_ms", "FLOAT"),
    ("inference_time_ms", "FLOAT"),
    ("segmentation_success", "BOOLEAN"),
]


def _migrate_study_columns() -> None:
    # Each ALTER gets its own transaction: on Postgres, one failed statement
    # (column already exists) aborts the whole transaction, which would silently
    # skip every later column if they all shared one `engine.begin()` block.
    for name, sql_type in _NEW_STUDY_COLUMNS:
        try:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE study ADD COLUMN {name} {sql_type}"))
        except Exception:
            pass  # column already exists — fine on both SQLite and Postgres


def init_db() -> None:
    # import models so their tables register on SQLModel.metadata
    from . import models_db  # noqa: F401
    SQLModel.metadata.create_all(engine)
    _migrate_study_columns()


def get_session():
    with Session(engine) as session:
        yield session
