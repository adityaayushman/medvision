
from __future__ import annotations

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

from .config import settings

_url = settings.database_url
if _url.startswith("postgres://"):
    _url = _url.replace("postgres://", "postgresql://", 1)

_is_sqlite = _url.startswith("sqlite")

engine = create_engine(
    _url,
    echo=False,
    pool_pre_ping=not _is_sqlite,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)


_NEW_STUDY_COLUMNS = [
    ("quality_score", "INTEGER"),
    ("analysis_stopped", "BOOLEAN"),
    ("model_version", "VARCHAR"),
    ("processing_time_ms", "FLOAT"),
    ("inference_time_ms", "FLOAT"),
    ("segmentation_success", "BOOLEAN"),
]


def _migrate_study_columns() -> None:
    for name, sql_type in _NEW_STUDY_COLUMNS:
        try:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE study ADD COLUMN {name} {sql_type}"))
        except Exception:
            pass


def init_db() -> None:
    from . import models_db
    SQLModel.metadata.create_all(engine)
    _migrate_study_columns()


def get_session():
    with Session(engine) as session:
        yield session
