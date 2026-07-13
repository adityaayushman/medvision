"""Database engine + session helpers (SQLModel over SQLite)."""

from __future__ import annotations

from sqlmodel import Session, SQLModel, create_engine

from .config import settings

# check_same_thread=False lets FastAPI's threadpool share the SQLite connection
engine = create_engine(
    settings.database_url,
    echo=False,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)


def init_db() -> None:
    # import models so their tables register on SQLModel.metadata
    from . import models_db  # noqa: F401
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
