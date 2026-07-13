"""Database engine + session helpers (SQLModel over SQLite locally, Postgres in
production via DATABASE_URL — e.g. a Supabase connection string)."""

from __future__ import annotations

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


def init_db() -> None:
    # import models so their tables register on SQLModel.metadata
    from . import models_db  # noqa: F401
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
