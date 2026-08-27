"""Database engine and session handling."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.config import get_settings
from app.models import Base

_settings = get_settings()


def _normalise_url(url: str) -> str:
    """Point old-style URLs at the driver actually installed.

    Hosted Postgres providers hand out `postgres://` and `postgresql://` URLs;
    SQLAlchemy needs a driver named explicitly, and psycopg 3 is what this
    project installs.
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def _engine_kwargs(url: str) -> dict:
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    if _settings.is_serverless:
        # A serverless instance is frozen between requests, so a pooled
        # connection is dead by the time it is reused and just consumes one of
        # Postgres' slots in the meantime. Connect per request instead and let
        # the provider's pooler do the pooling.
        return {"poolclass": NullPool, "pool_pre_ping": True}
    return {"pool_pre_ping": True, "pool_recycle": 1800}


DATABASE_URL = _normalise_url(_settings.database_url)

engine = create_engine(DATABASE_URL, future=True, **_engine_kwargs(DATABASE_URL))
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for background work."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
