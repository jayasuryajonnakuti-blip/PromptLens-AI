"""Engine, session management, and database initialization (Step 20).

Invariants:
- Sessions are short-lived, request-scoped, and explicitly closed.
- Global mutable sessions are strictly forbidden.
- SQLite connections use check_same_thread=False for FastAPI concurrency.
- init_db() creates tables idempotently and never executes drop_all().
"""
from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.config import ensure_db_directory, get_database_url, is_sqlite_url
from app.db.errors import DatabaseConfigurationError

LOGGER = logging.getLogger(__name__)

_ENGINE: Engine | None = None
_SESSION_FACTORY: sessionmaker[Session] | None = None


def create_db_engine(database_url: str | None = None, **kwargs: Any) -> Engine:
    """Create and configure a new SQLAlchemy Engine."""
    url = database_url or get_database_url()

    connect_args: dict[str, Any] = {}
    if is_sqlite_url(url):
        ensure_db_directory(url)
        connect_args["check_same_thread"] = False

    # Merge connect_args if passed in kwargs
    if "connect_args" in kwargs:
        connect_args.update(kwargs.pop("connect_args"))

    try:
        engine = create_engine(
            url,
            connect_args=connect_args,
            **kwargs,
        )
        return engine
    except Exception as exc:
        raise DatabaseConfigurationError(
            f"Failed to create database engine for URL '{url}': {exc}"
        ) from exc


def get_engine(database_url: str | None = None) -> Engine:
    """Return the cached database engine singleton, initializing it if necessary."""
    global _ENGINE
    if _ENGINE is None or database_url is not None:
        _ENGINE = create_db_engine(database_url)
    return _ENGINE


def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    """Return the cached sessionmaker factory."""
    global _SESSION_FACTORY
    if _SESSION_FACTORY is None or database_url is not None:
        engine = get_engine(database_url)
        _SESSION_FACTORY = sessionmaker(
            bind=engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _SESSION_FACTORY


def reset_db_engine() -> None:
    """Dispose and reset the cached engine and session factory (for test isolation)."""
    global _ENGINE, _SESSION_FACTORY
    if _ENGINE is not None:
        try:
            _ENGINE.dispose()
        except Exception:
            pass
    _ENGINE = None
    _SESSION_FACTORY = None


def init_db(database_url: str | None = None) -> None:
    """Initialize database tables idempotently.

    Ensures the target directory exists and executes create_all().
    Never executes drop_all().
    """
    url = database_url or get_database_url()
    LOGGER.info("Initializing database at: %s", url)

    if is_sqlite_url(url):
        ensure_db_directory(url)

    engine = get_engine(url)
    try:
        Base.metadata.create_all(bind=engine)
        LOGGER.info("Database schema initialized successfully.")
    except Exception as exc:
        LOGGER.error("Database table initialization failed: %s", exc, exc_info=True)
        raise DatabaseConfigurationError(f"Database initialization failed: {exc}") from exc


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a short-lived request-scoped database session."""
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
    finally:
        session.close()
