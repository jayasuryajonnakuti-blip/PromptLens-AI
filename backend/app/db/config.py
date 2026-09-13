"""Database configuration and connection settings (Step 20).

Invariants:
- SQLite is the default local persistence engine.
- DATABASE_URL can point to any SQLAlchemy 2.x supported database (e.g. PostgreSQL in the future).
- Local development works out-of-the-box without requiring an environment variable.
- Relative SQLite paths automatically ensure parent directory creation.
"""
from __future__ import annotations

import os
from pathlib import Path

# Default database path relative to backend root
DEFAULT_DB_REL_PATH = "./data/promptlens.db"
DEFAULT_DATABASE_URL = f"sqlite:///{DEFAULT_DB_REL_PATH}"


def get_database_url() -> str:
    """Return the active database connection URL.

    Priority:
    1. PROMPTLENS_DATABASE_URL
    2. DATABASE_URL
    3. Default: sqlite:///./data/promptlens.db
    """
    return (
        os.getenv("PROMPTLENS_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or DEFAULT_DATABASE_URL
    )


def is_sqlite_url(url: str) -> bool:
    """Return True if the database URL uses SQLite."""
    return url.startswith("sqlite:")


def ensure_db_directory(url: str) -> None:
    """Ensure the target directory for a SQLite database exists.

    No-op for in-memory SQLite (':memory:') or non-SQLite database URLs.
    """
    if not is_sqlite_url(url):
        return

    # Extract file path from sqlite:///path or sqlite:////abs/path
    cleaned = url.replace("sqlite:///", "", 1)
    if not cleaned or cleaned == ":memory:":
        return

    db_path = Path(cleaned)
    if not db_path.is_absolute():
        # Resolve relative to working directory or backend root
        db_path = Path.cwd() / db_path

    db_path.parent.mkdir(parents=True, exist_ok=True)
