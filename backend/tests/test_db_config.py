"""Unit tests for Database Configuration (Step 20)."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.db.config import (
    DEFAULT_DATABASE_URL,
    ensure_db_directory,
    get_database_url,
    is_sqlite_url,
)


class TestDBConfig(unittest.TestCase):
    """Test suite for database connection configuration."""

    def test_default_database_url(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            url = get_database_url()
            self.assertEqual(url, DEFAULT_DATABASE_URL)
            self.assertTrue(is_sqlite_url(url))

    def test_promptlens_database_url_env_override(self) -> None:
        custom_url = "sqlite:///./custom_data/custom.db"
        with patch.dict(os.environ, {"PROMPTLENS_DATABASE_URL": custom_url}, clear=True):
            self.assertEqual(get_database_url(), custom_url)

    def test_database_url_env_override(self) -> None:
        pg_url = "postgresql://user:pass@localhost:5432/promptlens"
        with patch.dict(os.environ, {"DATABASE_URL": pg_url}, clear=True):
            self.assertEqual(get_database_url(), pg_url)
            self.assertFalse(is_sqlite_url(pg_url))

    def test_is_sqlite_url_detection(self) -> None:
        self.assertTrue(is_sqlite_url("sqlite:///./data/promptlens.db"))
        self.assertTrue(is_sqlite_url("sqlite:///:memory:"))
        self.assertFalse(is_sqlite_url("postgresql://localhost/db"))
        self.assertFalse(is_sqlite_url("mysql://localhost/db"))

    def test_ensure_db_directory_creates_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_db = Path(tmp_dir) / "nested" / "sub" / "test.db"
            db_url = f"sqlite:///{nested_db.as_posix()}"
            self.assertFalse(nested_db.parent.exists())
            ensure_db_directory(db_url)
            self.assertTrue(nested_db.parent.exists())

    def test_ensure_db_directory_memory_noop(self) -> None:
        # Should execute cleanly without error
        ensure_db_directory("sqlite:///:memory:")
        ensure_db_directory("postgresql://localhost/db")


if __name__ == "__main__":
    unittest.main()
