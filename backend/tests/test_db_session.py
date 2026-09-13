"""Unit tests for Database Engine, Session Factory, and Initialization (Step 20)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from app.db.errors import DatabaseConfigurationError
from app.db.session import (
    create_db_engine,
    get_db,
    get_engine,
    get_session_factory,
    init_db,
    reset_db_engine,
)


class TestDBSession(unittest.TestCase):
    """Test suite for database sessions and engine lifecycle."""

    def setUp(self) -> None:
        reset_db_engine()

    def tearDown(self) -> None:
        reset_db_engine()

    def test_engine_initializes_and_caches(self) -> None:
        # Pass URL only on first call; second call uses cached singleton
        engine1 = get_engine("sqlite:///:memory:")
        engine2 = get_engine()  # no URL → returns cached singleton
        self.assertIs(engine1, engine2)
        self.assertEqual(engine1.dialect.name, "sqlite")

    def test_session_opens_and_closes(self) -> None:
        factory = get_session_factory("sqlite:///:memory:")
        session: Session = factory()
        self.assertIsInstance(session, Session)
        self.assertTrue(session.is_active)
        session.close()

    def test_init_db_creates_tables_idempotently(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            db_path = Path(tmp_dir) / "test_init.db"
            db_url = f"sqlite:///{db_path.as_posix()}"

            # First initialization
            init_db(db_url)
            engine = get_engine(db_url)
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            self.assertIn("agent_runs", tables)

            # Second initialization (must be idempotent without error)
            init_db(db_url)
            inspector2 = inspect(engine)
            self.assertIn("agent_runs", inspector2.get_table_names())

            # Dispose engine before temp dir cleanup to release Windows file lock
            engine.dispose()
            reset_db_engine()

    def test_get_db_generator_lifecycle(self) -> None:
        reset_db_engine()
        get_engine("sqlite:///:memory:")
        gen = get_db()
        session = next(gen)
        self.assertIsInstance(session, Session)
        self.assertTrue(session.is_active)
        with self.assertRaises(StopIteration):
            next(gen)

    def test_reset_db_engine_clears_singletons(self) -> None:
        e1 = get_engine("sqlite:///:memory:")
        reset_db_engine()
        e2 = get_engine("sqlite:///:memory:")
        self.assertIsNot(e1, e2)

    def test_invalid_database_url_raises_configuration_error(self) -> None:
        with self.assertRaises(DatabaseConfigurationError):
            create_db_engine("invalid_dialect://nonexistent_host:9999/testdb")


if __name__ == "__main__":
    unittest.main()
