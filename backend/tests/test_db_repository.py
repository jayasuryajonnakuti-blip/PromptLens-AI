"""Unit tests for AgentRunRepository (Step 20)."""
from __future__ import annotations

import unittest
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.errors import AgentRunNotFoundError, DatabaseIntegrityError
from app.db.models import AgentRun
from app.db.repositories import AgentRunRepository


class TestDBRepository(unittest.TestCase):
    """Test suite for AgentRunRepository CRUD and queries."""

    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.session: Session = self.session_factory()
        self.repo = AgentRunRepository(self.session)

    def tearDown(self) -> None:
        self.session.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    def _make_run(self, prompt: str = "Test prompt", **kwargs) -> AgentRun:
        params = {
            "original_prompt": prompt,
            "final_prompt": f"Optimized: {prompt}",
            "status": "COMPLETED",
            "termination_reason": "VALIDATED",
            "is_validated": True,
            "iteration_count": 1,
            "final_score": 85.0,
            "agent_version": "1.0.0",
            "mode": "balanced",
            "iterations": [],
            "score_history": [70.0, 85.0],
        }
        params.update(kwargs)
        return AgentRun(**params)

    def test_create_and_get_agent_run(self) -> None:
        run = self._make_run()
        created = self.repo.create(run)
        self.assertIsNotNone(created.id)

        retrieved = self.repo.get(created.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, created.id)
        self.assertEqual(retrieved.original_prompt, "Test prompt")

    def test_get_nonexistent_returns_none(self) -> None:
        self.assertIsNone(self.repo.get("nonexistent-uuid"))

    def test_get_or_raise_raises_not_found(self) -> None:
        with self.assertRaises(AgentRunNotFoundError):
            self.repo.get_or_raise("nonexistent-uuid")

    def test_list_agent_runs_ordering(self) -> None:
        t0 = datetime.now(timezone.utc)
        run1 = self._make_run(prompt="First", created_at=t0)
        run2 = self._make_run(prompt="Second", created_at=t0 + timedelta(seconds=10))
        run3 = self._make_run(prompt="Third", created_at=t0 + timedelta(seconds=20))

        self.repo.create(run1)
        self.repo.create(run2)
        self.repo.create(run3)

        runs = self.repo.list(skip=0, limit=10)
        self.assertEqual(len(runs), 3)
        # Most recent first
        self.assertEqual(runs[0].original_prompt, "Third")
        self.assertEqual(runs[1].original_prompt, "Second")
        self.assertEqual(runs[2].original_prompt, "First")

    def test_list_pagination(self) -> None:
        for i in range(5):
            self.repo.create(self._make_run(prompt=f"Prompt {i}"))

        page1 = self.repo.list(skip=0, limit=2)
        page2 = self.repo.list(skip=2, limit=2)
        self.assertEqual(len(page1), 2)
        self.assertEqual(len(page2), 2)
        self.assertNotEqual(page1[0].id, page2[0].id)

    def test_delete_agent_run(self) -> None:
        run = self._make_run()
        created = self.repo.create(run)
        run_id = created.id

        self.assertTrue(self.repo.delete(run_id))
        self.assertIsNone(self.repo.get(run_id))
        self.assertFalse(self.repo.delete(run_id))

    def test_count_agent_runs(self) -> None:
        self.assertEqual(self.repo.count(), 0)
        self.repo.create(self._make_run())
        self.repo.create(self._make_run())
        self.assertEqual(self.repo.count(), 2)

    def test_integrity_error_triggers_rollback(self) -> None:
        invalid_run = self._make_run(iteration_count=-5)
        with self.assertRaises(DatabaseIntegrityError):
            self.repo.create(invalid_run)

        # Ensure session remains healthy after rollback
        valid_run = self._make_run(iteration_count=1)
        created = self.repo.create(valid_run)
        self.assertIsNotNone(created.id)


if __name__ == "__main__":
    unittest.main()
