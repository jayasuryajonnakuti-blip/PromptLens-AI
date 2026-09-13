"""Unit tests for AgentRun ORM Model and Database Constraints (Step 20)."""
from __future__ import annotations

import unittest
import uuid
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.models import AgentRun, generate_uuid, utc_now


class TestDBModels(unittest.TestCase):
    """Test suite for AgentRun ORM model and SQLite constraints."""

    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.session: Session = self.session_factory()

    def tearDown(self) -> None:
        self.session.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    def _create_valid_run(self, **kwargs) -> AgentRun:
        params = {
            "original_prompt": "Write a Python sorting function.",
            "final_prompt": "Write a Python quicksort implementation.",
            "status": "COMPLETED",
            "termination_reason": "VALIDATED",
            "is_validated": True,
            "iteration_count": 1,
            "final_score": 88.5,
            "agent_version": "1.0.0",
            "mode": "balanced",
            "iterations": [{"iteration_number": 1, "decision": "PASS"}],
            "score_history": [70.0, 88.5],
            "critic_result": {"decision": "PASS", "score": 88.5},
            "final_validation": {"decision": "PASS", "is_valid": True},
            "metrics": {"total_latency_ms": 120.0},
            "run_metadata": {"disclaimers": ["Optimization is bounded."]},
        }
        params.update(kwargs)
        return AgentRun(**params)

    def test_agent_run_creation_and_defaults(self) -> None:
        run = self._create_valid_run()
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)

        self.assertIsNotNone(run.id)
        # Verify valid UUID format
        parsed_uuid = uuid.UUID(run.id)
        self.assertEqual(str(parsed_uuid), run.id)

        self.assertIsInstance(run.created_at, datetime)
        self.assertIsInstance(run.updated_at, datetime)
        self.assertEqual(run.status, "COMPLETED")
        self.assertEqual(run.iteration_count, 1)
        self.assertEqual(run.final_score, 88.5)
        self.assertTrue(run.is_validated)

    def test_timestamps_are_utc(self) -> None:
        now = utc_now()
        self.assertIsNotNone(now.tzinfo)
        self.assertEqual(now.tzinfo, timezone.utc)

    def test_uuid_generator_generates_unique_valid_uuids(self) -> None:
        u1 = generate_uuid()
        u2 = generate_uuid()
        self.assertNotEqual(u1, u2)
        self.assertEqual(len(u1), 36)

    def test_final_score_boundary_zero_accepted(self) -> None:
        run = self._create_valid_run(final_score=0.0)
        self.session.add(run)
        self.session.commit()
        self.assertEqual(run.final_score, 0.0)

    def test_final_score_boundary_hundred_accepted(self) -> None:
        run = self._create_valid_run(final_score=100.0)
        self.session.add(run)
        self.session.commit()
        self.assertEqual(run.final_score, 100.0)

    def test_final_score_nullable_accepted(self) -> None:
        run = self._create_valid_run(final_score=None)
        self.session.add(run)
        self.session.commit()
        self.assertIsNone(run.final_score)

    def test_negative_final_score_rejected(self) -> None:
        run = self._create_valid_run(final_score=-0.5)
        self.session.add(run)
        with self.assertRaises(IntegrityError):
            self.session.commit()
        self.session.rollback()

    def test_final_score_over_hundred_rejected(self) -> None:
        run = self._create_valid_run(final_score=100.1)
        self.session.add(run)
        with self.assertRaises(IntegrityError):
            self.session.commit()
        self.session.rollback()

    def test_negative_iteration_count_rejected(self) -> None:
        run = self._create_valid_run(iteration_count=-1)
        self.session.add(run)
        with self.assertRaises(IntegrityError):
            self.session.commit()
        self.session.rollback()

    def test_json_structured_data_persists(self) -> None:
        complex_iterations = [
            {"num": 1, "nested": {"key": "val1"}},
            {"num": 2, "nested": {"key": "val2"}},
        ]
        run = self._create_valid_run(iterations=complex_iterations)
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)

        self.assertEqual(len(run.iterations), 2)
        self.assertEqual(run.iterations[1]["nested"]["key"], "val2")


if __name__ == "__main__":
    unittest.main()
