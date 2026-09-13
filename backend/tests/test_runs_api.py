"""Unit and integration tests for Agent Run History API endpoints (Step 21)."""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
import uuid
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import AgentRun
from app.db.session import reset_db_engine
from app.main import app
from app.services.agent_run_service import AgentRunService, set_agent_run_service


class TestRunsAPI(unittest.TestCase):
    """Test suite for GET /api/v1/runs, GET /api/v1/runs/{run_id}, and DELETE /api/v1/runs/{run_id}."""

    def setUp(self) -> None:
        reset_db_engine()
        set_agent_run_service(None)

        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        self.run_service = AgentRunService(session_factory=self.session_factory)
        set_agent_run_service(self.run_service)

        self.client = TestClient(app)

    def tearDown(self) -> None:
        set_agent_run_service(None)
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()
        reset_db_engine()

    def _create_sample_run(self, run_id: str | None = None) -> AgentRun:
        run = AgentRun(
            id=run_id or str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            original_prompt="Summarize this quarterly earnings report for executives.",
            final_prompt="Generate an executive summary of the quarterly earnings report...",
            status="COMPLETED",
            termination_reason="VALIDATED",
            is_validated=True,
            iteration_count=1,
            final_score=88.5,
            agent_version="1.0.0",
            mode="balanced",
            iterations=[
                {
                    "iteration_number": 1,
                    "score_before": 68.0,
                    "score_after": 88.5,
                    "decision": "PASS",
                }
            ],
            score_history=[68.0, 88.5],
            critic_result={"overall_critique_score": 85.0, "decision": "PASS"},
            final_validation={"is_valid": True, "decision": "PASS"},
            metrics={"total_latency_ms": 125.0},
            run_metadata={"disclaimers": ["Optimization is iterative, but bounded."]},
        )
        with self.session_factory() as session:
            session.add(run)
            session.commit()
            session.refresh(run)
        return run

    def test_list_runs_empty(self) -> None:
        response = self.client.get("/api/v1/runs")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["items"], [])
        self.assertEqual(data["total"], 0)
        self.assertEqual(data["skip"], 0)
        self.assertEqual(data["limit"], 50)

    def test_list_runs_populated(self) -> None:
        self._create_sample_run()
        self._create_sample_run()
        response = self.client.get("/api/v1/runs")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["items"]), 2)
        self.assertEqual(data["total"], 2)
        self.assertEqual(data["items"][0]["status"], "COMPLETED")
        self.assertTrue(data["items"][0]["is_validated"])

    def test_list_runs_pagination(self) -> None:
        for _ in range(5):
            self._create_sample_run()
        response = self.client.get("/api/v1/runs?skip=2&limit=2")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["items"]), 2)
        self.assertEqual(data["total"], 5)
        self.assertEqual(data["skip"], 2)
        self.assertEqual(data["limit"], 2)

    def test_get_run_success(self) -> None:
        sample = self._create_sample_run()
        response = self.client.get(f"/api/v1/runs/{sample.id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["id"], sample.id)
        self.assertEqual(data["original_prompt"], sample.original_prompt)
        self.assertEqual(data["final_prompt"], sample.final_prompt)
        self.assertEqual(len(data["iterations"]), 1)
        self.assertEqual(data["score_history"], [68.0, 88.5])

    def test_get_run_not_found(self) -> None:
        random_uuid = str(uuid.uuid4())
        response = self.client.get(f"/api/v1/runs/{random_uuid}")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"].lower())

    def test_get_run_invalid_uuid(self) -> None:
        response = self.client.get("/api/v1/runs/invalid-uuid-123")
        self.assertEqual(response.status_code, 422)

    def test_delete_run_success(self) -> None:
        sample = self._create_sample_run()
        response = self.client.delete(f"/api/v1/runs/{sample.id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["deleted_id"], sample.id)

        # Verify deletion
        get_resp = self.client.get(f"/api/v1/runs/{sample.id}")
        self.assertEqual(get_resp.status_code, 404)

    def test_delete_run_not_found(self) -> None:
        random_uuid = str(uuid.uuid4())
        response = self.client.delete(f"/api/v1/runs/{random_uuid}")
        self.assertEqual(response.status_code, 404)

    def test_delete_run_invalid_uuid(self) -> None:
        response = self.client.delete("/api/v1/runs/not-a-uuid")
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
