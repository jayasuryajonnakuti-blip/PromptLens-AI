"""Unit tests for FastAPI Agent Loop router POST /api/v1/agent/run (Step 19)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.agent.schemas import (
    AgentMetrics,
    AgentRequest,
    AgentResponse,
    AgentStatus,
    AgentTerminationReason,
)
from app.main import app
from app.services.agent_service import AgentService, set_agent_service


class TestAgentAPI(unittest.TestCase):
    """Test suite for /api/v1/agent/run endpoint."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        set_agent_service(None)

    def tearDown(self) -> None:
        set_agent_service(None)

    def test_200_ok_valid_request(self) -> None:
        mock_svc = MagicMock(spec=AgentService)
        mock_resp = AgentResponse(
            original_prompt="Write a function",
            final_prompt="def write_function(): ...",
            status=AgentStatus.COMPLETED,
            termination_reason=AgentTerminationReason.VALIDATED,
            iteration_count=1,
            is_validated=True,
            iterations=[],
            metrics=AgentMetrics(
                total_latency_ms=45.0,
                total_iterations=1,
                prompt_length_before=16,
                prompt_length_after=25,
                expansion_ratio=1.56,
            ),
        )
        mock_svc.run.return_value = mock_resp
        set_agent_service(mock_svc)

        payload = {
            "prompt": "Write a function",
            "mode": "balanced",
            "max_iterations": 3,
        }
        res = self.client.post("/api/v1/agent/run", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "COMPLETED")
        self.assertEqual(data["termination_reason"], "VALIDATED")
        self.assertTrue(data["is_validated"])
        self.assertEqual(data["iteration_count"], 1)

    def test_422_blank_prompt(self) -> None:
        payload = {"prompt": "   "}
        res = self.client.post("/api/v1/agent/run", json=payload)
        self.assertEqual(res.status_code, 422)

    def test_422_missing_prompt(self) -> None:
        payload = {"mode": "balanced"}
        res = self.client.post("/api/v1/agent/run", json=payload)
        self.assertEqual(res.status_code, 422)

    def test_422_invalid_max_iterations(self) -> None:
        payload = {"prompt": "Valid prompt", "max_iterations": 5}
        res = self.client.post("/api/v1/agent/run", json=payload)
        self.assertEqual(res.status_code, 422)

        payload_zero = {"prompt": "Valid prompt", "max_iterations": 0}
        res_zero = self.client.post("/api/v1/agent/run", json=payload_zero)
        self.assertEqual(res_zero.status_code, 422)

    def test_422_extra_fields(self) -> None:
        payload = {"prompt": "Valid prompt", "unexpected_field": "disallowed"}
        res = self.client.post("/api/v1/agent/run", json=payload)
        self.assertEqual(res.status_code, 422)

    def test_503_when_agent_unavailable(self) -> None:
        mock_svc = MagicMock(spec=AgentService)
        mock_resp = AgentResponse(
            original_prompt="Write a function",
            final_prompt="Write a function",
            status=AgentStatus.UNAVAILABLE,
            termination_reason=AgentTerminationReason.SERVICE_UNAVAILABLE,
            iteration_count=0,
            is_validated=False,
            iterations=[],
            metrics=AgentMetrics(),
        )
        mock_svc.run.return_value = mock_resp
        set_agent_service(mock_svc)

        payload = {"prompt": "Write a function"}
        res = self.client.post("/api/v1/agent/run", json=payload)
        self.assertEqual(res.status_code, 503)


if __name__ == "__main__":
    unittest.main()
