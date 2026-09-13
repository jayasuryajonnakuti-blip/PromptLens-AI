"""Unit tests for AgentService singleton lifecycle and test isolation (Step 19)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from app.agent.schemas import AgentRequest, AgentResponse, AgentStatus, AgentTerminationReason, AgentMetrics
from app.agent.orchestrator import AgentOrchestrator
from app.services.agent_service import AgentService, get_agent_service, set_agent_service


class TestAgentService(unittest.TestCase):
    """Test suite for AgentService singleton and delegation."""

    def setUp(self) -> None:
        set_agent_service(None)

    def tearDown(self) -> None:
        set_agent_service(None)

    def test_singleton_retrieval(self) -> None:
        svc1 = get_agent_service()
        svc2 = get_agent_service()
        self.assertIs(svc1, svc2)
        self.assertIsInstance(svc1, AgentService)

    def test_singleton_reset(self) -> None:
        svc1 = get_agent_service()
        set_agent_service(None)
        svc2 = get_agent_service()
        self.assertIsNot(svc1, svc2)

    def test_singleton_replacement(self) -> None:
        mock_orch = MagicMock(spec=AgentOrchestrator)
        custom_svc = AgentService(orchestrator=mock_orch)
        set_agent_service(custom_svc)
        self.assertIs(get_agent_service(), custom_svc)

    def test_delegation_to_orchestrator(self) -> None:
        mock_orch = MagicMock(spec=AgentOrchestrator)
        expected_resp = AgentResponse(
            original_prompt="test",
            final_prompt="test opt",
            status=AgentStatus.COMPLETED,
            termination_reason=AgentTerminationReason.VALIDATED,
            iteration_count=1,
            is_validated=True,
            iterations=[],
            metrics=AgentMetrics(),
        )
        mock_orch.run.return_value = expected_resp
        svc = AgentService(orchestrator=mock_orch)

        req = AgentRequest(prompt="test prompt")
        res = svc.run(req)
        self.assertEqual(res, expected_resp)
        mock_orch.run.assert_called_once_with(req)


if __name__ == "__main__":
    unittest.main()
