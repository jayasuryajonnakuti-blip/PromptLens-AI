"""AgentService: Process-level singleton wrapping AgentOrchestrator (Step 19).

Follows the same singleton + test-isolation pattern as ValidatorService (Step 18),
CriticService (Step 17), OptimizerService (Step 16), AnalyzerService (Step 15),
and LLMService (Step 14):
- get_agent_service() returns the process-level singleton.
- set_agent_service(None) resets the singleton for test tearDown().
"""
from __future__ import annotations

import logging

from app.agent.orchestrator import AgentOrchestrator
from app.agent.schemas import AgentRequest, AgentResponse

LOGGER = logging.getLogger(__name__)


class AgentService:
    """Thin service wrapper around AgentOrchestrator providing lifecycle management and persistence."""

    def __init__(self, orchestrator: AgentOrchestrator | None = None) -> None:
        self._orchestrator = orchestrator or AgentOrchestrator()

    def run(self, request: AgentRequest) -> AgentResponse:
        """Execute AgentOrchestrator and persist the terminal AgentRun."""
        response = self._orchestrator.run(request)
        try:
            from app.services.agent_run_service import get_agent_run_service
            run_svc = get_agent_run_service()
            persisted_run = run_svc.save_agent_run(response, mode=request.mode.value)
            response.run_id = persisted_run.id
            response.persistence_status = "persisted"
        except Exception as exc:
            LOGGER.error("Failed to persist AgentRun: %s", exc, exc_info=True)
            response.persistence_status = "failed"
        return response


_SERVICE_INSTANCE: AgentService | None = None


def get_agent_service() -> AgentService:
    """Return the process-level singleton AgentService."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = AgentService()
    return _SERVICE_INSTANCE


def set_agent_service(service: AgentService | None) -> None:
    """Set or reset the global AgentService instance (for test isolation)."""
    global _SERVICE_INSTANCE
    _SERVICE_INSTANCE = service
