"""AgentRunService: Persistence service for Agent Loop executions (Step 20).

Bridges the Agent Loop (Step 19) and the database persistence layer:
- Converts AgentResponse and AgentIteration models into JSON-safe persistence structures.
- Enforces session management and transaction boundary isolation.
- Provides process-level singleton with test-isolation reset support.
"""
from __future__ import annotations

import logging
from typing import Any
from sqlalchemy.orm import Session, sessionmaker

from app.agent.schemas import AgentResponse
from app.db.models import AgentRun
from app.db.repositories import AgentRunRepository
from app.db.session import get_session_factory

LOGGER = logging.getLogger(__name__)


class AgentRunService:
    """Service managing persistence and retrieval of AgentRun records."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None) -> None:
        self._session_factory = session_factory or get_session_factory()

    def save_agent_run(
        self,
        response: AgentResponse,
        mode: str = "balanced",
        run_id: str | None = None,
    ) -> AgentRun:
        """Serialize and persist a completed AgentResponse.

        Args:
            response: The completed AgentResponse from the Agent Loop.
            mode: Optimization mode applied (balanced, analytical, creative, expert).
            run_id: Optional predetermined UUID identifier.

        Returns:
            The persisted AgentRun model.
        """
        # Convert structured Pydantic models into JSON-safe dictionaries
        serialized_iterations = [
            it.model_dump(mode="json") for it in response.iterations
        ]

        # Extract score history from iterations
        score_history: list[float] = []
        for it in response.iterations:
            if it.score_before is not None and not score_history:
                score_history.append(it.score_before)
            if it.score_after is not None:
                score_history.append(it.score_after)

        serialized_critic = (
            response.final_critic_result.model_dump(mode="json")
            if response.final_critic_result
            else None
        )
        serialized_val = (
            response.final_validation.model_dump(mode="json")
            if response.final_validation
            else None
        )
        serialized_metrics = (
            response.metrics.model_dump(mode="json")
            if response.metrics
            else None
        )
        run_metadata: dict[str, Any] = {
            "disclaimers": list(response.disclaimers),
        }

        # Build AgentRun ORM entity
        agent_run = AgentRun(
            original_prompt=response.original_prompt,
            final_prompt=response.final_prompt,
            status=response.status.value,
            termination_reason=response.termination_reason.value,
            is_validated=response.is_validated,
            iteration_count=response.iteration_count,
            final_score=response.final_score,
            agent_version="1.0.0",
            mode=mode,
            iterations=serialized_iterations,
            score_history=score_history,
            critic_result=serialized_critic,
            final_validation=serialized_val,
            metrics=serialized_metrics,
            run_metadata=run_metadata,
        )
        if run_id:
            agent_run.id = run_id

        # Persist within a dedicated short-lived session
        session: Session = self._session_factory()
        try:
            repo = AgentRunRepository(session)
            persisted = repo.create(agent_run)
            LOGGER.info("Successfully persisted AgentRun id=%s, status=%s", persisted.id, persisted.status)
            return persisted
        finally:
            session.close()

    def get_agent_run(self, run_id: str) -> AgentRun | None:
        """Retrieve an AgentRun by ID."""
        session: Session = self._session_factory()
        try:
            repo = AgentRunRepository(session)
            return repo.get(run_id)
        finally:
            session.close()

    def list_agent_runs(self, skip: int = 0, limit: int = 50) -> list[AgentRun]:
        """List persisted AgentRuns ordered by created_at DESC."""
        session: Session = self._session_factory()
        try:
            repo = AgentRunRepository(session)
            return repo.list(skip=skip, limit=limit)
        finally:
            session.close()

    def delete_agent_run(self, run_id: str) -> bool:
        """Delete an AgentRun record by ID."""
        session: Session = self._session_factory()
        try:
            repo = AgentRunRepository(session)
            return repo.delete(run_id)
        finally:
            session.close()

    def count_agent_runs(self) -> int:
        """Count total persisted AgentRun records."""
        session: Session = self._session_factory()
        try:
            repo = AgentRunRepository(session)
            return repo.count()
        finally:
            session.close()


_SERVICE_INSTANCE: AgentRunService | None = None


def get_agent_run_service() -> AgentRunService:
    """Return the process-level singleton AgentRunService."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = AgentRunService()
    return _SERVICE_INSTANCE


def set_agent_run_service(service: AgentRunService | None) -> None:
    """Set or reset the global AgentRunService instance (for test isolation)."""
    global _SERVICE_INSTANCE
    _SERVICE_INSTANCE = service
