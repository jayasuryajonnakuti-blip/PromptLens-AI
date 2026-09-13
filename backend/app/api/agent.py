"""FastAPI router for PromptLens AI Agent Loop (Step 19).

Exposes:
- POST /api/v1/agent/run: Iteratively analyze, optimize, critique, and validate a prompt.
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status

from app.agent.errors import AgentServiceUnavailableError
from app.agent.schemas import AgentRequest, AgentResponse, AgentStatus
from app.services.agent_service import get_agent_service

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agent", tags=["Agent"])


@router.post(
    "/run",
    response_model=AgentResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute the iterative PromptLens Agent Loop",
    description=(
        "Coordinates Analyzer -> Optimizer -> Critic -> Validator in a bounded iterative "
        "loop (max 3 iterations). Immediately stops on Validator PASS or NEEDS_REVIEW. "
        "Guarantees that an unvalidated prompt is never falsely reported as successful."
    ),
)
def run_agent_loop(request: AgentRequest) -> AgentResponse:
    """Execute the bounded Agent Loop on the provided user prompt."""
    try:
        service = get_agent_service()
        response = service.run(request)

        if response.status == AgentStatus.UNAVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent Loop is currently unavailable due to upstream service limitations.",
            )

        return response

    except AgentServiceUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Agent service unavailable: {exc}",
        )
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.error("Unhandled error in POST /api/v1/agent/run: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing the Agent Loop: {str(exc)}",
        )
