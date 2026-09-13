"""FastAPI router for PromptLens AI Agent Run History (Step 21).

Exposes:
- GET /api/v1/runs: List persisted AgentRuns with pagination.
- GET /api/v1/runs/{run_id}: Retrieve full details of a specific AgentRun.
- DELETE /api/v1/runs/{run_id}: Delete an AgentRun record.
"""
from __future__ import annotations

import logging
import uuid
from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.runs import (
    AgentRunDetail,
    AgentRunListResponse,
    AgentRunSummary,
    DeleteRunResponse,
)
from app.services.agent_run_service import get_agent_run_service

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/runs", tags=["Runs"])


def _validate_uuid(run_id: str) -> None:
    """Validate that run_id is a valid UUID string."""
    try:
        uuid.UUID(run_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid UUID format: '{run_id}'",
        )


@router.get(
    "",
    response_model=AgentRunListResponse,
    status_code=status.HTTP_200_OK,
    summary="List persisted AgentRuns",
    description="Retrieve paginated list of persisted agent runs ordered by created_at DESC.",
)
def list_runs(
    skip: int = Query(0, ge=0, description="Number of records to skip."),
    limit: int = Query(50, ge=1, le=100, description="Maximum records to return."),
) -> AgentRunListResponse:
    try:
        service = get_agent_run_service()
        runs = service.list_agent_runs(skip=skip, limit=limit)
        total = service.count_agent_runs()
        items = [AgentRunSummary.model_validate(run) for run in runs]
        return AgentRunListResponse(items=items, total=total, skip=skip, limit=limit)
    except Exception as exc:
        LOGGER.error("Failed to list agent runs: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list agent runs: {exc}",
        )


@router.get(
    "/{run_id}",
    response_model=AgentRunDetail,
    status_code=status.HTTP_200_OK,
    summary="Get AgentRun details",
    description="Retrieve full execution history, metrics, and structured artifacts of a specific run.",
)
def get_run(run_id: str) -> AgentRunDetail:
    _validate_uuid(run_id)
    try:
        service = get_agent_run_service()
        run = service.get_agent_run(run_id)
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent run with id '{run_id}' not found.",
            )
        return AgentRunDetail.model_validate(run)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.error("Failed to retrieve agent run '%s': %s", run_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve agent run: {exc}",
        )


@router.delete(
    "/{run_id}",
    response_model=DeleteRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete an AgentRun record",
    description="Delete a persisted agent run from the database.",
)
def delete_run(run_id: str) -> DeleteRunResponse:
    _validate_uuid(run_id)
    try:
        service = get_agent_run_service()
        deleted = service.delete_agent_run(run_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent run with id '{run_id}' not found.",
            )
        return DeleteRunResponse(
            success=True,
            message=f"Agent run '{run_id}' deleted successfully.",
            deleted_id=run_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.error("Failed to delete agent run '%s': %s", run_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete agent run: {exc}",
        )
