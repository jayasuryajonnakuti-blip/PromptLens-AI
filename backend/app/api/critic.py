"""FastAPI router for PromptLens AI Critic endpoint (Step 17).

Exposes:
    POST /api/v1/critic/evaluate
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.critic.schemas import CriticRequest, CriticResponse
from app.services.critic_service import get_critic_service

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/critic", tags=["critic"])


@router.post(
    "/evaluate",
    response_model=CriticResponse,
    summary="Evaluate whether an optimized prompt is genuinely better than the original",
    description=(
        "Runs the PromptLens AI Critic pipeline to evaluate an optimized prompt against the original. "
        "Evaluates intent preservation, requirement preservation, semantic similarity, quality deltas, "
        "unsupported introduced requirements, contradictions, and verbosity. "
        "Returns a PASS, FAIL, or NEEDS_REVIEW verdict with detailed multidimensional evidence. "
        "Does not rewrite or optimize the prompt."
    ),
)
def evaluate_optimization(request: CriticRequest) -> CriticResponse:
    """Evaluate an optimized prompt against the original prompt."""
    try:
        service = get_critic_service()
        evaluation = service.evaluate(
            original_prompt=request.original_prompt,
            optimized_prompt=request.optimized_prompt,
            optimization_metadata=request.optimization_metadata,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        LOGGER.exception("Unexpected error in critic endpoint: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Critic evaluation error: {exc}",
        ) from exc

    return CriticResponse(
        evaluation=evaluation,
        original_prompt_length=len(request.original_prompt),
        optimized_prompt_length=len(request.optimized_prompt),
    )
