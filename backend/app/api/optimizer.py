"""FastAPI router for PromptLens AI Optimizer endpoint (Step 16).

Exposes:
    POST /api/v1/optimizer/optimize
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.optimizer.schemas import (
    OptimizationMode,
    OptimizerRequest,
    OptimizerResponse,
)
from app.services.optimizer_service import get_optimizer_service

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/optimizer", tags=["optimizer"])


@router.post(
    "/optimize",
    response_model=OptimizerResponse,
    summary="Generate an optimized version of a prompt",
    description=(
        "Runs the PromptLens AI Optimizer pipeline. "
        "Accepts an original prompt, an optional Step 15 AnalyzerAnalysis result, "
        "and an optimization mode (balanced | analytical | creative | expert). "
        "Returns OptimizerResponse with the optimized prompt, a structured change log, "
        "preserved requirements, and accuracy metadata (LOCAL_LLM | MOCK | UNAVAILABLE). "
        "When the LLM is unavailable the original prompt is returned unchanged — "
        "no fabricated optimization is ever returned silently."
    ),
)
def optimize_prompt(request: OptimizerRequest) -> OptimizerResponse:
    """Optimize a prompt and return structured improvement results."""
    try:
        service = get_optimizer_service()
        result = service.optimize(
            prompt=request.prompt,
            mode=request.mode,
            analyzer_result=request.analyzer_result,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        LOGGER.exception("Unexpected error in optimizer endpoint: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimizer error: {exc}",
        ) from exc

    return OptimizerResponse(
        result=result,
        original_prompt_length=len(request.prompt),
        optimized_prompt_length=len(result.optimized_prompt),
        optimization_mode=request.mode,
    )
