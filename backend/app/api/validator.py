"""FastAPI router for PromptLens AI Validator endpoint (Step 18).

Exposes:
    POST /api/v1/validator/validate
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.services.validator_service import get_validator_service
from app.validator.schemas import ValidatorRequest, ValidatorResponse

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/validator", tags=["validator"])


@router.post(
    "/validate",
    response_model=ValidatorResponse,
    summary="Validate that an optimized prompt is structurally sound, safe, and acceptable",
    description=(
        "Runs the final validation gatekeeper pipeline on an optimized prompt against the original. "
        "Deterministically checks field integrity, intent preservation, explicit requirements, "
        "negative constraints, output formats, contradictions, bloat, and consistency with Step 17 Critic findings. "
        "Optionally uses the local LLM for semantic verification. "
        "Returns a PASS, FAIL, or NEEDS_REVIEW decision with structured evidence. "
        "Does not rewrite or modify prompts."
    ),
)
def validate_prompt(request: ValidatorRequest) -> ValidatorResponse:
    """Validate an optimized prompt against its original prompt and Critic result."""
    try:
        service = get_validator_service()
        result = service.validate(
            original_prompt=request.original_prompt,
            optimized_prompt=request.optimized_prompt,
            critic_result=request.critic_result,
            optimization_metadata=request.optimization_metadata,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        LOGGER.exception("Unexpected error in validator endpoint: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validator evaluation error: {exc}",
        ) from exc

    return ValidatorResponse(
        result=result,
        original_prompt_length=len(request.original_prompt),
        optimized_prompt_length=len(request.optimized_prompt),
    )
