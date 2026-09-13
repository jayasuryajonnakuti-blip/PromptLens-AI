"""FastAPI router for PromptLens AI Analyzer endpoint (Step 15).

Exposes:
    POST /api/v1/analyzer/analyze
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.analyzer.schemas import AnalyzerRequest, AnalyzerResponse
from app.services.analyzer_service import get_analyzer_service

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analyzer", tags=["analyzer"])


@router.post(
    "/analyze",
    response_model=AnalyzerResponse,
    summary="Analyze a prompt with evidence-grounded AI analysis",
    description=(
        "Runs the full PromptLens AI Analyzer pipeline: assembles deterministic evidence from "
        "Steps 8–13 (preprocessing, NLP, intent, quality ML, scoring), then optionally invokes "
        "the local LLM (Step 14) for richer structured analysis. "
        "Returns AnalyzerResponse with mode label (LOCAL_LLM | MOCK | UNAVAILABLE). "
        "The Step 13 overall quality score is preserved and exposed as context — it is NOT "
        "replaced by the LLM instruction_quality score."
    ),
)
def analyze_prompt(request: AnalyzerRequest) -> AnalyzerResponse:
    """Analyze a prompt and return structured, evidence-grounded AI analysis."""
    prompt = request.prompt

    try:
        service = get_analyzer_service()
        analysis = service.analyze(prompt)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        LOGGER.exception("Unexpected error in analyzer endpoint: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analyzer error: {exc}",
        ) from exc

    # Step 13 score: read from evidence or re-derive from context if available.
    # We surface it directly in the response envelope as a first-class field so
    # the caller never has to scan the evidence list for it.
    step13_score = 0.0
    step13_category = "UNAVAILABLE"
    try:
        from app.services.scoring_service import score_prompt

        score_resp = score_prompt(prompt)
        step13_score = float(score_resp.overall_score.score)
        step13_category = score_resp.overall_score.category
    except Exception:
        # Graceful fallback — scoring is best-effort in the API response envelope
        pass

    return AnalyzerResponse(
        analysis=analysis,
        prompt_length=len(prompt),
        step13_overall_score=step13_score,
        step13_quality_category=step13_category,
    )
