"""FastAPI router for the PromptLens scoring and fusion endpoint (Step 13)."""
from fastapi import APIRouter, HTTPException, status

from app.schemas.scoring import ScoreRequest, ScoreResponse
from app.services.quality_service import QualityModelUnavailableError
from app.services.scoring_service import score_prompt

router = APIRouter(prefix="/api/v1", tags=["scoring"])


@router.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    """Evaluate and score a prompt using multi-signal fusion.

    Combines rules (Step 8), NLP (Step 9), and Quality ML (Step 12),
    renormalizing weights dynamically while the LLM signal is unavailable.
    """
    try:
        return score_prompt(request.prompt)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    except QualityModelUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring calculation failed: {error}",
        ) from error
