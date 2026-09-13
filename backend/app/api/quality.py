"""FastAPI router for the PromptLens quality prediction endpoint (Step 12)."""
from fastapi import APIRouter, HTTPException, status

from app.schemas.quality import QualityPredictionRequest, QualityPredictionResponse
from app.services.quality_service import QualityModelUnavailableError, predict_quality

router = APIRouter(prefix="/api/v1/quality", tags=["quality"])


@router.post("/predict", response_model=QualityPredictionResponse)
def predict(request: QualityPredictionRequest) -> QualityPredictionResponse:
    """Predict the quality score (0-100) and quality label for a prompt.

    The score is a baseline ML estimate. It does not represent ground-truth
    prompt quality. See the model disclaimer in the backend README.
    """
    try:
        return predict_quality(request.prompt)
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
