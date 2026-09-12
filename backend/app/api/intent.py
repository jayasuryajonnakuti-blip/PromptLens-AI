from fastapi import APIRouter, HTTPException, status

from app.ml.intent.inference import IntentModelUnavailableError
from app.schemas.intent import IntentClassificationRequest, IntentClassificationResponse
from app.services.intent_service import classify_prompt

router = APIRouter(prefix="/api/v1/intent", tags=["intent"])


@router.post("/classify", response_model=IntentClassificationResponse)
def classify(request: IntentClassificationRequest) -> IntentClassificationResponse:
    try:
        return classify_prompt(request.prompt)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    except IntentModelUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error