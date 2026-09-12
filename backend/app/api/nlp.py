from fastapi import APIRouter, HTTPException, status

from app.engines.nlp import NlpModelUnavailableError
from app.schemas.nlp import NlpAnalysisResult, NlpAnalyzeRequest
from app.services.nlp_service import analyze_prompt_nlp

router = APIRouter(prefix="/api/v1/nlp", tags=["nlp"])


@router.post("/analyze", response_model=NlpAnalysisResult)
def analyze(request: NlpAnalyzeRequest) -> NlpAnalysisResult:
    try:
        return analyze_prompt_nlp(request.prompt)
    except NlpModelUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error