from fastapi import APIRouter

from app.schemas.preprocessing import PromptPreprocessRequest, PromptPreprocessingResult
from app.services.preprocessing_service import preprocess_prompt

router = APIRouter(prefix="/api/v1", tags=["preprocessing"])


@router.post("/preprocess", response_model=PromptPreprocessingResult)
def preprocess(request: PromptPreprocessRequest) -> PromptPreprocessingResult:
    return preprocess_prompt(request.prompt)
