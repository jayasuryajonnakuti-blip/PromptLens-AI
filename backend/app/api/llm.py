"""FastAPI router for the PromptLens Local LLM Layer (Step 14)."""
from fastapi import APIRouter, HTTPException, status

from app.llm.errors import (
    ContextOverflowError,
    InvalidGenerationError,
    InvalidJsonError,
    LLMDisabledError,
    ModelLoadingError,
    ModelNotDownloadedError,
    ModelRuntimeError,
    RetryExhaustedError,
    SchemaValidationError,
)
from app.llm.schemas import LLMAnalysisRequest, LLMAnalysisResponse
from app.services.llm_service import get_llm_service

router = APIRouter(prefix="/api/v1/llm", tags=["llm"])


@router.post("/analyze", response_model=LLMAnalysisResponse)
def analyze(request: LLMAnalysisRequest) -> LLMAnalysisResponse:
    """Analyze prompt engineering structure and characteristics using the local LLM.

    Returns strict, validated JSON with goal interpretation, ambiguity analysis,
    missing constraints, strengths, weaknesses, and recommendations.
    """
    service = get_llm_service()
    try:
        return service.analyze_prompt(request.prompt)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except LLMDisabledError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e
    except ModelNotDownloadedError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e
    except (ModelLoadingError, ModelRuntimeError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Local model runtime error: {e}",
        ) from e
    except (RetryExhaustedError, SchemaValidationError, InvalidJsonError, InvalidGenerationError) as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Local LLM structured output contract violation: {e}",
        ) from e
    except ContextOverflowError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal LLM analysis failure: {e}",
        ) from e


@router.get("/health")
def llm_health() -> dict[str, str]:
    """Check the health and operational availability of the local LLM."""
    service = get_llm_service()
    state = service.health()
    meta = service.provider.model_metadata()
    return {
        "status": state.value,
        "model": meta.get("name", service.config.model),
        "provider": meta.get("provider", service.config.provider),
    }
