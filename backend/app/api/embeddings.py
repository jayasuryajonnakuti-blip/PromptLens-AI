from fastapi import APIRouter, HTTPException, status

from app.engines.embeddings.model import EmbeddingModelUnavailableError
from app.schemas.embeddings import (
    EmbeddingRequest,
    EmbeddingResponse,
    SimilarityRequest,
    SimilarityResponse,
)
from app.services.embedding_service import get_embedding_service

router = APIRouter(prefix="/api/v1/embeddings", tags=["embeddings"])


@router.post("/generate", response_model=EmbeddingResponse)
def generate_embedding(request: EmbeddingRequest) -> EmbeddingResponse:
    try:
        service = get_embedding_service()
        embedding = service.encode(request.prompt)
        return EmbeddingResponse(
            embedding=embedding.tolist(),
            dimensions=service.metadata.dimension,
            model=service.metadata,
            normalized=service.metadata.normalized,
        )
    except EmbeddingModelUnavailableError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error


@router.post("/similarity", response_model=SimilarityResponse)
def calculate_similarity(request: SimilarityRequest) -> SimilarityResponse:
    try:
        service = get_embedding_service()
        return SimilarityResponse(
            similarity=service.similarity(request.prompt_a, request.prompt_b),
            model=service.metadata,
        )
    except EmbeddingModelUnavailableError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error