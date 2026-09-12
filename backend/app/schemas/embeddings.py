from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


class EmbeddingModelMetadata(BaseModel):
    name: str
    version: str
    dimension: int
    normalized: bool


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: StrictStr = Field(min_length=1)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty or whitespace-only")
        return value


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    dimensions: int
    model: EmbeddingModelMetadata
    normalized: bool


class SimilarityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_a: StrictStr = Field(min_length=1)
    prompt_b: StrictStr = Field(min_length=1)

    @field_validator("prompt_a", "prompt_b")
    @classmethod
    def prompts_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompts must not be empty or whitespace-only")
        return value


class SimilarityResponse(BaseModel):
    similarity: float = Field(ge=-1.0, le=1.0)
    metric: str = "cosine"
    model: EmbeddingModelMetadata