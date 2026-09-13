"""Pydantic schemas for the PromptLens quality prediction endpoint (Step 12)."""
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from app.ml.quality.categories import QualityLabel


class QualityPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: StrictStr = Field(description="The raw prompt to assess for quality.")

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty or whitespace-only")
        return value


class QualityModelMetadata(BaseModel):
    name: str
    version: str


class QualityPredictionResponse(BaseModel):
    score: float = Field(
        ge=0.0,
        le=100.0,
        description=(
            "Estimated quality score in [0, 100]. This is a baseline ML estimate, "
            "not ground-truth prompt quality."
        ),
    )
    label: QualityLabel = Field(
        description="Quality category derived from the score using PromptLens heuristic thresholds."
    )
    model: QualityModelMetadata
