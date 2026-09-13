"""Pydantic schemas for the PromptLens Scoring & Fusion endpoint (Step 13)."""
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from app.ml.quality.categories import QualityLabel


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: StrictStr = Field(description="The prompt to evaluate and score.")

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty or whitespace-only")
        return value


class OverallScoreResult(BaseModel):
    score: float = Field(ge=0.0, le=100.0, description="Overall fused quality score in [0.0, 100.0].")
    category: QualityLabel = Field(description="PromptLens heuristic quality tier.")
    status: str = Field(description="Quality category label string.")
    explanation: str = Field(description="Summary breakdown of contributing signals.")


class QualityDimensionResult(BaseModel):
    score: float = Field(ge=0.0, le=100.0, description="Dimension score in [0.0, 100.0].")
    status: QualityLabel = Field(description="Dimension quality tier.")
    reason: str = Field(description="Deterministic rationale for the score.")
    recommendation: str = Field(description="Targeted suggestion for improvement.")


class SignalResult(BaseModel):
    source: str = Field(description="Signal identifier (e.g. rules, nlp, quality_ml, llm).")
    raw_score: float | None = Field(default=None, description="Raw component score if available.")
    configured_weight: float = Field(description="Base configured weight in the scoring architecture.")
    effective_weight: float = Field(description="Dynamically normalized weight used in fusion.")
    available: bool = Field(description="Whether this signal was available during scoring.")
    explanation: str | None = Field(default=None, description="Explanation of signal contribution.")


class ScoringFindingResult(BaseModel):
    type: str
    severity: str
    dimension: str
    message: str


class IntentSummary(BaseModel):
    intent: str
    confidence: float
    model: str


class EmbeddingSummary(BaseModel):
    dimension: int
    normalized: bool
    model: str
    available: bool


class ScoringMetadata(BaseModel):
    scoring_version: str
    llm_available: bool
    signals_evaluated: int


class ScoreResponse(BaseModel):
    overall_score: OverallScoreResult
    dimensions: dict[str, QualityDimensionResult]
    signals: dict[str, SignalResult]
    intent: IntentSummary | None = None
    embeddings: EmbeddingSummary | None = None
    findings: list[ScoringFindingResult]
    recommendations: list[str]
    metadata: ScoringMetadata
