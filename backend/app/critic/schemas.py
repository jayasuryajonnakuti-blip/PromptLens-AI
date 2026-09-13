"""Pydantic schemas for PromptLens AI Critic (Step 17).

Provides strict, validated models for the Critic evaluation request and response.
Adheres strictly to the requirement: extra="forbid" and no `Any` types.
"""
from __future__ import annotations

from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class CriticDecision(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class IssueSeverity(StrEnum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class QualityStatus(StrEnum):
    EXCELLENT = "EXCELLENT"
    STRONG = "STRONG"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"


class CriticAnalysisMode(StrEnum):
    LOCAL_LLM = "LOCAL_LLM"
    MOCK = "MOCK"
    UNAVAILABLE = "UNAVAILABLE"


class EvidenceSource(StrEnum):
    ORIGINAL = "original"
    OPTIMIZED = "optimized"
    PREPROCESSING = "preprocessing"
    NLP = "nlp"
    INTENT = "intent"
    EMBEDDING = "embedding"
    SCORING = "scoring"
    ANALYZER = "analyzer"
    OPTIMIZER = "optimizer"
    LLM = "llm"


class EvidenceType(StrEnum):
    OBSERVATION = "observation"
    COMPARISON = "comparison"
    INFERENCE = "inference"


# ---------------------------------------------------------------------------
# Sub-models for evaluation components
# ---------------------------------------------------------------------------
class IntentPreservationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(
        ge=0.0,
        le=100.0,
        description="Intent preservation score from 0 to 100.",
    )
    status: QualityStatus = Field(description="Categorical evaluation of intent preservation.")
    reason: str = Field(description="Explanation of intent preservation evaluation.")


class RequirementPreservationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(
        ge=0.0,
        le=100.0,
        description="Requirement preservation score from 0 to 100.",
    )
    status: QualityStatus = Field(description="Categorical evaluation of requirement preservation.")
    reason: str = Field(description="Explanation of requirement preservation evaluation.")


class MetricChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_score: float = Field(ge=0.0, le=100.0, description="Score on original prompt.")
    optimized_score: float = Field(ge=0.0, le=100.0, description="Score on optimized prompt.")
    delta: float = Field(ge=-100.0, le=100.0, description="Score change (optimized - original).")
    assessment: str = Field(description="Qualitative assessment of the change.")


class CriticIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(description="Issue category or identifier.")
    severity: IssueSeverity = Field(description="Severity tier: ERROR, WARNING, or INFO.")
    description: str = Field(description="Clear explanation of the identified issue.")
    evidence: str = Field(description="Quotation or concrete reference from prompts or pipeline signals.")


class CriticEvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: EvidenceSource = Field(description="Originating signal source.")
    type: EvidenceType = Field(description="Type of evidence: observation, comparison, or inference.")
    statement: str = Field(description="Specific finding or observation.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence estimate in [0, 1].")


class CriticMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    critic_version: str = Field(description="Semantic version of the Critic component.")
    analysis_mode: CriticAnalysisMode = Field(
        description="Accurately reflects whether real local LLM, mock, or unavailable was used."
    )
    model: str = Field(description="Name of the model used for LLM evaluation, or 'none'.")
    semantic_similarity: float = Field(
        ge=0.0,
        le=1.0,
        description="Cosine similarity between original and optimized embeddings.",
    )
    latency_ms: float = Field(
        default=0.0,
        description="Total wall-clock evaluation latency in milliseconds.",
    )


# ---------------------------------------------------------------------------
# Core evaluation payload
# ---------------------------------------------------------------------------
class CriticEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: CriticDecision = Field(
        description="Final verdict on whether to accept the optimization: PASS, FAIL, or NEEDS_REVIEW."
    )
    overall_critique_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Overall critique score representing optimization quality.",
    )
    intent_preservation: IntentPreservationResult
    requirement_preservation: RequirementPreservationResult
    clarity_change: MetricChange
    specificity_change: MetricChange
    ambiguity_change: MetricChange
    completeness_change: MetricChange
    issues: list[CriticIssue] = Field(default_factory=list)
    preserved_requirements: list[str] = Field(default_factory=list)
    lost_requirements: list[str] = Field(default_factory=list)
    introduced_requirements: list[str] = Field(default_factory=list)
    unsupported_assumptions: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    evidence: list[CriticEvidenceItem] = Field(default_factory=list)
    metadata: CriticMetadata


# ---------------------------------------------------------------------------
# Request & Response models
# ---------------------------------------------------------------------------
class OptimizationMetadataInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optimization_mode: str | None = None
    changes: list[dict[str, str]] = Field(default_factory=list)
    preserved_requirements: list[dict[str, str]] = Field(default_factory=list)
    placeholders_inserted: list[str] = Field(default_factory=list)


class CriticRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_prompt: StrictStr = Field(description="The initial user prompt.")
    optimized_prompt: StrictStr = Field(description="The proposed optimized prompt.")
    optimization_metadata: OptimizationMetadataInput | None = Field(
        default=None,
        description="Optional metadata from Step 16 optimizer.",
    )

    @field_validator("original_prompt")
    @classmethod
    def original_prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("original_prompt must not be empty or whitespace-only")
        return value

    @field_validator("optimized_prompt")
    @classmethod
    def optimized_prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("optimized_prompt must not be empty or whitespace-only")
        return value


class CriticResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation: CriticEvaluation
    original_prompt_length: int = Field(description="Character count of original prompt.")
    optimized_prompt_length: int = Field(description="Character count of optimized prompt.")
