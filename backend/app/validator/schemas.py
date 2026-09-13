"""Pydantic schemas for PromptLens AI Validator (Step 18).

Provides strict, validated models for the Validator request, response, and results.
Enforces extra="forbid" and avoids Any across all models.
"""
from __future__ import annotations

from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ValidatorDecision(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class ValidationMode(StrEnum):
    LOCAL_LLM = "LOCAL_LLM"
    MOCK = "MOCK"
    UNAVAILABLE = "UNAVAILABLE"


class IssueSeverity(StrEnum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class EvidenceSource(StrEnum):
    DETERMINISTIC = "DETERMINISTIC"
    NLP = "NLP"
    EMBEDDING = "EMBEDDING"
    SCORING = "SCORING"
    CRITIC = "CRITIC"
    LLM = "LLM"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------
class PromptMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_length: int = Field(ge=0, description="Character count of prompt.")
    word_count: int = Field(ge=0, description="Word count of prompt.")
    sentence_count: int = Field(ge=0, description="Sentence count of prompt.")
    detected_intent: str = Field(description="Primary classified intent label.")
    intent_confidence: float = Field(ge=0.0, le=1.0, description="Intent classification confidence.")
    output_formats: list[str] = Field(default_factory=list, description="Detected output formats.")
    quality_score: float = Field(ge=0.0, le=100.0, description="Step 13 overall quality score.")


class CriticResultInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: str = Field(description="Step 17 Critic decision: PASS, FAIL, or NEEDS_REVIEW.")
    overall_critique_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Overall critique score from Step 17 Critic.",
    )
    issues: list[dict[str, str]] = Field(
        default_factory=list,
        description="Issues reported by the Critic.",
    )
    lost_requirements: list[str] = Field(
        default_factory=list,
        description="Requirements reported lost by Critic.",
    )
    introduced_requirements: list[str] = Field(
        default_factory=list,
        description="Requirements reported introduced by Critic.",
    )
    unsupported_assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions reported unsupported by Critic.",
    )


class OptimizationMetadataInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optimization_mode: str | None = None
    changes: list[dict[str, str]] = Field(default_factory=list)
    preserved_requirements: list[dict[str, str]] = Field(default_factory=list)
    placeholders_inserted: list[str] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="Unique code identifying the validation issue type.")
    severity: IssueSeverity = Field(description="Severity tier: ERROR, WARNING, or INFO.")
    message: str = Field(description="Human-readable explanation of why this check failed.")
    field_or_scope: str = Field(description="Scope or parameter where the violation occurred.")
    evidence: str = Field(description="Specific textual or metric evidence supporting the issue.")


class ValidationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: EvidenceSource = Field(description="Originating subsystem or analysis pipeline.")
    description: str = Field(description="Human-readable evidence statement.")
    metric_name: str | None = Field(default=None, description="Name of the evaluated metric, if applicable.")
    metric_value: float | None = Field(default=None, description="Numeric value of the metric, if applicable.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this evidence item in [0, 1].")


class ValidationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validator_version: str = Field(description="Semantic version of the Validator component.")
    validation_mode: ValidationMode = Field(description="Operational mode used: LOCAL_LLM, MOCK, or UNAVAILABLE.")
    model: str = Field(description="Identifier of the model used, or 'none' if unavailable.")
    latency_ms: float = Field(ge=0.0, description="Total evaluation latency in milliseconds.")
    critic_consistency_checked: bool = Field(description="Whether Step 17 Critic findings were integrated.")


# ---------------------------------------------------------------------------
# Core Result & Request/Response
# ---------------------------------------------------------------------------
class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: ValidatorDecision = Field(description="Validation decision: PASS, FAIL, or NEEDS_REVIEW.")
    is_valid: bool = Field(description="True if decision is PASS; False otherwise.")
    safety_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Composite validation and safety score in [0, 100].",
    )
    original_metadata: PromptMetadata
    optimized_metadata: PromptMetadata
    issues: list[ValidationIssue] = Field(default_factory=list)
    evidence: list[ValidationEvidence] = Field(default_factory=list)
    passed_checks: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    metadata: ValidationMetadata


class ValidatorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_prompt: StrictStr = Field(description="The original user prompt.")
    optimized_prompt: StrictStr = Field(description="The proposed optimized prompt.")
    critic_result: CriticResultInput | None = Field(
        default=None,
        description="Optional Step 17 Critic result for consistency checking.",
    )
    optimization_metadata: OptimizationMetadataInput | None = Field(
        default=None,
        description="Optional Step 16 Optimizer metadata.",
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


class ValidatorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: ValidationResult
    original_prompt_length: int = Field(ge=0, description="Character length of original prompt.")
    optimized_prompt_length: int = Field(ge=0, description="Character length of optimized prompt.")
