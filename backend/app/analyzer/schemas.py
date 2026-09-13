"""Pydantic schemas for PromptLens AI Analyzer (Step 15).

These schemas are DISTINCT from Step 14's LLM schemas.  They model the final,
enriched analysis output that combines deterministic pipeline evidence with
optional LLM-assisted reasoning.

Design notes:
- confidence values are 0.0–1.0 model-assisted estimates, NOT calibrated
  probabilities.  Callers should document them accordingly.
- analysis_mode is ALWAYS set accurately; the analyzer never silently falls
  back to mock output in production.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


# ---------------------------------------------------------------------------
# Analysis mode enum
# ---------------------------------------------------------------------------
class AnalysisMode(StrEnum):
    LOCAL_LLM = "LOCAL_LLM"
    MOCK = "MOCK"
    UNAVAILABLE = "UNAVAILABLE"


class RecommendationPriority(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class EvidenceSource(StrEnum):
    PREPROCESSING = "preprocessing"
    NLP = "nlp"
    INTENT = "intent"
    QUALITY_ML = "quality_ml"
    SCORING = "scoring"
    LLM = "llm"


class EvidenceType(StrEnum):
    OBSERVATION = "observation"
    INFERENCE = "inference"


# ---------------------------------------------------------------------------
# Evidence items
# ---------------------------------------------------------------------------
class EvidenceItem(BaseModel):
    """A single piece of evidence from a named upstream pipeline stage."""

    model_config = ConfigDict(extra="forbid")

    source: EvidenceSource = Field(description="Pipeline stage that produced this evidence.")
    type: EvidenceType = Field(description="Whether this is a direct observation or an inference.")
    statement: str = Field(description="Human-readable description of the finding.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Model-assisted confidence estimate in [0, 1]. "
            "Not a calibrated probability — treat as an ordinal indicator."
        ),
    )


# ---------------------------------------------------------------------------
# Structured analysis sub-items (richer than Step 14's plain strings)
# ---------------------------------------------------------------------------
class AmbiguityItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue: str = Field(description="Description of the ambiguity.")
    evidence: str = Field(description="Direct quotation or reference from the prompt text.")
    confidence: float = Field(ge=0.0, le=1.0)


class MissingInfoItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: str = Field(description="What is missing from the prompt.")
    why_it_matters: str = Field(description="How the absence affects prompt effectiveness.")
    confidence: float = Field(ge=0.0, le=1.0)


class ContradictionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue: str = Field(description="Description of the contradiction.")
    evidence: str = Field(description="Direct quotation or reference from the prompt text.")
    confidence: float = Field(ge=0.0, le=1.0)


class StrengthItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strength: str = Field(description="Identified strength in prompt construction.")
    evidence: str = Field(description="Supporting evidence from the prompt or upstream signals.")


class WeaknessItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weakness: str = Field(description="Identified weakness or deficit.")
    evidence: str = Field(description="Supporting evidence from the prompt or upstream signals.")


class RecommendationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendation: str = Field(description="Actionable improvement recommendation.")
    reason: str = Field(description="Why this recommendation matters for this prompt.")
    priority: RecommendationPriority = Field(description="Priority tier: HIGH, MEDIUM, or LOW.")


class InstructionQualityResult(BaseModel):
    """LLM-assessed instruction quality.  Distinct from Step 13's overall score."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(
        ge=0.0,
        le=100.0,
        description="Instruction quality score 0–100 assessed by the analyzer.",
    )
    reason: str = Field(description="Justification for the assigned score.")


class IntentResult(BaseModel):
    """Resolved intent label with confidence."""

    model_config = ConfigDict(extra="forbid")

    label: str
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Analyzer metadata
# ---------------------------------------------------------------------------
class AnalyzerMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analyzer_version: str = Field(description="Semantic version of the analyzer component.")
    llm_model: str = Field(
        description="Model identifier used for inference, or 'none' if LLM unavailable."
    )
    analysis_mode: AnalysisMode = Field(
        description="Accurately reflects which analysis path was taken."
    )
    latency_ms: float = Field(
        default=0.0,
        description="Total wall-clock latency of the analysis in milliseconds.",
    )


# ---------------------------------------------------------------------------
# Top-level analysis result
# ---------------------------------------------------------------------------
class AnalyzerAnalysis(BaseModel):
    """Complete AI-assisted prompt analysis result."""

    model_config = ConfigDict(extra="forbid")

    interpreted_goal: str = Field(
        description="The underlying objective the user wants to accomplish."
    )
    intent: IntentResult = Field(description="Resolved primary intent with confidence.")
    context_summary: str = Field(
        description="Background context, scenario assumptions, or domain framing."
    )
    ambiguities: list[AmbiguityItem] = Field(
        default_factory=list,
        description="Ambiguous phrases, vague requirements, or multiple interpretations.",
    )
    missing_information: list[MissingInfoItem] = Field(
        default_factory=list,
        description="Unstated inputs, missing parameter bounds, or omitted constraints.",
    )
    contradictions: list[ContradictionItem] = Field(
        default_factory=list,
        description="Conflicting instructions or incompatible output formats.",
    )
    instruction_quality: InstructionQualityResult = Field(
        description=(
            "LLM-assessed instruction quality. "
            "Does NOT replace the Step 13 overall quality score."
        )
    )
    strengths: list[StrengthItem] = Field(default_factory=list)
    weaknesses: list[WeaknessItem] = Field(default_factory=list)
    recommendations: list[RecommendationItem] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description=(
            "Ordered list of evidence items used to produce this analysis. "
            "Priority: explicit prompt text > preprocessing > NLP > intent > quality_ml > scoring > LLM."
        ),
    )
    metadata: AnalyzerMetadata


# ---------------------------------------------------------------------------
# API request / response wrappers
# ---------------------------------------------------------------------------
class AnalyzerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: StrictStr = Field(description="The prompt to analyze.")

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty or whitespace-only")
        return value


class AnalyzerResponse(BaseModel):
    """Top-level API response envelope for the analyzer endpoint."""

    model_config = ConfigDict(extra="forbid")

    analysis: AnalyzerAnalysis
    prompt_length: int = Field(description="Character count of the original prompt.")
    step13_overall_score: float = Field(
        description=(
            "Step 13 Scoring & Fusion overall quality score (0–100). "
            "Exposed as context; NOT replaced by the LLM instruction_quality score."
        )
    )
    step13_quality_category: str = Field(
        description="Step 13 quality category label (e.g. GOOD, STRONG)."
    )
