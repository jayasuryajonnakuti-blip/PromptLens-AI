"""Pydantic schemas for PromptLens AI Optimizer (Step 16).

These schemas are DISTINCT from Step 15's AnalyzerAnalysis schemas.
They model the structured output of the optimization step.

Design notes:
- optimized_prompt is the ONLY field that contains a rewritten prompt.
  All other fields are metadata / audit trail.
- optimization_mode is an explicit enum — never a free-form string.
- optimizer_mode is ALWAYS set accurately (LOCAL_LLM | MOCK | UNAVAILABLE).
- The optimizer NEVER invents requirements not present in the original prompt
  or the analyzer analysis.  Placeholders use the form [TARGET AUDIENCE].
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


# ---------------------------------------------------------------------------
# Optimization mode
# ---------------------------------------------------------------------------
class OptimizationMode(StrEnum):
    BALANCED = "balanced"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    EXPERT = "expert"


# ---------------------------------------------------------------------------
# Optimizer provider mode (mirrors AnalysisMode from Step 15)
# ---------------------------------------------------------------------------
class OptimizerMode(StrEnum):
    LOCAL_LLM = "LOCAL_LLM"
    MOCK = "MOCK"
    UNAVAILABLE = "UNAVAILABLE"


# ---------------------------------------------------------------------------
# Change record items
# ---------------------------------------------------------------------------
class ChangeRecord(BaseModel):
    """Documents a single deliberate change made during optimization."""

    model_config = ConfigDict(extra="forbid")

    category: str = Field(
        description=(
            "Category of change: clarity | specificity | structure | completeness | "
            "constraint | role | format | tone | placeholder_inserted"
        )
    )
    description: str = Field(description="Human-readable description of what changed and why.")


class PreservedRequirement(BaseModel):
    """Documents an original requirement that was explicitly preserved."""

    model_config = ConfigDict(extra="forbid")

    requirement: str = Field(
        description="The original requirement, constraint, or goal that was preserved verbatim."
    )
    reason: str = Field(description="Why this was kept unchanged.")


# ---------------------------------------------------------------------------
# Optimizer metadata
# ---------------------------------------------------------------------------
class OptimizerMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optimizer_version: str = Field(description="Semantic version of the optimizer component.")
    llm_model: str = Field(
        description="Model identifier used for optimization, or 'none' if LLM unavailable."
    )
    optimizer_mode: OptimizerMode = Field(
        description="Accurately reflects which optimization path was taken."
    )
    optimization_mode: OptimizationMode = Field(
        description="The optimization style applied."
    )
    latency_ms: float = Field(
        default=0.0,
        description="Total wall-clock latency of the optimization in milliseconds.",
    )


# ---------------------------------------------------------------------------
# Top-level optimization result
# ---------------------------------------------------------------------------
class OptimizerResult(BaseModel):
    """Complete structured output from the AI Optimizer."""

    model_config = ConfigDict(extra="forbid")

    optimized_prompt: str = Field(
        description=(
            "The rewritten, improved prompt. "
            "Uses [PLACEHOLDER] syntax for genuinely missing information. "
            "Original intent, domain, scope, and stated constraints are preserved."
        )
    )
    summary: str = Field(
        description="One-to-three sentence summary of the optimizations applied."
    )
    changes: list[ChangeRecord] = Field(
        default_factory=list,
        description="Ordered list of deliberate changes made during optimization.",
    )
    preserved_requirements: list[PreservedRequirement] = Field(
        default_factory=list,
        description="Original requirements explicitly preserved without modification.",
    )
    placeholders_inserted: list[str] = Field(
        default_factory=list,
        description=(
            "Placeholder strings inserted for missing information, "
            "e.g. '[TARGET AUDIENCE]', '[PROGRAMMING LANGUAGE]'."
        ),
    )
    improvement_score_delta: float = Field(
        default=0.0,
        description=(
            "Estimated improvement in prompt quality score relative to the original "
            "(positive = improvement). "
            "Model-assisted estimate only — not a calibrated measurement."
        ),
    )
    metadata: OptimizerMetadata


# ---------------------------------------------------------------------------
# API request / response wrappers
# ---------------------------------------------------------------------------
class OptimizerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: StrictStr = Field(description="The original prompt to optimize.")
    mode: OptimizationMode = Field(
        default=OptimizationMode.BALANCED,
        description="Optimization style: balanced | analytical | creative | expert",
    )
    analyzer_result: dict | None = Field(
        default=None,
        description=(
            "Optional serialised AnalyzerAnalysis from Step 15. "
            "When provided, the optimizer uses it as evidence context. "
            "When absent the optimizer runs its own lightweight context assembly."
        ),
    )

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty or whitespace-only")
        return value


class OptimizerResponse(BaseModel):
    """Top-level API response envelope for the optimizer endpoint."""

    model_config = ConfigDict(extra="forbid")

    result: OptimizerResult
    original_prompt_length: int = Field(description="Character count of the original prompt.")
    optimized_prompt_length: int = Field(description="Character count of the optimized prompt.")
    optimization_mode: OptimizationMode
