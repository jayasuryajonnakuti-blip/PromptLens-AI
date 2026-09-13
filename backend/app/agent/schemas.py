"""Strict Pydantic v2 schemas for PromptLens AI Agent Loop (Step 19).

Invariants:
- All models use extra="forbid".
- Zero use of Any.
- All input strings strictly checked against empty/whitespace-only values.
"""
from __future__ import annotations

from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from app.agent.config import (
    AGENT_DISCLAIMERS,
    DEFAULT_MAX_ITERATIONS,
    MAX_OPTIMIZATION_ITERATIONS,
)
from app.optimizer.schemas import OptimizationMode
from app.validator.schemas import ValidationResult


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class AgentStatus(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    MAX_ITERATIONS_REACHED = "MAX_ITERATIONS_REACHED"
    UNAVAILABLE = "UNAVAILABLE"


class AgentTerminationReason(StrEnum):
    VALIDATED = "VALIDATED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    VALIDATION_REVIEW = "VALIDATION_REVIEW"
    MAX_ITERATIONS = "MAX_ITERATIONS"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Subsystem Summary Models (Strict, typed, no Any)
# ---------------------------------------------------------------------------
class OptimizerResultSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optimized_prompt: str = Field(description="The improved prompt produced by the optimizer.")
    summary: str = Field(description="Summary of changes performed.")
    changes: list[dict[str, str]] = Field(
        default_factory=list,
        description="Structured category-description change records.",
    )
    preserved_requirements: list[dict[str, str]] = Field(
        default_factory=list,
        description="Explicit requirements preserved.",
    )
    placeholders_inserted: list[str] = Field(
        default_factory=list,
        description="Bracketed placeholder tags inserted.",
    )
    improvement_score_delta: float = Field(
        default=0.0,
        description="Estimated improvement score delta from optimizer.",
    )
    optimizer_mode: str = Field(
        default="none",
        description="Execution mode of the optimizer (LOCAL_LLM, MOCK, UNAVAILABLE).",
    )


class CriticResultSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: str = Field(description="Critic verdict: PASS, FAIL, or NEEDS_REVIEW.")
    overall_critique_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Overall critique score in [0, 100].",
    )
    issues: list[dict[str, str]] = Field(
        default_factory=list,
        description="Issues discovered by the Critic.",
    )
    lost_requirements: list[str] = Field(
        default_factory=list,
        description="Requirements flagged as lost.",
    )
    introduced_requirements: list[str] = Field(
        default_factory=list,
        description="Requirements flagged as introduced.",
    )
    unsupported_assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions flagged as unsupported.",
    )
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    semantic_similarity: float = Field(
        ge=0.0,
        le=1.0,
        default=1.0,
        description="Cosine similarity between prompt embeddings.",
    )


# ---------------------------------------------------------------------------
# Iteration Tracking Model
# ---------------------------------------------------------------------------
class AgentIteration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration_number: int = Field(
        ge=1,
        le=MAX_OPTIMIZATION_ITERATIONS,
        description="1-indexed iteration number (1, 2, or 3).",
    )
    prompt_before: str = Field(description="Prompt input to this iteration.")
    prompt_after: str = Field(description="Prompt produced by the optimizer in this iteration.")
    optimizer_result: OptimizerResultSummary = Field(description="Summary of optimizer execution.")
    critic_result: CriticResultSummary = Field(description="Summary of critic execution.")
    validator_result: ValidationResult = Field(description="Full validation result.")
    score_before: float | None = Field(default=None, description="Score before optimization attempt.")
    score_after: float | None = Field(default=None, description="Score after optimization attempt.")
    decision: str = Field(description="Validator decision for this iteration (PASS, FAIL, NEEDS_REVIEW).")
    latency_ms: float = Field(ge=0.0, default=0.0, description="Wall-clock latency of this iteration in ms.")


# ---------------------------------------------------------------------------
# Metrics & Summaries
# ---------------------------------------------------------------------------
class AgentMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_latency_ms: float = Field(ge=0.0, default=0.0)
    total_iterations: int = Field(ge=0, le=MAX_OPTIMIZATION_ITERATIONS, default=0)
    initial_score: float | None = Field(default=None)
    final_score: float | None = Field(default=None)
    score_delta: float | None = Field(default=None)
    prompt_length_before: int = Field(ge=0, default=0)
    prompt_length_after: int = Field(ge=0, default=0)
    expansion_ratio: float = Field(ge=0.0, default=1.0)
    semantic_similarity: float | None = Field(default=None)


class AgentStateSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration_count: int = Field(ge=0, le=MAX_OPTIMIZATION_ITERATIONS)
    current_status: AgentStatus
    last_decision: str | None = None
    is_validated: bool = False


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    should_continue: bool = Field(description="Whether the agent loop should proceed to another iteration.")
    target_status: AgentStatus = Field(description="Resulting agent status.")
    termination_reason: AgentTerminationReason | None = Field(
        default=None,
        description="Reason for termination, if loop is terminating.",
    )
    explanation: str = Field(description="Deterministic explanation of why this decision was reached.")
    is_non_improving: bool = Field(
        default=False,
        description="True if termination was triggered by non-improving loop detection.",
    )


# ---------------------------------------------------------------------------
# Request & Response Models
# ---------------------------------------------------------------------------
class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: StrictStr = Field(description="Original user prompt to iteratively analyze, optimize, and validate.")
    mode: OptimizationMode = Field(
        default=OptimizationMode.BALANCED,
        description="Optimization mode (balanced, analytical, creative, expert).",
    )
    max_iterations: int = Field(
        default=DEFAULT_MAX_ITERATIONS,
        ge=1,
        le=MAX_OPTIMIZATION_ITERATIONS,
        description=f"Maximum optimization attempts (hard limit: {MAX_OPTIMIZATION_ITERATIONS}).",
    )

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty or whitespace-only")
        return value


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_prompt: str = Field(description="The original user prompt.")
    final_prompt: str = Field(
        description="The final prompt selected. If unvalidated, is_validated is False."
    )
    status: AgentStatus = Field(description="High-level agent loop status.")
    termination_reason: AgentTerminationReason = Field(description="Deterministic termination reason.")
    iteration_count: int = Field(
        ge=0,
        le=MAX_OPTIMIZATION_ITERATIONS,
        description="Number of iterations executed.",
    )
    is_validated: bool = Field(
        description="True ONLY if the final prompt achieved a Validator PASS."
    )
    iterations: list[AgentIteration] = Field(
        default_factory=list,
        description="Complete chronological iteration history.",
    )
    final_validation: ValidationResult | None = Field(
        default=None,
        description="Validation result of the final iteration.",
    )
    final_critic_result: CriticResultSummary | None = Field(
        default=None,
        description="Critic summary of the final iteration.",
    )
    final_score: float | None = Field(
        default=None,
        description="Final composite quality score of the prompt.",
    )
    metrics: AgentMetrics = Field(description="Aggregated execution metrics.")
    run_id: str | None = Field(
        default=None,
        description="Unique UUID4 identifier of the persisted AgentRun record, if saved.",
    )
    persistence_status: str | None = Field(
        default=None,
        description="Persistence status indicator ('persisted' or 'failed').",
    )
    disclaimers: list[str] = Field(
        default_factory=lambda: list(AGENT_DISCLAIMERS),
        description="Mandatory architectural disclaimers.",
    )
