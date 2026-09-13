"""Strict Pydantic schemas for AgentRun history and persistence API (Step 21)."""
from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class AgentRunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: str = Field(description="Canonical UUID4 identifier for the agent run.")
    created_at: datetime = Field(description="UTC timestamp when the run was created.")
    updated_at: datetime = Field(description="UTC timestamp when the run was last updated.")
    original_prompt: str = Field(description="Initial prompt submitted to the agent loop.")
    final_prompt: str = Field(description="Final candidate prompt selected by the agent loop.")
    status: str = Field(description="Terminal AgentStatus string.")
    termination_reason: str = Field(description="Deterministic AgentTerminationReason string.")
    is_validated: bool = Field(description="True only if the final prompt achieved a Validator PASS.")
    iteration_count: int = Field(ge=0, description="Total number of optimization iterations executed.")
    final_score: float | None = Field(default=None, description="Composite quality score if scored.")
    agent_version: str = Field(default="1.0.0", description="Agent version used for this execution.")
    mode: str = Field(default="balanced", description="Optimization mode used.")


class AgentRunDetail(AgentRunSummary):
    iterations: list[dict[str, Any]] = Field(default_factory=list, description="Serialized iteration records.")
    score_history: list[float] = Field(default_factory=list, description="Score history across iterations.")
    critic_result: dict[str, Any] | None = Field(default=None, description="Final Critic evaluation summary.")
    final_validation: dict[str, Any] | None = Field(default=None, description="Final ValidationResult.")
    metrics: dict[str, Any] | None = Field(default=None, description="Execution metrics.")
    run_metadata: dict[str, Any] | None = Field(default=None, description="Execution metadata and disclaimers.")


class AgentRunListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[AgentRunSummary] = Field(default_factory=list, description="List of persisted run summaries.")
    total: int = Field(ge=0, description="Total count of persisted agent runs.")
    skip: int = Field(ge=0, description="Pagination offset.")
    limit: int = Field(ge=1, description="Pagination limit.")


class DeleteRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool = Field(description="True if record was successfully deleted.")
    message: str = Field(description="Status description message.")
    deleted_id: str = Field(description="ID of the deleted agent run.")
