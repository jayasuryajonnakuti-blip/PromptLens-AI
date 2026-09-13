"""SQLAlchemy ORM models for PromptLens AI persistence layer (Step 20).

Invariants:
- Database timestamps are stored in UTC.
- UUID4 strings are used for primary identifiers (safe to expose publicly).
- Strict database constraints enforce non-null values, non-negative iterations, and score bounds [0, 100].
- Structured data is serialized as JSON-compatible objects; arbitrary Python objects are forbidden.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    """Generate a canonical UUID4 string."""
    return str(uuid.uuid4())


class AgentRun(Base):
    """Represents a single completed or terminated execution of the PromptLens Agent Loop."""

    __tablename__ = "agent_runs"

    # Primary Key
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
        nullable=False,
        doc="Canonical UUID4 identifier for this agent run.",
    )

    # Timestamps (Stored in UTC)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        doc="UTC timestamp when the agent run was initiated/created.",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
        doc="UTC timestamp when the agent run was last updated.",
    )

    # Prompts
    original_prompt: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Initial prompt submitted to the agent loop.",
    )
    final_prompt: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Final candidate prompt selected by the agent loop.",
    )

    # Execution State
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Terminal AgentStatus (COMPLETED, FAILED, NEEDS_REVIEW, MAX_ITERATIONS_REACHED, UNAVAILABLE).",
    )
    termination_reason: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Deterministic AgentTerminationReason (VALIDATED, VALIDATION_FAILED, VALIDATION_REVIEW, etc.).",
    )
    is_validated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        doc="True only if the final prompt achieved a Validator PASS.",
    )
    iteration_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Total number of optimization iterations executed.",
    )

    # Scores and Versioning
    final_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Composite quality score in [0.0, 100.0] if scored, or NULL.",
    )
    agent_version: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="1.0.0",
        doc="Agent component version used for this execution.",
    )
    mode: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="balanced",
        doc="Optimization mode used (balanced, analytical, creative, expert).",
    )

    # Structured JSON Fields (JSON-compatible only, no raw Python objects)
    iterations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        doc="Chronological list of serialized iteration records.",
    )
    score_history: Mapped[list[float]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        doc="List of scores across initial analysis and iterations.",
    )
    critic_result: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        doc="Serialized summary of the final Critic evaluation.",
    )
    final_validation: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        doc="Serialized final ValidationResult.",
    )
    metrics: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        doc="Serialized execution metrics (latency, expansion ratio, etc.).",
    )
    run_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        doc="Additional execution metadata and disclaimers.",
    )

    # Database-level Constraints
    __table_args__ = (
        CheckConstraint("iteration_count >= 0", name="check_iteration_count_non_negative"),
        CheckConstraint(
            "final_score IS NULL OR (final_score >= 0.0 AND final_score <= 100.0)",
            name="check_final_score_range",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<AgentRun(id={self.id!r}, status={self.status!r}, "
            f"iterations={self.iteration_count}, validated={self.is_validated})>"
        )
