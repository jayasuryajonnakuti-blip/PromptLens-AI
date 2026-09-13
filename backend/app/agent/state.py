"""AgentState: Explicit, traceable state tracking for the Agent Loop (Step 19).

Invariants:
- Historical iteration records are immutable and cannot be mutated by later steps.
- Uses tuples for collection fields to prevent in-place mutation.
- State transitions create new state instances, ensuring strict traceability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent.config import MAX_OPTIMIZATION_ITERATIONS
from app.agent.schemas import (
    AgentIteration,
    AgentStateSummary,
    AgentStatus,
    AgentTerminationReason,
)
from app.analyzer.schemas import AnalyzerAnalysis
from app.critic.schemas import CriticEvaluation
from app.validator.schemas import ValidationResult


@dataclass(frozen=True)
class AgentState:
    """Immutable agent state capturing complete chronological execution history."""

    original_prompt: str
    current_prompt: str
    iteration_count: int = 0
    analyzer_result: AnalyzerAnalysis | None = None
    analyzer_dict: dict[str, Any] | None = None
    initial_score: float | None = None
    iterations: tuple[AgentIteration, ...] = ()
    score_history: tuple[float, ...] = ()
    validation_history: tuple[ValidationResult, ...] = ()
    critic_history: tuple[CriticEvaluation, ...] = ()
    status: AgentStatus = AgentStatus.UNAVAILABLE
    termination_reason: AgentTerminationReason | None = None
    final_prompt: str | None = None
    is_validated: bool = False

    @classmethod
    def create(cls, original_prompt: str) -> AgentState:
        """Create an initial empty AgentState for the given prompt."""
        return cls(
            original_prompt=original_prompt,
            current_prompt=original_prompt,
            iteration_count=0,
            analyzer_result=None,
            analyzer_dict=None,
            initial_score=None,
            iterations=(),
            score_history=(),
            validation_history=(),
            critic_history=(),
            status=AgentStatus.UNAVAILABLE,
            termination_reason=None,
            final_prompt=original_prompt,
            is_validated=False,
        )

    def with_analyzer(
        self,
        analyzer_result: AnalyzerAnalysis | None,
        analyzer_dict: dict[str, Any] | None,
        initial_score: float | None = None,
    ) -> AgentState:
        """Return a new state recording initial analysis and scoring."""
        scores = (initial_score,) if initial_score is not None else ()
        return AgentState(
            original_prompt=self.original_prompt,
            current_prompt=self.current_prompt,
            iteration_count=self.iteration_count,
            analyzer_result=analyzer_result,
            analyzer_dict=analyzer_dict,
            initial_score=initial_score,
            iterations=self.iterations,
            score_history=scores,
            validation_history=self.validation_history,
            critic_history=self.critic_history,
            status=self.status,
            termination_reason=self.termination_reason,
            final_prompt=self.final_prompt,
            is_validated=self.is_validated,
        )

    def add_iteration(
        self,
        iteration: AgentIteration,
        validation_result: ValidationResult,
        critic_result: CriticEvaluation,
        new_score: float | None = None,
    ) -> AgentState:
        """Return a new state with the completed iteration appended."""
        new_scores = (
            self.score_history + (new_score,)
            if new_score is not None
            else self.score_history
        )
        return AgentState(
            original_prompt=self.original_prompt,
            current_prompt=iteration.prompt_after,
            iteration_count=self.iteration_count + 1,
            analyzer_result=self.analyzer_result,
            analyzer_dict=self.analyzer_dict,
            initial_score=self.initial_score,
            iterations=self.iterations + (iteration,),
            score_history=new_scores,
            validation_history=self.validation_history + (validation_result,),
            critic_history=self.critic_history + (critic_result,),
            status=self.status,
            termination_reason=self.termination_reason,
            final_prompt=iteration.prompt_after,
            is_validated=validation_result.is_valid,
        )

    def with_termination(
        self,
        status: AgentStatus,
        termination_reason: AgentTerminationReason,
        final_prompt: str | None = None,
        is_validated: bool | None = None,
    ) -> AgentState:
        """Return a new terminal state with status and final prompt set."""
        resolved_final = (
            final_prompt if final_prompt is not None else self.final_prompt or self.original_prompt
        )
        resolved_is_valid = (
            is_validated if is_validated is not None else self.is_validated
        )
        return AgentState(
            original_prompt=self.original_prompt,
            current_prompt=self.current_prompt,
            iteration_count=self.iteration_count,
            analyzer_result=self.analyzer_result,
            analyzer_dict=self.analyzer_dict,
            initial_score=self.initial_score,
            iterations=self.iterations,
            score_history=self.score_history,
            validation_history=self.validation_history,
            critic_history=self.critic_history,
            status=status,
            termination_reason=termination_reason,
            final_prompt=resolved_final,
            is_validated=resolved_is_valid,
        )

    def get_summary(self) -> AgentStateSummary:
        """Generate a compact summary of current state."""
        last_decision = self.iterations[-1].decision if self.iterations else None
        return AgentStateSummary(
            iteration_count=self.iteration_count,
            current_status=self.status,
            last_decision=last_decision,
            is_validated=self.is_validated,
        )
