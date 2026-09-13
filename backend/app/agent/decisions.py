"""Deterministic Decision Engine for PromptLens AI Agent Loop (Step 19).

Invariants:
- 1. Validator PASS terminates successfully (COMPLETED, VALIDATED).
- 2. Validator NEEDS_REVIEW terminates immediately (NEEDS_REVIEW, VALIDATION_REVIEW).
- 3. Non-improving loops are detected and halted early (FAILED, VALIDATION_FAILED).
- 4. Iterations are strictly capped at max_iterations <= 3 (MAX_ITERATIONS_REACHED, MAX_ITERATIONS).
- 5. In all other cases, retries only if iteration < max_iterations.
"""
from __future__ import annotations

import difflib

from app.agent.config import (
    MAX_CONSECUTIVE_IDENTICAL_ERRORS,
    MAX_CONSECUTIVE_REGRESSIONS,
    MAX_OPTIMIZATION_ITERATIONS,
    MAX_PROMPT_EXPANSION_RATIO,
    MIN_CHAR_DELTA_RATIO,
    MIN_SEMANTIC_SIMILARITY,
)
from app.agent.schemas import (
    AgentDecision,
    AgentStatus,
    AgentTerminationReason,
)
from app.agent.state import AgentState
from app.validator.schemas import IssueSeverity, ValidatorDecision


def evaluate_agent_decision(
    state: AgentState,
    max_iterations: int = MAX_OPTIMIZATION_ITERATIONS,
) -> AgentDecision:
    """Evaluate whether the agent loop should continue or terminate, and why.

    Args:
        state: Current immutable AgentState after completing an iteration.
        max_iterations: Configured iteration cap (must be <= MAX_OPTIMIZATION_ITERATIONS).

    Returns:
        AgentDecision containing should_continue flag, target status, and reason.
    """
    if not state.iterations:
        return AgentDecision(
            should_continue=True,
            target_status=AgentStatus.UNAVAILABLE,
            termination_reason=None,
            explanation="No iterations executed yet; starting iteration 1.",
            is_non_improving=False,
        )

    last_iteration = state.iterations[-1]
    last_validator = state.validation_history[-1]
    last_critic = state.critic_history[-1]

    # -----------------------------------------------------------------------
    # Rule 1: Validator PASS -> Immediate success termination
    # -----------------------------------------------------------------------
    if last_validator.decision == ValidatorDecision.PASS:
        return AgentDecision(
            should_continue=False,
            target_status=AgentStatus.COMPLETED,
            termination_reason=AgentTerminationReason.VALIDATED,
            explanation=(
                f"Validator returned PASS at iteration {last_iteration.iteration_number}. "
                f"Prompt is successfully validated."
            ),
            is_non_improving=False,
        )

    # -----------------------------------------------------------------------
    # Rule 2: Validator NEEDS_REVIEW -> Immediate safe termination
    # -----------------------------------------------------------------------
    if last_validator.decision == ValidatorDecision.NEEDS_REVIEW:
        return AgentDecision(
            should_continue=False,
            target_status=AgentStatus.NEEDS_REVIEW,
            termination_reason=AgentTerminationReason.VALIDATION_REVIEW,
            explanation=(
                f"Validator returned NEEDS_REVIEW at iteration {last_iteration.iteration_number}. "
                f"Optimization stopped for human review."
            ),
            is_non_improving=False,
        )

    # -----------------------------------------------------------------------
    # Rule 3: Non-Improving Loop Prevention (Early termination for non-progress)
    # -----------------------------------------------------------------------
    non_improving_reason = check_non_improving_loop(state)
    if non_improving_reason:
        return AgentDecision(
            should_continue=False,
            target_status=AgentStatus.FAILED,
            termination_reason=AgentTerminationReason.VALIDATION_FAILED,
            explanation=(
                f"Non-improving loop detected at iteration {last_iteration.iteration_number}: "
                f"{non_improving_reason}. Safely terminating."
            ),
            is_non_improving=True,
        )

    # -----------------------------------------------------------------------
    # Rule 4: Maximum Iterations Reached
    # -----------------------------------------------------------------------
    effective_max = min(max_iterations, MAX_OPTIMIZATION_ITERATIONS)
    if state.iteration_count >= effective_max:
        return AgentDecision(
            should_continue=False,
            target_status=AgentStatus.MAX_ITERATIONS_REACHED,
            termination_reason=AgentTerminationReason.MAX_ITERATIONS,
            explanation=(
                f"Maximum allowed optimization iterations ({effective_max}) reached "
                f"without achieving Validator PASS."
            ),
            is_non_improving=False,
        )

    # -----------------------------------------------------------------------
    # Rule 5: Continue to Next Iteration
    # -----------------------------------------------------------------------
    return AgentDecision(
        should_continue=True,
        target_status=AgentStatus.FAILED,  # Status remains FAILED until a pass is achieved
        termination_reason=None,
        explanation=(
            f"Iteration {last_iteration.iteration_number} failed validation "
            f"({len(last_validator.issues)} issues). Proceeding to iteration "
            f"{state.iteration_count + 1} with structured diagnostic feedback."
        ),
        is_non_improving=False,
    )


def check_non_improving_loop(state: AgentState) -> str | None:
    """Check whether the agent loop has stagnated or degraded.

    Returns:
        A human-readable reason string if a non-improving condition is met;
        None otherwise.
    """
    if not state.iterations:
        return None

    last_iter = state.iterations[-1]
    last_validator = state.validation_history[-1]
    last_critic = state.critic_history[-1]

    # Check 1: Severe prompt bloat
    orig_len = max(len(state.original_prompt.strip()), 1)
    cand_len = len(last_iter.prompt_after.strip())
    expansion_ratio = cand_len / orig_len
    if expansion_ratio > MAX_PROMPT_EXPANSION_RATIO:
        return (
            f"Severe prompt bloat detected (expansion ratio {expansion_ratio:.2f}x "
            f"exceeds limit of {MAX_PROMPT_EXPANSION_RATIO}x)"
        )

    # Check 2: Semantic similarity collapse
    if last_critic.metadata.semantic_similarity < MIN_SEMANTIC_SIMILARITY:
        return (
            f"Semantic similarity collapse detected ({last_critic.metadata.semantic_similarity:.3f} "
            f"below floor of {MIN_SEMANTIC_SIMILARITY})"
        )

    # Check 3: Identical prompt candidate (no change from prompt before or previous candidate)
    if last_iter.prompt_after.strip() == last_iter.prompt_before.strip():
        return "Optimizer produced candidate prompt identical to input prompt"

    if len(state.iterations) >= 2:
        prev_iter = state.iterations[-2]
        if last_iter.prompt_after.strip() == prev_iter.prompt_after.strip():
            return "Optimizer produced candidate prompt identical to previous iteration"

    # Check 4: Negligible prompt change
    char_delta = abs(len(last_iter.prompt_after) - len(last_iter.prompt_before))
    char_ratio = char_delta / max(len(last_iter.prompt_before), 1)
    # Check if text edit similarity is nearly 100%
    matcher = difflib.SequenceMatcher(None, last_iter.prompt_before, last_iter.prompt_after)
    similarity = matcher.ratio()
    if similarity > (1.0 - MIN_CHAR_DELTA_RATIO) and last_validator.issues:
        return (
            f"Negligible prompt change detected ({similarity * 100:.1f}% identical) "
            f"with unresolved validation issues"
        )

    # Check 5: Repeated identical validation failure codes across 2 iterations
    if len(state.validation_history) >= MAX_CONSECUTIVE_IDENTICAL_ERRORS:
        curr_errors = {
            i.code for i in state.validation_history[-1].issues
            if i.severity == IssueSeverity.ERROR
        }
        prev_errors = {
            i.code for i in state.validation_history[-2].issues
            if i.severity == IssueSeverity.ERROR
        }
        if curr_errors and curr_errors == prev_errors:
            return (
                f"Repeated identical validation failure codes across consecutive iterations: "
                f"{sorted(curr_errors)}"
            )

    # Check 6: Consecutive score regressions
    if len(state.score_history) >= (MAX_CONSECUTIVE_REGRESSIONS + 1):
        s0, s1, s2 = state.score_history[-3], state.score_history[-2], state.score_history[-1]
        if s2 < s1 < s0:
            return (
                f"Consecutive score regression detected across {MAX_CONSECUTIVE_REGRESSIONS} "
                f"iterations ({s0:.1f} -> {s1:.1f} -> {s2:.1f})"
            )

    return None
