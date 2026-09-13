"""AgentOrchestrator: Bounded Agent Loop Orchestrator for PromptLens AI (Step 19).

Coordinates:
- Step 15: Analyzer
- Step 16: Optimizer
- Step 17: Critic
- Step 18: Validator

Invariants:
- Loop is bounded to a hard maximum of 3 iterations (MAX_OPTIMIZATION_ITERATIONS).
- Validator PASS terminates the loop immediately (COMPLETED, VALIDATED).
- Validator NEEDS_REVIEW terminates the loop immediately (NEEDS_REVIEW, VALIDATION_REVIEW).
- Non-improving loops (identical prompt, bloat, collapse, score regression) terminate early.
- An unvalidated prompt is NEVER marked as successfully validated.
- Zero raw stack traces exposed.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from app.agent.config import (
    DEFAULT_MAX_ITERATIONS,
    MAX_OPTIMIZATION_ITERATIONS,
)
from app.agent.context import build_iteration_context
from app.agent.decisions import evaluate_agent_decision
from app.agent.errors import AgentException, AgentServiceUnavailableError
from app.agent.schemas import (
    AgentIteration,
    AgentMetrics,
    AgentRequest,
    AgentResponse,
    AgentStatus,
    AgentTerminationReason,
    CriticResultSummary,
    OptimizerResultSummary,
)
from app.agent.state import AgentState
from app.analyzer.schemas import AnalyzerAnalysis
from app.critic.schemas import (
    CriticEvaluation,
    OptimizationMetadataInput as CriticOptimizationMetadata,
)
from app.optimizer.schemas import OptimizationMode, OptimizerResult
from app.validator.schemas import (
    CriticResultInput,
    OptimizationMetadataInput as ValidatorOptimizationMetadata,
    ValidationResult,
    ValidatorDecision,
)

LOGGER = logging.getLogger(__name__)


class AgentOrchestrator:
    """Coordinates iterative prompt analysis, optimization, critique, and validation."""

    def run(
        self,
        request: AgentRequest,
    ) -> AgentResponse:
        """Execute the bounded Agent Loop.

        Args:
            request: Validated AgentRequest containing prompt, mode, and max_iterations.

        Returns:
            Structured AgentResponse containing iteration history, metrics, and decision.
        """
        start_time = time.perf_counter()
        original_prompt = request.prompt.strip()
        effective_max = min(request.max_iterations, MAX_OPTIMIZATION_ITERATIONS)

        # -------------------------------------------------------------------
        # STEP A — INPUT: Initialize State
        # -------------------------------------------------------------------
        state = AgentState.create(original_prompt)

        # -------------------------------------------------------------------
        # STEP B — ANALYSIS: Run Analyzer on Original Prompt
        # -------------------------------------------------------------------
        analyzer_analysis: AnalyzerAnalysis | None = None
        analyzer_dict: dict[str, Any] | None = None
        initial_score: float | None = None

        try:
            from app.services.analyzer_service import get_analyzer_service
            analyzer_svc = get_analyzer_service()
            analyzer_analysis = analyzer_svc.analyze(original_prompt)
            analyzer_dict = analyzer_analysis.model_dump()
            from app.services.scoring_service import score_prompt
            score_resp = score_prompt(original_prompt)
            initial_score = score_resp.overall_score.score
        except Exception as exc:
            LOGGER.warning("Analyzer or scoring error during initial analysis: %s; using fallback.", exc)
            initial_score = 50.0

        state = state.with_analyzer(
            analyzer_result=analyzer_analysis,
            analyzer_dict=analyzer_dict,
            initial_score=initial_score,
        )

        # -------------------------------------------------------------------
        # ITERATION LOOP: Up to effective_max attempts
        # -------------------------------------------------------------------
        try:
            from app.services.critic_service import get_critic_service
            from app.services.optimizer_service import get_optimizer_service
            from app.services.validator_service import get_validator_service

            optimizer_svc = get_optimizer_service()
            critic_svc = get_critic_service()
            validator_svc = get_validator_service()

            for iteration_idx in range(1, effective_max + 1):
                iter_start = time.perf_counter()
                prompt_before = state.current_prompt

                # -----------------------------------------------------------
                # STEP C — OPTIMIZATION
                # -----------------------------------------------------------
                context_dict = build_iteration_context(state, request.mode)
                opt_result = optimizer_svc.optimize(
                    prompt=state.original_prompt,
                    mode=request.mode,
                    analyzer_result=context_dict,
                )
                candidate_prompt = opt_result.optimized_prompt

                opt_summary = OptimizerResultSummary(
                    optimized_prompt=candidate_prompt,
                    summary=opt_result.summary,
                    changes=[
                        {"category": c.category, "description": c.description}
                        for c in opt_result.changes
                    ],
                    preserved_requirements=[
                        {"requirement": p.requirement, "reason": p.reason}
                        for p in opt_result.preserved_requirements
                    ],
                    placeholders_inserted=list(opt_result.placeholders_inserted),
                    improvement_score_delta=opt_result.improvement_score_delta,
                    optimizer_mode=opt_result.metadata.optimizer_mode.value,
                )

                # -----------------------------------------------------------
                # STEP D — CRITIC
                # -----------------------------------------------------------
                critic_meta = CriticOptimizationMetadata(
                    optimization_mode=opt_result.metadata.optimization_mode.value,
                    changes=opt_summary.changes,
                    preserved_requirements=opt_summary.preserved_requirements,
                    placeholders_inserted=opt_summary.placeholders_inserted,
                )
                critic_eval = critic_svc.evaluate(
                    original_prompt=state.original_prompt,
                    optimized_prompt=candidate_prompt,
                    optimization_metadata=critic_meta,
                )

                critic_summary = CriticResultSummary(
                    decision=critic_eval.decision.value,
                    overall_critique_score=critic_eval.overall_critique_score,
                    issues=[
                        {
                            "category": str(getattr(i, "type", "") or ""),
                            "severity": i.severity.value,
                            "description": i.description,
                        }
                        for i in critic_eval.issues
                    ],
                    lost_requirements=list(critic_eval.lost_requirements),
                    introduced_requirements=list(critic_eval.introduced_requirements),
                    unsupported_assumptions=list(critic_eval.unsupported_assumptions),
                    strengths=list(critic_eval.strengths),
                    weaknesses=list(critic_eval.weaknesses),
                    recommendations=list(critic_eval.recommendations),
                    semantic_similarity=critic_eval.metadata.semantic_similarity,
                )

                # -----------------------------------------------------------
                # STEP E — VALIDATOR
                # -----------------------------------------------------------
                critic_input = CriticResultInput(
                    decision=critic_eval.decision.value,
                    overall_critique_score=critic_eval.overall_critique_score,
                    issues=[
                        {
                            "severity": i.severity.value,
                            "category": str(getattr(i, "type", "") or ""),
                            "description": i.description,
                        }
                        for i in critic_eval.issues
                    ],
                    lost_requirements=list(critic_eval.lost_requirements),
                    introduced_requirements=list(critic_eval.introduced_requirements),
                    unsupported_assumptions=list(critic_eval.unsupported_assumptions),
                )
                val_meta = ValidatorOptimizationMetadata(
                    optimization_mode=opt_result.metadata.optimization_mode.value,
                    changes=opt_summary.changes,
                    preserved_requirements=opt_summary.preserved_requirements,
                    placeholders_inserted=opt_summary.placeholders_inserted,
                )
                val_result = validator_svc.validate(
                    original_prompt=state.original_prompt,
                    optimized_prompt=candidate_prompt,
                    critic_result=critic_input,
                    optimization_metadata=val_meta,
                )

                score_before = (
                    state.score_history[-1] if state.score_history else state.initial_score
                )
                score_after = val_result.optimized_metadata.quality_score
                iter_latency = round((time.perf_counter() - iter_start) * 1000, 2)

                # Record completed iteration
                iteration_record = AgentIteration(
                    iteration_number=iteration_idx,
                    prompt_before=prompt_before,
                    prompt_after=candidate_prompt,
                    optimizer_result=opt_summary,
                    critic_result=critic_summary,
                    validator_result=val_result,
                    score_before=score_before,
                    score_after=score_after,
                    decision=val_result.decision.value,
                    latency_ms=iter_latency,
                )

                state = state.add_iteration(
                    iteration=iteration_record,
                    validation_result=val_result,
                    critic_result=critic_eval,
                    new_score=score_after,
                )

                # -----------------------------------------------------------
                # STEP F — DECISION
                # -----------------------------------------------------------
                decision = evaluate_agent_decision(state, effective_max)

                if not decision.should_continue:
                    # Final prompt selection:
                    # Only mark is_validated True if Validator PASS was achieved.
                    if val_result.decision == ValidatorDecision.PASS:
                        final_prompt = candidate_prompt
                        is_valid = True
                    else:
                        # Unvalidated candidate prompt clearly marked as unvalidated
                        final_prompt = candidate_prompt
                        is_valid = False

                    state = state.with_termination(
                        status=decision.target_status,
                        termination_reason=decision.termination_reason or AgentTerminationReason.VALIDATION_FAILED,
                        final_prompt=final_prompt,
                        is_validated=is_valid,
                    )
                    break

        except Exception as exc:
            LOGGER.error("Agent loop caught unexpected exception during execution: %s", exc, exc_info=True)
            # Safe termination without crashing or exposing raw stack trace
            total_elapsed = round((time.perf_counter() - start_time) * 1000, 2)
            last_val = state.validation_history[-1] if state.validation_history else None
            last_critic_summary = None
            if state.critic_history:
                c = state.critic_history[-1]
                last_critic_summary = CriticResultSummary(
                    decision=c.decision.value,
                    overall_critique_score=c.overall_critique_score,
                    issues=[
                        {"category": str(getattr(i, "type", "") or ""), "severity": i.severity.value, "description": i.description}
                        for i in c.issues
                    ],
                    lost_requirements=list(c.lost_requirements),
                    introduced_requirements=list(c.introduced_requirements),
                    unsupported_assumptions=list(c.unsupported_assumptions),
                    strengths=list(c.strengths),
                    weaknesses=list(c.weaknesses),
                    recommendations=list(c.recommendations),
                    semantic_similarity=c.metadata.semantic_similarity,
                )

            return AgentResponse(
                original_prompt=state.original_prompt,
                final_prompt=state.final_prompt or state.original_prompt,
                status=AgentStatus.FAILED,
                termination_reason=AgentTerminationReason.ERROR,
                iteration_count=state.iteration_count,
                is_validated=False,
                iterations=list(state.iterations),
                final_validation=last_val,
                final_critic_result=last_critic_summary,
                final_score=state.score_history[-1] if state.score_history else None,
                metrics=AgentMetrics(
                    total_latency_ms=total_elapsed,
                    total_iterations=state.iteration_count,
                    initial_score=state.initial_score,
                    final_score=state.score_history[-1] if state.score_history else None,
                    score_delta=None,
                    prompt_length_before=len(state.original_prompt),
                    prompt_length_after=len(state.final_prompt or state.original_prompt),
                    expansion_ratio=len(state.final_prompt or state.original_prompt) / max(len(state.original_prompt), 1),
                    semantic_similarity=None,
                ),
            )

        # If loop finished all iterations without explicit break
        if state.status == AgentStatus.UNAVAILABLE or state.termination_reason is None:
            last_candidate = state.iterations[-1].prompt_after if state.iterations else state.original_prompt
            last_val = state.validation_history[-1] if state.validation_history else None
            is_valid = (last_val.is_valid if last_val else False)
            state = state.with_termination(
                status=AgentStatus.MAX_ITERATIONS_REACHED,
                termination_reason=AgentTerminationReason.MAX_ITERATIONS,
                final_prompt=last_candidate,
                is_validated=is_valid,
            )

        # -------------------------------------------------------------------
        # Build Final Response
        # -------------------------------------------------------------------
        total_latency = round((time.perf_counter() - start_time) * 1000, 2)
        initial_sc = state.initial_score
        final_sc = state.score_history[-1] if state.score_history else initial_sc
        score_delta = round(final_sc - initial_sc, 2) if (final_sc is not None and initial_sc is not None) else None
        orig_len = len(state.original_prompt)
        final_len = len(state.final_prompt or state.original_prompt)
        expansion_ratio = round(final_len / max(orig_len, 1), 2)
        final_similarity = (
            state.critic_history[-1].metadata.semantic_similarity
            if state.critic_history
            else None
        )

        last_val = state.validation_history[-1] if state.validation_history else None
        last_critic_summary = None
        if state.critic_history:
            c = state.critic_history[-1]
            last_critic_summary = CriticResultSummary(
                decision=c.decision.value,
                overall_critique_score=c.overall_critique_score,
                issues=[
                    {
                        "category": str(getattr(i, "type", "") or ""),
                        "severity": i.severity.value,
                        "description": i.description,
                    }
                    for i in c.issues
                ],
                lost_requirements=list(c.lost_requirements),
                introduced_requirements=list(c.introduced_requirements),
                unsupported_assumptions=list(c.unsupported_assumptions),
                strengths=list(c.strengths),
                weaknesses=list(c.weaknesses),
                recommendations=list(c.recommendations),
                semantic_similarity=c.metadata.semantic_similarity,
            )

        return AgentResponse(
            original_prompt=state.original_prompt,
            final_prompt=state.final_prompt or state.original_prompt,
            status=state.status,
            termination_reason=state.termination_reason or AgentTerminationReason.VALIDATION_FAILED,
            iteration_count=state.iteration_count,
            is_validated=state.is_validated,
            iterations=list(state.iterations),
            final_validation=last_val,
            final_critic_result=last_critic_summary,
            final_score=final_sc,
            metrics=AgentMetrics(
                total_latency_ms=total_latency,
                total_iterations=state.iteration_count,
                initial_score=initial_sc,
                final_score=final_sc,
                score_delta=score_delta,
                prompt_length_before=orig_len,
                prompt_length_after=final_len,
                expansion_ratio=expansion_ratio,
                semantic_similarity=final_similarity,
            ),
        )
