"""Unit tests for deterministic decision logic and non-improving loop detection (Step 19)."""
from __future__ import annotations

import unittest

from app.agent.decisions import evaluate_agent_decision
from app.agent.schemas import (
    AgentIteration,
    AgentStatus,
    AgentTerminationReason,
    CriticResultSummary,
    OptimizerResultSummary,
)
from app.agent.state import AgentState
from app.critic.schemas import (
    CriticAnalysisMode,
    CriticDecision,
    CriticEvaluation,
    CriticMetadata,
    IntentPreservationResult,
    MetricChange,
    QualityStatus,
    RequirementPreservationResult,
)
from app.validator.schemas import (
    IssueSeverity,
    PromptMetadata,
    ValidationIssue,
    ValidationMetadata,
    ValidationMode,
    ValidationResult,
    ValidatorDecision,
)


class TestAgentDecisions(unittest.TestCase):
    """Test suite for Step 19 decision rules and non-improving loop prevention."""

    def _build_state_with_iteration(
        self,
        decision: ValidatorDecision,
        prompt_before: str = "Short original prompt",
        prompt_after: str = "A much better optimized prompt with clear instructions",
        issues: list[ValidationIssue] | None = None,
        semantic_sim: float = 0.85,
        score: float = 75.0,
        iteration_num: int = 1,
        prior_state: AgentState | None = None,
    ) -> AgentState:
        val_meta = ValidationMetadata(
            validator_version="1.0.0",
            validation_mode=ValidationMode.MOCK,
            model="none",
            latency_ms=10.0,
            critic_consistency_checked=True,
        )
        prompt_meta = PromptMetadata(
            prompt_length=len(prompt_after),
            word_count=len(prompt_after.split()),
            sentence_count=1,
            detected_intent="CODE",
            intent_confidence=0.9,
            quality_score=score,
        )
        val_res = ValidationResult(
            decision=decision,
            is_valid=(decision == ValidatorDecision.PASS),
            safety_score=85.0 if decision == ValidatorDecision.PASS else 45.0,
            original_metadata=prompt_meta,
            optimized_metadata=prompt_meta,
            issues=issues or [],
            metadata=val_meta,
        )
        critic_res = CriticEvaluation(
            decision=CriticDecision.PASS if decision == ValidatorDecision.PASS else CriticDecision.FAIL,
            overall_critique_score=score,
            intent_preservation=IntentPreservationResult(
                score=score,
                status=QualityStatus.GOOD if decision == ValidatorDecision.PASS else QualityStatus.FAIR,
                reason="Intent evaluation",
            ),
            requirement_preservation=RequirementPreservationResult(
                score=score,
                status=QualityStatus.GOOD if decision == ValidatorDecision.PASS else QualityStatus.POOR,
                reason="Requirement evaluation",
            ),
            clarity_change=MetricChange(original_score=60.0, optimized_score=score, delta=15.0, assessment="improved"),
            specificity_change=MetricChange(original_score=60.0, optimized_score=score, delta=15.0, assessment="improved"),
            ambiguity_change=MetricChange(original_score=60.0, optimized_score=score, delta=15.0, assessment="improved"),
            completeness_change=MetricChange(original_score=60.0, optimized_score=score, delta=15.0, assessment="improved"),
            metadata=CriticMetadata(
                critic_version="1.0.0",
                analysis_mode=CriticAnalysisMode.MOCK,
                model="none",
                semantic_similarity=semantic_sim,
                latency_ms=10.0,
            ),
        )
        iteration = AgentIteration(
            iteration_number=iteration_num,
            prompt_before=prompt_before,
            prompt_after=prompt_after,
            optimizer_result=OptimizerResultSummary(
                optimized_prompt=prompt_after,
                summary="summary",
            ),
            critic_result=CriticResultSummary(
                decision=decision.value,
                overall_critique_score=score,
                semantic_similarity=semantic_sim,
            ),
            validator_result=val_res,
            score_before=60.0,
            score_after=score,
            decision=decision.value,
            latency_ms=25.0,
        )
        base = prior_state or AgentState.create(prompt_before)
        return base.add_iteration(iteration, val_res, critic_res, new_score=score)

    def test_pass_stops_immediately(self) -> None:
        state = self._build_state_with_iteration(ValidatorDecision.PASS)
        dec = evaluate_agent_decision(state, max_iterations=3)
        self.assertFalse(dec.should_continue)
        self.assertEqual(dec.target_status, AgentStatus.COMPLETED)
        self.assertEqual(dec.termination_reason, AgentTerminationReason.VALIDATED)

    def test_needs_review_stops_immediately(self) -> None:
        state = self._build_state_with_iteration(ValidatorDecision.NEEDS_REVIEW)
        dec = evaluate_agent_decision(state, max_iterations=3)
        self.assertFalse(dec.should_continue)
        self.assertEqual(dec.target_status, AgentStatus.NEEDS_REVIEW)
        self.assertEqual(dec.termination_reason, AgentTerminationReason.VALIDATION_REVIEW)

    def test_fail_retries_if_iteration_under_max(self) -> None:
        issue = ValidationIssue(
            code="LOST_REQUIREMENT",
            severity=IssueSeverity.ERROR,
            message="Lost CSV delimiter requirement",
            field_or_scope="requirements",
            evidence="delimiter",
        )
        state = self._build_state_with_iteration(ValidatorDecision.FAIL, issues=[issue], iteration_num=1)
        dec = evaluate_agent_decision(state, max_iterations=3)
        self.assertTrue(dec.should_continue)
        self.assertEqual(dec.target_status, AgentStatus.FAILED)

    def test_fail_terminates_at_max_iterations(self) -> None:
        issue = ValidationIssue(
            code="LOST_REQUIREMENT",
            severity=IssueSeverity.ERROR,
            message="Lost CSV delimiter requirement",
            field_or_scope="requirements",
            evidence="delimiter",
        )
        state1 = self._build_state_with_iteration(ValidatorDecision.FAIL, prompt_after="cand 1", issues=[issue], iteration_num=1)
        # Change issue code on next iteration so identical error check isn't triggered
        issue2 = ValidationIssue(
            code="CONTRADICTION",
            severity=IssueSeverity.ERROR,
            message="Contradiction",
            field_or_scope="requirements",
            evidence="contradiction",
        )
        state2 = self._build_state_with_iteration(ValidatorDecision.FAIL, prompt_before="cand 1", prompt_after="cand 2", issues=[issue2], iteration_num=2, prior_state=state1)
        issue3 = ValidationIssue(
            code="FORMAT_ERROR",
            severity=IssueSeverity.ERROR,
            message="Format error",
            field_or_scope="format",
            evidence="format",
        )
        state3 = self._build_state_with_iteration(ValidatorDecision.FAIL, prompt_before="cand 2", prompt_after="cand 3", issues=[issue3], iteration_num=3, prior_state=state2)

        dec = evaluate_agent_decision(state3, max_iterations=3)
        self.assertFalse(dec.should_continue)
        self.assertEqual(dec.target_status, AgentStatus.MAX_ITERATIONS_REACHED)
        self.assertEqual(dec.termination_reason, AgentTerminationReason.MAX_ITERATIONS)

    def test_non_improving_identical_prompt(self) -> None:
        # Candidate is identical to prompt_before
        state = self._build_state_with_iteration(
            ValidatorDecision.FAIL,
            prompt_before="Identical prompt text",
            prompt_after="Identical prompt text",
        )
        dec = evaluate_agent_decision(state, max_iterations=3)
        self.assertFalse(dec.should_continue)
        self.assertTrue(dec.is_non_improving)
        self.assertEqual(dec.target_status, AgentStatus.FAILED)
        self.assertEqual(dec.termination_reason, AgentTerminationReason.VALIDATION_FAILED)

    def test_non_improving_severe_bloat(self) -> None:
        # Prompt length bloat > 4.0x
        short_prompt = "Short prompt"
        bloated_prompt = "A" * (len(short_prompt) * 5)
        state = self._build_state_with_iteration(
            ValidatorDecision.FAIL,
            prompt_before=short_prompt,
            prompt_after=bloated_prompt,
        )
        dec = evaluate_agent_decision(state, max_iterations=3)
        self.assertFalse(dec.should_continue)
        self.assertTrue(dec.is_non_improving)
        self.assertIn("Severe prompt bloat", dec.explanation)

    def test_non_improving_semantic_collapse(self) -> None:
        state = self._build_state_with_iteration(
            ValidatorDecision.FAIL,
            semantic_sim=0.25,  # Below 0.40 floor
        )
        dec = evaluate_agent_decision(state, max_iterations=3)
        self.assertFalse(dec.should_continue)
        self.assertTrue(dec.is_non_improving)
        self.assertIn("Semantic similarity collapse", dec.explanation)

    def test_non_improving_repeated_same_errors(self) -> None:
        issue = ValidationIssue(
            code="UNRESOLVED_CONTRADICTION",
            severity=IssueSeverity.ERROR,
            message="Same error",
            field_or_scope="scope",
            evidence="same",
        )
        state1 = self._build_state_with_iteration(
            ValidatorDecision.FAIL,
            prompt_before="Orig prompt text",
            prompt_after="Cand 1 different text",
            issues=[issue],
            iteration_num=1,
        )
        state2 = self._build_state_with_iteration(
            ValidatorDecision.FAIL,
            prompt_before="Cand 1 different text",
            prompt_after="Cand 2 completely different text",
            issues=[issue],
            iteration_num=2,
            prior_state=state1,
        )
        dec = evaluate_agent_decision(state2, max_iterations=3)
        self.assertFalse(dec.should_continue)
        self.assertTrue(dec.is_non_improving)
        self.assertIn("Repeated identical validation failure", dec.explanation)


if __name__ == "__main__":
    unittest.main()
