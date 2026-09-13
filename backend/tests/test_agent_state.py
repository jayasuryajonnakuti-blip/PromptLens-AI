"""Unit tests for Agent Loop state immutability and tracking (Step 19)."""
from __future__ import annotations

import unittest

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
    PromptMetadata,
    ValidationMetadata,
    ValidationMode,
    ValidationResult,
    ValidatorDecision,
)


class TestAgentState(unittest.TestCase):
    """Test suite for AgentState immutability and lifecycle transitions."""

    def _create_mock_iteration(self, num: int, decision: str = "FAIL") -> tuple[AgentIteration, ValidationResult, CriticEvaluation]:
        val_meta = ValidationMetadata(
            validator_version="1.0.0",
            validation_mode=ValidationMode.MOCK,
            model="none",
            latency_ms=10.0,
            critic_consistency_checked=True,
        )
        prompt_meta = PromptMetadata(
            prompt_length=20,
            word_count=4,
            sentence_count=1,
            detected_intent="CODE",
            intent_confidence=0.9,
            quality_score=75.0,
        )
        val_res = ValidationResult(
            decision=ValidatorDecision(decision),
            is_valid=(decision == "PASS"),
            safety_score=80.0,
            original_metadata=prompt_meta,
            optimized_metadata=prompt_meta,
            metadata=val_meta,
        )
        critic_res = CriticEvaluation(
            decision=CriticDecision(decision),
            overall_critique_score=75.0,
            intent_preservation=IntentPreservationResult(
                score=80.0,
                status=QualityStatus.GOOD,
                reason="Intent preserved",
            ),
            requirement_preservation=RequirementPreservationResult(
                score=80.0,
                status=QualityStatus.GOOD,
                reason="Requirements preserved",
            ),
            clarity_change=MetricChange(original_score=70.0, optimized_score=75.0, delta=5.0, assessment="improved"),
            specificity_change=MetricChange(original_score=70.0, optimized_score=75.0, delta=5.0, assessment="improved"),
            ambiguity_change=MetricChange(original_score=70.0, optimized_score=75.0, delta=5.0, assessment="improved"),
            completeness_change=MetricChange(original_score=70.0, optimized_score=75.0, delta=5.0, assessment="improved"),
            metadata=CriticMetadata(
                critic_version="1.0.0",
                analysis_mode=CriticAnalysisMode.MOCK,
                model="none",
                semantic_similarity=0.9,
                latency_ms=10.0,
            ),
        )
        iteration = AgentIteration(
            iteration_number=num,
            prompt_before="orig",
            prompt_after=f"cand_{num}",
            optimizer_result=OptimizerResultSummary(
                optimized_prompt=f"cand_{num}",
                summary="summary",
            ),
            critic_result=CriticResultSummary(
                decision=decision,
                overall_critique_score=75.0,
            ),
            validator_result=val_res,
            score_before=70.0,
            score_after=75.0,
            decision=decision,
            latency_ms=50.0,
        )
        return iteration, val_res, critic_res

    def test_state_creation_and_defaults(self) -> None:
        state = AgentState.create("Write a prompt")
        self.assertEqual(state.original_prompt, "Write a prompt")
        self.assertEqual(state.current_prompt, "Write a prompt")
        self.assertEqual(state.iteration_count, 0)
        self.assertEqual(len(state.iterations), 0)
        self.assertEqual(len(state.score_history), 0)
        self.assertFalse(state.is_validated)

    def test_state_immutability(self) -> None:
        state0 = AgentState.create("Initial")
        state1 = state0.with_analyzer(analyzer_result=None, analyzer_dict=None, initial_score=60.0)

        # state0 remains unchanged
        self.assertIsNone(state0.initial_score)
        self.assertEqual(state1.initial_score, 60.0)

        iter1, val1, crit1 = self._create_mock_iteration(1, "FAIL")
        state2 = state1.add_iteration(iter1, val1, crit1, new_score=65.0)

        # state1 remains with 0 iterations
        self.assertEqual(state1.iteration_count, 0)
        self.assertEqual(len(state1.iterations), 0)

        # state2 has 1 iteration
        self.assertEqual(state2.iteration_count, 1)
        self.assertEqual(len(state2.iterations), 1)
        self.assertEqual(state2.current_prompt, "cand_1")
        self.assertEqual(state2.score_history, (60.0, 65.0))

    def test_state_termination(self) -> None:
        state = AgentState.create("Initial")
        iter1, val1, crit1 = self._create_mock_iteration(1, "PASS")
        state = state.add_iteration(iter1, val1, crit1, new_score=85.0)
        final_state = state.with_termination(
            status=AgentStatus.COMPLETED,
            termination_reason=AgentTerminationReason.VALIDATED,
            final_prompt="cand_1",
            is_validated=True,
        )
        self.assertEqual(final_state.status, AgentStatus.COMPLETED)
        self.assertEqual(final_state.termination_reason, AgentTerminationReason.VALIDATED)
        self.assertTrue(final_state.is_validated)
        self.assertEqual(final_state.final_prompt, "cand_1")

    def test_get_summary(self) -> None:
        state = AgentState.create("Initial")
        iter1, val1, crit1 = self._create_mock_iteration(1, "FAIL")
        state = state.add_iteration(iter1, val1, crit1, new_score=65.0)
        summary = state.get_summary()
        self.assertEqual(summary.iteration_count, 1)
        self.assertEqual(summary.last_decision, "FAIL")
        self.assertFalse(summary.is_validated)


if __name__ == "__main__":
    unittest.main()
