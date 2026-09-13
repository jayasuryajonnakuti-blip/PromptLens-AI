"""Unit tests for Agent Loop context building and feedback injection (Step 19)."""
from __future__ import annotations

import unittest

from app.agent.config import AGENT_CONTEXT_CHAR_LIMIT
from app.agent.context import build_iteration_context
from app.agent.schemas import (
    AgentIteration,
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
from app.optimizer.schemas import OptimizationMode
from app.validator.schemas import (
    IssueSeverity,
    PromptMetadata,
    ValidationIssue,
    ValidationMetadata,
    ValidationMode,
    ValidationResult,
    ValidatorDecision,
)


class TestAgentContext(unittest.TestCase):
    """Test suite for Step 19 context building with structured diagnostic feedback."""

    def test_iteration_1_context_clean(self) -> None:
        state = AgentState.create("Write a python script to parse CSV")
        state = state.with_analyzer(
            analyzer_result=None,
            analyzer_dict={
                "interpreted_goal": "Parse CSV in python",
                "intent": {"label": "CODE", "confidence": 0.95},
                "weaknesses": [{"weakness": "Missing error handling"}],
                "recommendations": [{"recommendation": "Specify dialect"}],
            },
            initial_score=60.0,
        )

        ctx = build_iteration_context(state, OptimizationMode.BALANCED)
        self.assertEqual(ctx["interpreted_goal"], "Parse CSV in python")
        self.assertEqual(len(ctx["weaknesses"]), 1)
        self.assertEqual(len(ctx["recommendations"]), 1)
        self.assertLessEqual(len(str(ctx)), AGENT_CONTEXT_CHAR_LIMIT * 2)

    def test_iteration_2_feedback_injection(self) -> None:
        state = AgentState.create("Write an async scraper. Do not use sync libraries.")
        state = state.with_analyzer(
            analyzer_result=None,
            analyzer_dict={"interpreted_goal": "Async scraper"},
            initial_score=55.0,
        )

        val_meta = ValidationMetadata(
            validator_version="1.0.0",
            validation_mode=ValidationMode.MOCK,
            model="none",
            latency_ms=10.0,
            critic_consistency_checked=True,
        )
        prompt_meta = PromptMetadata(
            prompt_length=30,
            word_count=8,
            sentence_count=1,
            detected_intent="CODE",
            intent_confidence=0.9,
            quality_score=65.0,
        )
        val_res = ValidationResult(
            decision=ValidatorDecision.FAIL,
            is_valid=False,
            safety_score=40.0,
            original_metadata=prompt_meta,
            optimized_metadata=prompt_meta,
            issues=[
                ValidationIssue(
                    code="LOST_NEGATIVE_CONSTRAINT",
                    severity=IssueSeverity.ERROR,
                    message="Removed 'do not use sync libraries' constraint",
                    field_or_scope="negative_constraints",
                    evidence="sync libraries",
                ),
                ValidationIssue(
                    code="CONTRADICTION",
                    severity=IssueSeverity.WARNING,
                    message="Introduced synchronous requests alongside aiohttp",
                    field_or_scope="logic",
                    evidence="requests.get",
                ),
            ],
            metadata=val_meta,
        )
        critic_res = CriticEvaluation(
            decision=CriticDecision.FAIL,
            overall_critique_score=50.0,
            intent_preservation=IntentPreservationResult(
                score=60.0,
                status=QualityStatus.FAIR,
                reason="Intent preserved with minor deviation",
            ),
            requirement_preservation=RequirementPreservationResult(
                score=40.0,
                status=QualityStatus.POOR,
                reason="Lost async constraint",
            ),
            clarity_change=MetricChange(original_score=50.0, optimized_score=65.0, delta=15.0, assessment="improved"),
            specificity_change=MetricChange(original_score=50.0, optimized_score=65.0, delta=15.0, assessment="improved"),
            ambiguity_change=MetricChange(original_score=50.0, optimized_score=65.0, delta=15.0, assessment="improved"),
            completeness_change=MetricChange(original_score=50.0, optimized_score=65.0, delta=15.0, assessment="improved"),
            lost_requirements=["asyncio execution only"],
            unsupported_assumptions=["Requires beautifulsoup4"],
            metadata=CriticMetadata(
                critic_version="1.0.0",
                analysis_mode=CriticAnalysisMode.MOCK,
                model="none",
                semantic_similarity=0.88,
                latency_ms=10.0,
            ),
        )

        iteration = AgentIteration(
            iteration_number=1,
            prompt_before=state.original_prompt,
            prompt_after="Candidate 1 without negative constraint",
            optimizer_result=OptimizerResultSummary(
                optimized_prompt="Candidate 1 without negative constraint",
                summary="Expanded scraper",
            ),
            critic_result=CriticResultSummary(
                decision="FAIL",
                overall_critique_score=50.0,
                lost_requirements=["asyncio execution only"],
            ),
            validator_result=val_res,
            score_before=55.0,
            score_after=65.0,
            decision="FAIL",
            latency_ms=45.0,
        )

        state = state.add_iteration(iteration, val_res, critic_res, new_score=65.0)

        # Build context for iteration 2
        ctx2 = build_iteration_context(state, OptimizationMode.BALANCED)

        # Verify feedback elements are present in recommendations and weaknesses
        rec_texts = [r["recommendation"] for r in ctx2["recommendations"]]
        weakness_texts = [w["weakness"] for w in ctx2["weaknesses"]]
        missing_texts = [m["item"] for m in ctx2["missing_information"]]

        # Check that LOST_NEGATIVE_CONSTRAINT reaches optimizer recommendations
        self.assertTrue(any("LOST_NEGATIVE_CONSTRAINT" in r for r in rec_texts))
        # Check that lost requirement from critic is in missing_information
        self.assertTrue(any("asyncio execution only" in m for m in missing_texts))
        # Check that context summary has guidance for iteration 2
        self.assertIn("Iteration 2", ctx2["context_summary"])


if __name__ == "__main__":
    unittest.main()
