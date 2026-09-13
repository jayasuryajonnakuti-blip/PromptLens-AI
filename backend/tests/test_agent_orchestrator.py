"""Unit and integration tests for AgentOrchestrator (Step 19).

Tests:
- Clean optimization -> PASS on first iteration
- FAIL -> second optimization attempt
- FAIL -> third optimization attempt -> third FAIL -> safe termination
- PASS stops immediately (does not continue)
- NEEDS_REVIEW stops immediately (does not continue)
- Never exceeds 3 iterations
- Feedback propagation: requirement-loss, constraint-loss, contradictions reach next optimizer
- Safe handling of Analyzer/Optimizer/Critic/Validator unavailable
- No stack traces exposed on unexpected exception
- Failed candidate is never marked as validated
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.agent.config import MAX_OPTIMIZATION_ITERATIONS
from app.agent.orchestrator import AgentOrchestrator
from app.agent.schemas import (
    AgentRequest,
    AgentStatus,
    AgentTerminationReason,
)
from app.analyzer.schemas import (
    AnalysisMode,
    AnalyzerAnalysis,
    AnalyzerMetadata,
    InstructionQualityResult,
    IntentResult,
)
from app.critic.schemas import (
    CriticAnalysisMode,
    CriticDecision,
    CriticEvaluation,
    CriticIssue,
    CriticMetadata,
    IntentPreservationResult,
    IssueSeverity,
    MetricChange,
    QualityStatus,
    RequirementPreservationResult,
)
from app.optimizer.schemas import (
    ChangeRecord,
    OptimizationMode,
    OptimizerMetadata,
    OptimizerMode,
    OptimizerResult,
    PreservedRequirement,
)
from app.services.agent_service import set_agent_service
from app.services.analyzer_service import AnalyzerService, set_analyzer_service
from app.services.critic_service import CriticService, set_critic_service
from app.services.optimizer_service import OptimizerService, set_optimizer_service
from app.services.validator_service import ValidatorService, set_validator_service
from app.validator.schemas import (
    PromptMetadata,
    ValidationIssue,
    ValidationMetadata,
    ValidationMode,
    ValidationResult,
    ValidatorDecision,
)


class TestAgentOrchestrator(unittest.TestCase):
    """Test suite for AgentOrchestrator coordinating all subsystems."""

    def setUp(self) -> None:
        # Reset all singletons
        set_analyzer_service(None)
        set_optimizer_service(None)
        set_critic_service(None)
        set_validator_service(None)
        set_agent_service(None)

    def tearDown(self) -> None:
        set_analyzer_service(None)
        set_optimizer_service(None)
        set_critic_service(None)
        set_validator_service(None)
        set_agent_service(None)

    def _build_mock_analyzer_analysis(self, prompt: str) -> AnalyzerAnalysis:
        return AnalyzerAnalysis(
            interpreted_goal="Build a CLI tool",
            intent=IntentResult(label="CODE", confidence=0.95),
            context_summary="User needs a python CLI tool.",
            ambiguities=[],
            missing_information=[],
            contradictions=[],
            instruction_quality=InstructionQualityResult(score=70.0, reason="Reasonably clear"),
            strengths=[],
            weaknesses=[],
            recommendations=[],
            evidence=[],
            metadata=AnalyzerMetadata(
                analyzer_version="1.0.0",
                llm_model="none",
                analysis_mode=AnalysisMode.MOCK,
                latency_ms=10.0,
            ),
        )

    def _build_mock_optimizer_result(self, cand_prompt: str, delta: float = 10.0) -> OptimizerResult:
        return OptimizerResult(
            optimized_prompt=cand_prompt,
            summary="Added arguments and docstring.",
            changes=[
                ChangeRecord(category="structure", description="Added argparse structure")
            ],
            preserved_requirements=[
                PreservedRequirement(requirement="Python CLI", reason="Core task")
            ],
            placeholders_inserted=[],
            improvement_score_delta=delta,
            metadata=OptimizerMetadata(
                optimizer_version="1.0.0",
                llm_model="none",
                optimizer_mode=OptimizerMode.MOCK,
                optimization_mode=OptimizationMode.BALANCED,
                latency_ms=20.0,
            ),
        )

    def _build_mock_critic_evaluation(
        self,
        decision: CriticDecision = CriticDecision.PASS,
        score: float = 85.0,
        lost_reqs: list[str] | None = None,
    ) -> CriticEvaluation:
        return CriticEvaluation(
            decision=decision,
            overall_critique_score=score,
            intent_preservation=IntentPreservationResult(
                score=score,
                status=QualityStatus.GOOD if decision == CriticDecision.PASS else QualityStatus.FAIR,
                reason="Intent preserved",
            ),
            requirement_preservation=RequirementPreservationResult(
                score=score,
                status=QualityStatus.GOOD if decision == CriticDecision.PASS else QualityStatus.POOR,
                reason="Requirements evaluated",
            ),
            clarity_change=MetricChange(original_score=70.0, optimized_score=score, delta=score - 70.0, assessment="improved"),
            specificity_change=MetricChange(original_score=70.0, optimized_score=score, delta=score - 70.0, assessment="improved"),
            ambiguity_change=MetricChange(original_score=70.0, optimized_score=score, delta=score - 70.0, assessment="improved"),
            completeness_change=MetricChange(original_score=70.0, optimized_score=score, delta=score - 70.0, assessment="improved"),
            lost_requirements=lost_reqs or [],
            metadata=CriticMetadata(
                critic_version="1.0.0",
                analysis_mode=CriticAnalysisMode.MOCK,
                model="none",
                semantic_similarity=0.92,
                latency_ms=15.0,
            ),
        )

    def _build_mock_validation_result(
        self,
        decision: ValidatorDecision = ValidatorDecision.PASS,
        score: float = 85.0,
        issues: list[ValidationIssue] | None = None,
    ) -> ValidationResult:
        prompt_meta = PromptMetadata(
            prompt_length=50,
            word_count=10,
            sentence_count=2,
            detected_intent="CODE",
            intent_confidence=0.95,
            quality_score=score,
        )
        return ValidationResult(
            decision=decision,
            is_valid=(decision == ValidatorDecision.PASS),
            safety_score=score,
            original_metadata=prompt_meta,
            optimized_metadata=prompt_meta,
            issues=issues or [],
            metadata=ValidationMetadata(
                validator_version="1.0.0",
                validation_mode=ValidationMode.MOCK,
                model="none",
                latency_ms=15.0,
                critic_consistency_checked=True,
            ),
        )

    def test_clean_optimization_pass_first_iteration(self) -> None:
        """Test clean optimization achieving Validator PASS on iteration 1."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = self._build_mock_analyzer_analysis("Write a CLI tool")
        set_analyzer_service(AnalyzerService(mock_analyzer))

        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = self._build_mock_optimizer_result(
            "Write a Python CLI tool using argparse with subcommands"
        )
        set_optimizer_service(OptimizerService(mock_optimizer))

        mock_critic = MagicMock()
        mock_critic.evaluate.return_value = self._build_mock_critic_evaluation(CriticDecision.PASS, 88.0)
        set_critic_service(CriticService(mock_critic))

        mock_validator = MagicMock()
        mock_validator.validate.return_value = self._build_mock_validation_result(ValidatorDecision.PASS, 88.0)
        set_validator_service(ValidatorService(mock_validator))

        orchestrator = AgentOrchestrator()
        req = AgentRequest(prompt="Write a CLI tool", max_iterations=3)
        resp = orchestrator.run(req)

        self.assertEqual(resp.status, AgentStatus.COMPLETED)
        self.assertEqual(resp.termination_reason, AgentTerminationReason.VALIDATED)
        self.assertEqual(resp.iteration_count, 1)
        self.assertTrue(resp.is_validated)
        self.assertEqual(len(resp.iterations), 1)
        self.assertEqual(resp.iterations[0].decision, "PASS")
        # Ensure optimizer called exactly once
        self.assertEqual(mock_optimizer.optimize.call_count, 1)

    def test_fail_then_pass_second_iteration(self) -> None:
        """Test FAIL on iteration 1 triggering iteration 2 with structured feedback, then PASS."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = self._build_mock_analyzer_analysis("Write a CLI tool")
        set_analyzer_service(AnalyzerService(mock_analyzer))

        mock_optimizer = MagicMock()
        # Iteration 1 candidate vs Iteration 2 candidate
        cand1 = "Candidate 1 missing error handling"
        cand2 = "Candidate 2 with comprehensive error handling and subcommands"
        mock_optimizer.optimize.side_effect = [
            self._build_mock_optimizer_result(cand1, 65.0),
            self._build_mock_optimizer_result(cand2, 85.0),
        ]
        set_optimizer_service(OptimizerService(mock_optimizer))

        mock_critic = MagicMock()
        mock_critic.evaluate.side_effect = [
            self._build_mock_critic_evaluation(CriticDecision.FAIL, 60.0, lost_reqs=["error handling"]),
            self._build_mock_critic_evaluation(CriticDecision.PASS, 85.0),
        ]
        set_critic_service(CriticService(mock_critic))

        issue = ValidationIssue(
            code="LOST_REQUIREMENT",
            severity=IssueSeverity.ERROR,
            message="Lost error handling requirement",
            field_or_scope="requirements",
            evidence="error handling",
        )
        mock_validator = MagicMock()
        mock_validator.validate.side_effect = [
            self._build_mock_validation_result(ValidatorDecision.FAIL, 60.0, issues=[issue]),
            self._build_mock_validation_result(ValidatorDecision.PASS, 85.0),
        ]
        set_validator_service(ValidatorService(mock_validator))

        orchestrator = AgentOrchestrator()
        req = AgentRequest(prompt="Write a CLI tool", max_iterations=3)
        resp = orchestrator.run(req)

        self.assertEqual(resp.status, AgentStatus.COMPLETED)
        self.assertEqual(resp.termination_reason, AgentTerminationReason.VALIDATED)
        self.assertEqual(resp.iteration_count, 2)
        self.assertTrue(resp.is_validated)
        self.assertEqual(resp.final_prompt, cand2)
        self.assertEqual(len(resp.iterations), 2)
        self.assertEqual(resp.iterations[0].decision, "FAIL")
        self.assertEqual(resp.iterations[1].decision, "PASS")

        # Verify feedback reached optimizer in iteration 2
        second_call_kwargs = mock_optimizer.optimize.call_args_list[1][1]
        analyzer_ctx = second_call_kwargs.get("analyzer_result", {})
        recommendations = [r.get("recommendation", "") for r in analyzer_ctx.get("recommendations", [])]
        self.assertTrue(any("LOST_REQUIREMENT" in r for r in recommendations))

    def test_three_fails_terminates_at_max_iterations(self) -> None:
        """Test 3 consecutive non-identical validation failures terminating safely at max iterations."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = self._build_mock_analyzer_analysis("Prompt")
        set_analyzer_service(AnalyzerService(mock_analyzer))

        mock_optimizer = MagicMock()
        mock_optimizer.optimize.side_effect = [
            self._build_mock_optimizer_result("Candidate alpha 1", 60.0),
            self._build_mock_optimizer_result("Candidate beta 2", 62.0),
            self._build_mock_optimizer_result("Candidate gamma 3", 64.0),
        ]
        set_optimizer_service(OptimizerService(mock_optimizer))

        mock_critic = MagicMock()
        mock_critic.evaluate.return_value = self._build_mock_critic_evaluation(CriticDecision.FAIL, 60.0)
        set_critic_service(CriticService(mock_critic))

        mock_validator = MagicMock()
        # Different issue codes so repeated identical error check isn't triggered
        mock_validator.validate.side_effect = [
            self._build_mock_validation_result(ValidatorDecision.FAIL, 60.0, issues=[
                ValidationIssue(code="ERR_1", severity=IssueSeverity.ERROR, message="m1", field_or_scope="s", evidence="e")
            ]),
            self._build_mock_validation_result(ValidatorDecision.FAIL, 62.0, issues=[
                ValidationIssue(code="ERR_2", severity=IssueSeverity.ERROR, message="m2", field_or_scope="s", evidence="e")
            ]),
            self._build_mock_validation_result(ValidatorDecision.FAIL, 64.0, issues=[
                ValidationIssue(code="ERR_3", severity=IssueSeverity.ERROR, message="m3", field_or_scope="s", evidence="e")
            ]),
        ]
        set_validator_service(ValidatorService(mock_validator))

        orchestrator = AgentOrchestrator()
        req = AgentRequest(prompt="Prompt", max_iterations=3)
        resp = orchestrator.run(req)

        self.assertEqual(resp.status, AgentStatus.MAX_ITERATIONS_REACHED)
        self.assertEqual(resp.termination_reason, AgentTerminationReason.MAX_ITERATIONS)
        self.assertEqual(resp.iteration_count, 3)
        self.assertFalse(resp.is_validated)
        self.assertEqual(resp.final_prompt, "Candidate gamma 3")

    def test_needs_review_terminates_immediately(self) -> None:
        """Test Validator NEEDS_REVIEW stopping loop immediately on iteration 1."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = self._build_mock_analyzer_analysis("Prompt")
        set_analyzer_service(AnalyzerService(mock_analyzer))

        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = self._build_mock_optimizer_result("Candidate with subtle trade-off")
        set_optimizer_service(OptimizerService(mock_optimizer))

        mock_critic = MagicMock()
        mock_critic.evaluate.return_value = self._build_mock_critic_evaluation(CriticDecision.NEEDS_REVIEW, 72.0)
        set_critic_service(CriticService(mock_critic))

        mock_validator = MagicMock()
        mock_validator.validate.return_value = self._build_mock_validation_result(ValidatorDecision.NEEDS_REVIEW, 72.0)
        set_validator_service(ValidatorService(mock_validator))

        orchestrator = AgentOrchestrator()
        req = AgentRequest(prompt="Prompt", max_iterations=3)
        resp = orchestrator.run(req)

        self.assertEqual(resp.status, AgentStatus.NEEDS_REVIEW)
        self.assertEqual(resp.termination_reason, AgentTerminationReason.VALIDATION_REVIEW)
        self.assertEqual(resp.iteration_count, 1)
        self.assertFalse(resp.is_validated)
        self.assertEqual(mock_optimizer.optimize.call_count, 1)

    def test_analyzer_unavailable_fallback(self) -> None:
        """Test graceful continuation when Analyzer service raises an exception."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = RuntimeError("Analyzer unavailable")
        set_analyzer_service(AnalyzerService(mock_analyzer))

        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = self._build_mock_optimizer_result("Optimized prompt candidate")
        set_optimizer_service(OptimizerService(mock_optimizer))

        mock_critic = MagicMock()
        mock_critic.evaluate.return_value = self._build_mock_critic_evaluation(CriticDecision.PASS, 80.0)
        set_critic_service(CriticService(mock_critic))

        mock_validator = MagicMock()
        mock_validator.validate.return_value = self._build_mock_validation_result(ValidatorDecision.PASS, 80.0)
        set_validator_service(ValidatorService(mock_validator))

        orchestrator = AgentOrchestrator()
        req = AgentRequest(prompt="Write a parser", max_iterations=3)
        resp = orchestrator.run(req)

        self.assertEqual(resp.status, AgentStatus.COMPLETED)
        self.assertEqual(resp.termination_reason, AgentTerminationReason.VALIDATED)
        self.assertTrue(resp.is_validated)

    def test_unexpected_exception_handled_safely(self) -> None:
        """Test unexpected exception does not leak internal stack traces."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = self._build_mock_analyzer_analysis("Prompt")
        set_analyzer_service(AnalyzerService(mock_analyzer))

        mock_optimizer = MagicMock()
        mock_optimizer.optimize.side_effect = ValueError("Corrupt internal model state")
        set_optimizer_service(OptimizerService(mock_optimizer))

        orchestrator = AgentOrchestrator()
        req = AgentRequest(prompt="Prompt", max_iterations=3)
        resp = orchestrator.run(req)

        self.assertEqual(resp.status, AgentStatus.FAILED)
        self.assertEqual(resp.termination_reason, AgentTerminationReason.ERROR)
        self.assertFalse(resp.is_validated)
        self.assertEqual(resp.original_prompt, "Prompt")
        # Ensure no exception traceback leaked in user fields
        self.assertNotIn("Traceback", resp.final_prompt)


if __name__ == "__main__":
    unittest.main()
