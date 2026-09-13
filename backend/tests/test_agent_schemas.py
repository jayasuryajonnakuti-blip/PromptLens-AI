"""Unit tests for Agent Loop strict schemas and validation (Step 19)."""
from __future__ import annotations

import unittest
from pydantic import ValidationError

from app.agent.schemas import (
    AgentDecision,
    AgentIteration,
    AgentMetrics,
    AgentRequest,
    AgentResponse,
    AgentStateSummary,
    AgentStatus,
    AgentTerminationReason,
    CriticResultSummary,
    OptimizerResultSummary,
)
from app.optimizer.schemas import OptimizationMode
from app.validator.schemas import (
    PromptMetadata,
    ValidationMetadata,
    ValidationMode,
    ValidationResult,
    ValidatorDecision,
)


class TestAgentSchemas(unittest.TestCase):
    """Test suite for Step 19 Pydantic v2 strict schemas."""

    def test_enums(self) -> None:
        self.assertEqual(AgentStatus.COMPLETED, "COMPLETED")
        self.assertEqual(AgentStatus.FAILED, "FAILED")
        self.assertEqual(AgentStatus.NEEDS_REVIEW, "NEEDS_REVIEW")
        self.assertEqual(AgentStatus.MAX_ITERATIONS_REACHED, "MAX_ITERATIONS_REACHED")
        self.assertEqual(AgentStatus.UNAVAILABLE, "UNAVAILABLE")

        self.assertEqual(AgentTerminationReason.VALIDATED, "VALIDATED")
        self.assertEqual(AgentTerminationReason.VALIDATION_FAILED, "VALIDATION_FAILED")
        self.assertEqual(AgentTerminationReason.VALIDATION_REVIEW, "VALIDATION_REVIEW")
        self.assertEqual(AgentTerminationReason.MAX_ITERATIONS, "MAX_ITERATIONS")
        self.assertEqual(AgentTerminationReason.SERVICE_UNAVAILABLE, "SERVICE_UNAVAILABLE")
        self.assertEqual(AgentTerminationReason.ERROR, "ERROR")

    def test_valid_request(self) -> None:
        req = AgentRequest(
            prompt="Write a concise python CLI tool",
            mode=OptimizationMode.BALANCED,
            max_iterations=3,
        )
        self.assertEqual(req.prompt, "Write a concise python CLI tool")
        self.assertEqual(req.mode, OptimizationMode.BALANCED)
        self.assertEqual(req.max_iterations, 3)

    def test_request_blank_prompt_raises(self) -> None:
        with self.assertRaises(ValidationError):
            AgentRequest(prompt="")
        with self.assertRaises(ValidationError):
            AgentRequest(prompt="   \n\t  ")

    def test_request_max_iterations_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            AgentRequest(prompt="Valid prompt", max_iterations=0)
        with self.assertRaises(ValidationError):
            AgentRequest(prompt="Valid prompt", max_iterations=4)

    def test_request_extra_fields_forbidden(self) -> None:
        with self.assertRaises(ValidationError):
            AgentRequest(prompt="Valid prompt", unexpected_param="injected")  # type: ignore[call-arg]

    def test_valid_response(self) -> None:
        resp = AgentResponse(
            original_prompt="Orig",
            final_prompt="Opt",
            status=AgentStatus.COMPLETED,
            termination_reason=AgentTerminationReason.VALIDATED,
            iteration_count=1,
            is_validated=True,
            iterations=[],
            final_validation=None,
            final_critic_result=None,
            final_score=85.0,
            metrics=AgentMetrics(
                total_latency_ms=120.0,
                total_iterations=1,
                initial_score=70.0,
                final_score=85.0,
                score_delta=15.0,
                prompt_length_before=4,
                prompt_length_after=3,
                expansion_ratio=0.75,
                semantic_similarity=0.92,
            ),
        )
        self.assertEqual(resp.status, AgentStatus.COMPLETED)
        self.assertTrue(resp.is_validated)
        self.assertEqual(len(resp.disclaimers), 5)

    def test_response_extra_fields_forbidden(self) -> None:
        with self.assertRaises(ValidationError):
            AgentResponse(
                original_prompt="Orig",
                final_prompt="Opt",
                status=AgentStatus.COMPLETED,
                termination_reason=AgentTerminationReason.VALIDATED,
                iteration_count=1,
                is_validated=True,
                metrics=AgentMetrics(),
                unknown_field="fail",  # type: ignore[call-arg]
            )

    def test_agent_decision_schema(self) -> None:
        dec = AgentDecision(
            should_continue=False,
            target_status=AgentStatus.COMPLETED,
            termination_reason=AgentTerminationReason.VALIDATED,
            explanation="Validated",
            is_non_improving=False,
        )
        self.assertFalse(dec.should_continue)
        self.assertEqual(dec.target_status, AgentStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
