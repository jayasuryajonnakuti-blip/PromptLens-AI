"""Tests for PromptValidator core class (Step 18).

Uses ValidatorMockProvider injected via set_llm_service().
Always resets state in tearDown().
"""
import json
import unittest

from app.llm.config import LLMConfig
from app.llm.model import MockLLMProvider
from app.llm.schemas import LLMHealthStatus
from app.services.llm_service import LLMService, set_llm_service
from app.services.validator_service import set_validator_service
from app.validator.schemas import (
    CriticResultInput,
    ValidationMode,
    ValidationResult,
    ValidatorDecision,
)
from app.validator.validator import PromptValidator


class ValidatorMockProvider(MockLLMProvider):
    """Deterministic mock provider returning valid Validator JSON."""

    def __init__(
        self,
        mode: str = "valid",
        decision: str = "PASS",
        safety_score: float = 90.0,
    ) -> None:
        super().__init__(mode=mode)
        self.decision = decision
        self.safety_score = safety_score

    def _valid_payload(self) -> str:
        return json.dumps({
            "decision": self.decision,
            "safety_score": self.safety_score,
            "justification": "Optimization preserves intent and constraints while improving structure.",
            "issues": [],
            "passed_checks": [
                "semantic_intent_preservation",
                "constraint_consistency",
                "format_validity",
            ],
            "failed_checks": [],
        })


def _make_mock_service(
    mode: str = "valid", decision: str = "PASS", score: float = 90.0
) -> LLMService:
    config = LLMConfig(enabled=True, provider="mock", max_retries=2)
    provider = ValidatorMockProvider(mode=mode, decision=decision, safety_score=score)
    return LLMService(config=config, provider=provider)


# ---------------------------------------------------------------------------
# UNAVAILABLE mode tests (Deterministic validation)
# ---------------------------------------------------------------------------
class TestPromptValidatorUnavailableMode(unittest.TestCase):
    def setUp(self):
        config = LLMConfig(enabled=False, provider="transformers")
        provider = MockLLMProvider(health_status=LLMHealthStatus.LLM_DISABLED)
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

    def tearDown(self):
        set_llm_service(None)
        set_validator_service(None)

    def test_valid_optimization_returns_pass(self):
        validator = PromptValidator()
        orig = "Write a Python script to sort items in an array."
        opt = "Write a clean Python function to sort items in an array in ascending order."
        res = validator.validate(orig, opt)

        self.assertIsInstance(res, ValidationResult)
        self.assertEqual(res.decision, ValidatorDecision.PASS)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.metadata.validation_mode, ValidationMode.UNAVAILABLE)
        self.assertGreaterEqual(res.safety_score, 80.0)

    def test_empty_prompts_raise_value_error(self):
        validator = PromptValidator()
        with self.assertRaises(ValueError):
            validator.validate("   ", "Valid")
        with self.assertRaises(ValueError):
            validator.validate("Valid", "   ")

    def test_lost_negative_constraint_fails(self):
        validator = PromptValidator()
        orig = "Write an API client. Avoid using synchronous network requests."
        opt = "Write an API client using standard urllib."
        res = validator.validate(orig, opt)

        self.assertEqual(res.decision, ValidatorDecision.FAIL)
        self.assertFalse(res.is_valid)
        self.assertTrue(any(i.code == "LOST_NEGATIVE_CONSTRAINT" for i in res.issues))

    def test_contradiction_fails(self):
        validator = PromptValidator()
        orig = "Return data."
        opt = "Return JSON only. Include a detailed conversational explanation describing your thought process."
        res = validator.validate(orig, opt)

        self.assertEqual(res.decision, ValidatorDecision.FAIL)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("CONTRADICTORY" in i.code for i in res.issues))

    def test_critic_fail_propagates_to_validator_fail(self):
        validator = PromptValidator()
        orig = "Explain recursion."
        opt = "Explain recursion with a simple example."
        critic_in = CriticResultInput(
            decision="FAIL",
            overall_critique_score=42.0,
            issues=[{"type": "fatal", "description": "Critical requirement lost"}],
            lost_requirements=["Target audience"],
            introduced_requirements=[],
            unsupported_assumptions=[],
        )
        res = validator.validate(orig, opt, critic_result=critic_in)

        self.assertEqual(res.decision, ValidatorDecision.FAIL)
        self.assertFalse(res.is_valid)
        self.assertTrue(any(i.code == "CRITIC_FAIL_UNRESOLVED" for i in res.issues))

    def test_critic_needs_review_handled(self):
        validator = PromptValidator()
        orig = "Write a function."
        opt = "Write a function."
        critic_in = CriticResultInput(
            decision="NEEDS_REVIEW",
            overall_critique_score=68.0,
            issues=[],
            lost_requirements=[],
            introduced_requirements=[],
            unsupported_assumptions=[],
        )
        res = validator.validate(orig, opt, critic_result=critic_in)
        self.assertEqual(res.decision, ValidatorDecision.NEEDS_REVIEW)
        self.assertFalse(res.is_valid)

    def test_validator_does_not_modify_optimized_prompt(self):
        validator = PromptValidator()
        orig = "Write code."
        opt = "Write clean, tested Python code."
        res = validator.validate(orig, opt)
        # Verify metadata lengths accurately match inputs
        self.assertEqual(res.original_metadata.prompt_length, len(orig))
        self.assertEqual(res.optimized_metadata.prompt_length, len(opt))


# ---------------------------------------------------------------------------
# MOCK mode tests (Local LLM simulation)
# ---------------------------------------------------------------------------
class TestPromptValidatorMockMode(unittest.TestCase):
    def setUp(self):
        set_llm_service(_make_mock_service("valid", decision="PASS", score=92.0))

    def tearDown(self):
        set_llm_service(None)
        set_validator_service(None)

    def test_mock_mode_validation_pass(self):
        validator = PromptValidator()
        orig = "You are a Python engineer. Write a function to calculate factorial."
        opt = "You are a Python engineer. Write a recursive function `factorial(n: int) -> int` with input validation."
        res = validator.validate(orig, opt)

        self.assertEqual(res.metadata.validation_mode, ValidationMode.MOCK)
        self.assertEqual(res.decision, ValidatorDecision.PASS)
        self.assertTrue(res.is_valid)
        self.assertGreaterEqual(res.safety_score, 85.0)

    def test_mock_mode_deterministic_fail_overrides_llm_pass(self):
        # Even if mock LLM says PASS, a fatal contradiction MUST result in Validator FAIL
        validator = PromptValidator()
        orig = "Return JSON."
        opt = "Return JSON only. Include a detailed conversational explanation describing your thought process."
        res = validator.validate(orig, opt)

        self.assertEqual(res.decision, ValidatorDecision.FAIL)
        self.assertFalse(res.is_valid)


# ---------------------------------------------------------------------------
# Fallback & Error Recovery tests
# ---------------------------------------------------------------------------
class TestPromptValidatorFallback(unittest.TestCase):
    def tearDown(self):
        set_llm_service(None)
        set_validator_service(None)

    def test_malformed_json_falls_back_to_unavailable(self):
        config = LLMConfig(enabled=True, provider="mock", max_retries=0)
        provider = MockLLMProvider(mode="malformed_json")
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

        validator = PromptValidator()
        res = validator.validate("Write a script.", "Write a Python script.")
        self.assertEqual(res.metadata.validation_mode, ValidationMode.UNAVAILABLE)
        self.assertIn(res.decision, {ValidatorDecision.PASS, ValidatorDecision.NEEDS_REVIEW})

    def test_error_mode_falls_back_to_unavailable(self):
        config = LLMConfig(enabled=True, provider="mock", max_retries=0)
        provider = MockLLMProvider(mode="error")
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

        validator = PromptValidator()
        res = validator.validate("Write a script.", "Write a Python script.")
        self.assertEqual(res.metadata.validation_mode, ValidationMode.UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
