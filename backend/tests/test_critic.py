"""Tests for PromptCritic core class (Step 17).

Uses CriticMockProvider injected via set_llm_service().
Always resets state in tearDown().
"""
import json
import unittest

from app.critic.critic import PromptCritic
from app.critic.schemas import (
    CriticAnalysisMode,
    CriticDecision,
    CriticEvaluation,
)
from app.llm.config import LLMConfig
from app.llm.model import MockLLMProvider
from app.llm.schemas import LLMHealthStatus
from app.services.critic_service import set_critic_service
from app.services.llm_service import LLMService, set_llm_service


class CriticMockProvider(MockLLMProvider):
    """Deterministic mock provider returning valid Critic JSON."""

    def __init__(
        self,
        mode: str = "valid",
        decision: str = "PASS",
        overall_score: float = 84.0,
    ) -> None:
        super().__init__(mode=mode)
        self.decision = decision
        self.overall_score = overall_score

    def _valid_payload(self) -> str:
        return json.dumps({
            "decision": self.decision,
            "overall_critique_score": self.overall_score,
            "intent_preservation": {
                "score": 92.0,
                "status": "EXCELLENT",
                "reason": "Original objective to build an auth endpoint was preserved.",
            },
            "requirement_preservation": {
                "score": 88.0,
                "status": "STRONG",
                "reason": "All negative constraints and format requirements maintained.",
            },
            "clarity_assessment": "Clear structured bullet points improve readability.",
            "specificity_assessment": "Replaced generic instructions with explicit token parameters.",
            "ambiguity_assessment": "Ambiguity reduced by specifying HTTP status codes.",
            "completeness_assessment": "Added explicit error handling criteria.",
            "issues": [
                {
                    "type": "minor_assumption",
                    "severity": "INFO",
                    "description": "Recommended FastAPI as preferred framework.",
                    "evidence": "Framework: [WEB_FRAMEWORK] (FastAPI recommended)",
                }
            ],
            "preserved_requirements": [
                "Python development role",
                "JWT authentication mechanism",
                "Avoid synchronous libraries",
            ],
            "lost_requirements": [],
            "introduced_requirements": [],
            "unsupported_assumptions": [],
            "strengths": [
                "Well-structured numbered requirements",
                "Explicit error handling",
            ],
            "weaknesses": [],
            "recommendations": [
                "Proceed with the optimization.",
            ],
        })


def _make_mock_service(
    mode: str = "valid", decision: str = "PASS", score: float = 84.0
) -> LLMService:
    config = LLMConfig(enabled=True, provider="mock", max_retries=2)
    provider = CriticMockProvider(mode=mode, decision=decision, overall_score=score)
    return LLMService(config=config, provider=provider)


# ---------------------------------------------------------------------------
# UNAVAILABLE mode tests
# ---------------------------------------------------------------------------
class TestPromptCriticUnavailableMode(unittest.TestCase):
    def setUp(self):
        config = LLMConfig(enabled=False, provider="transformers")
        provider = MockLLMProvider(health_status=LLMHealthStatus.LLM_DISABLED)
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

    def tearDown(self):
        set_llm_service(None)
        set_critic_service(None)

    def test_returns_critic_evaluation(self):
        critic = PromptCritic()
        res = critic.evaluate("Write a Python script.", "Write a Python script to sort items.")
        self.assertIsInstance(res, CriticEvaluation)
        self.assertEqual(res.metadata.analysis_mode, CriticAnalysisMode.UNAVAILABLE)

    def test_decision_pass_on_clean_improvement(self):
        critic = PromptCritic()
        orig = "Write a function to add numbers in Python."
        opt = "Write a Python function `add(a: int, b: int) -> int` that returns the sum of two numbers."
        res = critic.evaluate(orig, opt)
        self.assertIn(res.decision, {CriticDecision.PASS, CriticDecision.NEEDS_REVIEW})
        self.assertGreaterEqual(res.overall_critique_score, 50.0)

    def test_decision_fail_on_lost_output_format(self):
        critic = PromptCritic()
        orig = "Extract names from this text and return JSON only."
        opt = "Extract names from this text in paragraph form."
        res = critic.evaluate(orig, opt)
        self.assertEqual(res.decision, CriticDecision.FAIL)
        self.assertTrue(any(i.type == "lost_output_format" for i in res.issues))

    def test_blank_inputs_raise_value_error(self):
        critic = PromptCritic()
        with self.assertRaises(ValueError):
            critic.evaluate("  ", "Valid")
        with self.assertRaises(ValueError):
            critic.evaluate("Valid", "  ")


# ---------------------------------------------------------------------------
# MOCK mode tests
# ---------------------------------------------------------------------------
class TestPromptCriticMockMode(unittest.TestCase):
    def setUp(self):
        set_llm_service(_make_mock_service("valid", decision="PASS", score=86.0))

    def tearDown(self):
        set_llm_service(None)
        set_critic_service(None)

    def test_mock_mode_evaluation_pass(self):
        critic = PromptCritic()
        orig = "You are a senior Python developer. Write a REST API endpoint for user login using JWT tokens."
        opt = (
            "You are a senior Python developer. Write a REST API endpoint for user login using JWT tokens.\n"
            "Requirements:\n"
            "- Use [WEB_FRAMEWORK]\n"
            "- Return JSON response with access_token and token_type\n"
            "- Avoid synchronous libraries"
        )
        res = critic.evaluate(orig, opt)
        self.assertEqual(res.metadata.analysis_mode, CriticAnalysisMode.MOCK)
        self.assertEqual(res.decision, CriticDecision.PASS)
        self.assertGreaterEqual(res.overall_critique_score, 80.0)
        self.assertGreater(len(res.preserved_requirements), 0)
        self.assertGreater(len(res.strengths), 0)

    def test_mock_mode_deterministic_fail_overrides_llm_pass(self):
        # Even if mock says PASS, a fatal format loss MUST result in FAIL
        critic = PromptCritic()
        orig = "Return JSON only."
        opt = "Return conversational text."
        res = critic.evaluate(orig, opt)
        self.assertEqual(res.decision, CriticDecision.FAIL)


# ---------------------------------------------------------------------------
# Fallback & Retry tests
# ---------------------------------------------------------------------------
class TestPromptCriticFallback(unittest.TestCase):
    def tearDown(self):
        set_llm_service(None)
        set_critic_service(None)

    def test_malformed_json_falls_back_to_unavailable(self):
        config = LLMConfig(enabled=True, provider="mock", max_retries=0)
        provider = MockLLMProvider(mode="malformed_json")
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

        critic = PromptCritic()
        res = critic.evaluate("Write Python code.", "Write clean Python code.")
        self.assertEqual(res.metadata.analysis_mode, CriticAnalysisMode.UNAVAILABLE)

    def test_error_mode_falls_back_to_unavailable(self):
        config = LLMConfig(enabled=True, provider="mock", max_retries=0)
        provider = MockLLMProvider(mode="error")
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

        critic = PromptCritic()
        res = critic.evaluate("Write Python code.", "Write clean Python code.")
        self.assertEqual(res.metadata.analysis_mode, CriticAnalysisMode.UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
