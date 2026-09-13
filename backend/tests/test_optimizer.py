"""Tests for PromptOptimizer core class (Step 16).

All tests use an OptimizerMockProvider injected via set_llm_service().
No LLM weights downloaded; no commercial APIs called.
Always call set_optimizer_service(None) + set_llm_service(None) in tearDown().
"""
import json
import unittest

from app.llm.model import MockLLMProvider
from app.llm.schemas import LLMHealthStatus
from app.optimizer.optimizer import PromptOptimizer
from app.optimizer.schemas import (
    OptimizationMode,
    OptimizerMode,
    OptimizerResult,
)
from app.services.llm_service import LLMService, set_llm_service
from app.services.optimizer_service import set_optimizer_service


# ---------------------------------------------------------------------------
# Optimizer-compatible mock provider
# ---------------------------------------------------------------------------
class OptimizerMockProvider(MockLLMProvider):
    """Mock that returns valid optimizer JSON for all modes."""

    def _valid_payload(self) -> str:
        return json.dumps({
            "optimized_prompt": (
                "You are a senior Python developer specializing in API design. "
                "Write a REST API endpoint for user login using JWT tokens. "
                "Requirements:\n"
                "- Use [WEB_FRAMEWORK] (e.g. FastAPI or Flask)\n"
                "- Implement [AUTH_LIBRARY] for JWT handling\n"
                "- Return a JSON response with access_token and token_type fields\n"
                "- Handle invalid credentials with a 401 error\n"
                "- Avoid synchronous I/O libraries"
            ),
            "summary": (
                "Added explicit framework and library placeholders. "
                "Structured requirements as a bullet list. "
                "Added error handling requirement."
            ),
            "changes": [
                {"category": "specificity", "description": "Added [WEB_FRAMEWORK] placeholder for missing framework."},
                {"category": "structure", "description": "Converted prose to bullet-point requirements."},
                {"category": "completeness", "description": "Added error handling (401) requirement."},
                {"category": "placeholder_inserted", "description": "Inserted [AUTH_LIBRARY] for missing JWT library."},
            ],
            "preserved_requirements": [
                {"requirement": "You are a senior Python developer.", "reason": "Role definition preserved verbatim."},
                {"requirement": "Avoid synchronous libraries.", "reason": "Explicit constraint kept unchanged."},
                {"requirement": "Return JSON only.", "reason": "Output format constraint preserved."},
            ],
            "placeholders_inserted": ["[WEB_FRAMEWORK]", "[AUTH_LIBRARY]"],
            "improvement_score_delta": 18.5,
        })


def _make_mock_svc(mode: str = "valid") -> LLMService:
    from app.llm.config import LLMConfig
    config = LLMConfig(enabled=True, provider="mock", max_retries=2)
    provider = OptimizerMockProvider(mode=mode)
    return LLMService(config=config, provider=provider)


# ---------------------------------------------------------------------------
# UNAVAILABLE mode tests
# ---------------------------------------------------------------------------
class TestPromptOptimizerUnavailableMode(unittest.TestCase):
    def setUp(self):
        from app.llm.config import LLMConfig
        config = LLMConfig(enabled=False, provider="transformers")
        provider = MockLLMProvider(health_status=LLMHealthStatus.LLM_DISABLED)
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

    def tearDown(self):
        set_llm_service(None)
        set_optimizer_service(None)

    def test_returns_optimizer_result(self):
        opt = PromptOptimizer()
        result = opt.optimize("Write a Python function.", OptimizationMode.BALANCED)
        self.assertIsInstance(result, OptimizerResult)

    def test_mode_is_unavailable(self):
        opt = PromptOptimizer()
        result = opt.optimize("Write a Python function.", OptimizationMode.BALANCED)
        self.assertEqual(result.metadata.optimizer_mode, OptimizerMode.UNAVAILABLE)

    def test_optimized_prompt_equals_original(self):
        """UNAVAILABLE must return the original prompt unchanged."""
        original = "Write a Python function to sort a list."
        opt = PromptOptimizer()
        result = opt.optimize(original, OptimizationMode.BALANCED)
        self.assertEqual(result.optimized_prompt, original)

    def test_improvement_delta_zero_when_unavailable(self):
        opt = PromptOptimizer()
        result = opt.optimize("Write a test.", OptimizationMode.BALANCED)
        self.assertEqual(result.improvement_score_delta, 0.0)

    def test_llm_model_is_none(self):
        opt = PromptOptimizer()
        result = opt.optimize("Write a test.", OptimizationMode.EXPERT)
        self.assertEqual(result.metadata.llm_model, "none")

    def test_optimization_mode_stored_correctly(self):
        for mode in OptimizationMode:
            opt = PromptOptimizer()
            result = opt.optimize("Test.", mode)
            self.assertEqual(result.metadata.optimization_mode, mode)

    def test_latency_ms_positive(self):
        opt = PromptOptimizer()
        result = opt.optimize("Write a test.", OptimizationMode.BALANCED)
        self.assertGreater(result.metadata.latency_ms, 0.0)

    def test_blank_prompt_raises(self):
        opt = PromptOptimizer()
        with self.assertRaises(ValueError):
            opt.optimize("   ", OptimizationMode.BALANCED)

    def test_empty_prompt_raises(self):
        opt = PromptOptimizer()
        with self.assertRaises(ValueError):
            opt.optimize("", OptimizationMode.BALANCED)

    def test_summary_mentions_unavailable(self):
        opt = PromptOptimizer()
        result = opt.optimize("Write a test.", OptimizationMode.BALANCED)
        self.assertIn("unavailable", result.summary.lower())

    def test_placeholders_empty_when_unavailable(self):
        opt = PromptOptimizer()
        result = opt.optimize("Write a test.", OptimizationMode.BALANCED)
        self.assertEqual(result.placeholders_inserted, [])


# ---------------------------------------------------------------------------
# MOCK mode tests
# ---------------------------------------------------------------------------
class TestPromptOptimizerMockMode(unittest.TestCase):
    def setUp(self):
        set_llm_service(_make_mock_svc("valid"))

    def tearDown(self):
        set_llm_service(None)
        set_optimizer_service(None)

    def test_mode_is_mock(self):
        opt = PromptOptimizer()
        result = opt.optimize(
            "Write a REST API endpoint for user login using JWT tokens.",
            OptimizationMode.BALANCED,
        )
        self.assertEqual(result.metadata.optimizer_mode, OptimizerMode.MOCK)

    def test_optimized_prompt_non_empty(self):
        opt = PromptOptimizer()
        result = opt.optimize("Write a REST API endpoint.", OptimizationMode.BALANCED)
        self.assertGreater(len(result.optimized_prompt), 0)

    def test_changes_list_populated(self):
        opt = PromptOptimizer()
        result = opt.optimize("Write a REST API endpoint.", OptimizationMode.BALANCED)
        self.assertIsInstance(result.changes, list)
        self.assertGreater(len(result.changes), 0)

    def test_changes_have_category_and_description(self):
        opt = PromptOptimizer()
        result = opt.optimize("Write a REST API endpoint.", OptimizationMode.BALANCED)
        for ch in result.changes:
            self.assertIsNotNone(ch.category)
            self.assertIsNotNone(ch.description)

    def test_preserved_requirements_list(self):
        opt = PromptOptimizer()
        result = opt.optimize("Write a REST API endpoint.", OptimizationMode.BALANCED)
        self.assertIsInstance(result.preserved_requirements, list)

    def test_preserved_requirements_have_reason(self):
        opt = PromptOptimizer()
        result = opt.optimize("Write a REST API endpoint.", OptimizationMode.BALANCED)
        for pr in result.preserved_requirements:
            self.assertIsNotNone(pr.reason)

    def test_placeholders_inserted_list(self):
        opt = PromptOptimizer()
        result = opt.optimize("Write a REST API endpoint.", OptimizationMode.BALANCED)
        self.assertIsInstance(result.placeholders_inserted, list)

    def test_improvement_score_delta_positive(self):
        opt = PromptOptimizer()
        result = opt.optimize("Write a REST API endpoint.", OptimizationMode.BALANCED)
        self.assertGreater(result.improvement_score_delta, 0.0)

    def test_summary_non_empty(self):
        opt = PromptOptimizer()
        result = opt.optimize("Write a REST API endpoint.", OptimizationMode.BALANCED)
        self.assertGreater(len(result.summary), 0)

    def test_metadata_version_set(self):
        opt = PromptOptimizer()
        result = opt.optimize("Write a REST API endpoint.", OptimizationMode.BALANCED)
        self.assertGreater(len(result.metadata.optimizer_version), 0)

    def test_all_four_modes_work(self):
        for mode in OptimizationMode:
            opt = PromptOptimizer()
            result = opt.optimize("Write a REST API endpoint.", mode)
            self.assertEqual(result.metadata.optimization_mode, mode)
            self.assertGreater(len(result.optimized_prompt), 0)

    def test_with_analyzer_result(self):
        analyzer_result = {
            "interpreted_goal": "Build a login endpoint.",
            "intent": {"label": "code_generation", "confidence": 0.90},
            "context_summary": "Python backend context.",
            "ambiguities": [{"issue": "JWT library not specified.", "evidence": "none", "confidence": 0.7}],
            "missing_information": [{"item": "Framework", "why_it_matters": "structure", "confidence": 0.8}],
            "contradictions": [],
            "instruction_quality": {"score": 72.0, "reason": "Good but incomplete."},
            "weaknesses": [{"weakness": "No error handling.", "evidence": "none"}],
            "recommendations": [{"recommendation": "Add framework spec.", "reason": "Clarity.", "priority": "HIGH"}],
        }
        opt = PromptOptimizer()
        result = opt.optimize(
            "Write a REST API endpoint for login.",
            OptimizationMode.ANALYTICAL,
            analyzer_result=analyzer_result,
        )
        self.assertEqual(result.metadata.optimizer_mode, OptimizerMode.MOCK)
        self.assertGreater(len(result.optimized_prompt), 0)


# ---------------------------------------------------------------------------
# Retry / fallback tests
# ---------------------------------------------------------------------------
class TestPromptOptimizerRetry(unittest.TestCase):
    def tearDown(self):
        set_llm_service(None)
        set_optimizer_service(None)

    def test_malformed_json_falls_back_to_unavailable(self):
        from app.llm.config import LLMConfig
        config = LLMConfig(enabled=True, provider="mock", max_retries=0)
        provider = MockLLMProvider(mode="malformed_json")
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

        opt = PromptOptimizer()
        result = opt.optimize("Write a unit test.", OptimizationMode.BALANCED)
        self.assertEqual(result.metadata.optimizer_mode, OptimizerMode.UNAVAILABLE)
        # Original prompt must be returned
        self.assertEqual(result.optimized_prompt, "Write a unit test.")

    def test_error_mode_falls_back_to_unavailable(self):
        from app.llm.config import LLMConfig
        config = LLMConfig(enabled=True, provider="mock", max_retries=0)
        provider = MockLLMProvider(mode="error")
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

        opt = PromptOptimizer()
        result = opt.optimize("Design a DB schema.", OptimizationMode.EXPERT)
        self.assertEqual(result.metadata.optimizer_mode, OptimizerMode.UNAVAILABLE)


# ---------------------------------------------------------------------------
# JSON extraction tests
# ---------------------------------------------------------------------------
class TestPromptOptimizerJsonExtraction(unittest.TestCase):
    def setUp(self):
        self.opt = PromptOptimizer()

    def _valid_payload(self):
        return {
            "optimized_prompt": "Improved prompt.",
            "summary": "Applied clarity improvements.",
            "changes": [{"category": "clarity", "description": "Clearer phrasing."}],
            "preserved_requirements": [],
            "placeholders_inserted": [],
            "improvement_score_delta": 5.0,
        }

    def test_extract_plain_json(self):
        raw = json.dumps(self._valid_payload())
        result = self.opt._extract_json(raw)
        self.assertEqual(result["optimized_prompt"], "Improved prompt.")

    def test_extract_markdown_fenced(self):
        raw = f"```json\n{json.dumps(self._valid_payload())}\n```"
        result = self.opt._extract_json(raw)
        self.assertEqual(result["optimized_prompt"], "Improved prompt.")

    def test_malformed_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.opt._extract_json("{ not valid json }")

    def test_no_json_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.opt._extract_json("This is just plain text with no JSON.")


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------
class TestPromptOptimizerValidation(unittest.TestCase):
    def setUp(self):
        self.opt = PromptOptimizer()

    def test_validate_empty_optimized_prompt_raises(self):
        with self.assertRaises(ValueError):
            self.opt._validate_llm_dict({
                "optimized_prompt": "  ",
                "summary": "ok",
                "improvement_score_delta": 5.0,
            })

    def test_validate_missing_summary_raises(self):
        with self.assertRaises(ValueError):
            self.opt._validate_llm_dict({
                "optimized_prompt": "Valid.",
                "improvement_score_delta": 5.0,
            })

    def test_validate_non_numeric_delta_raises(self):
        with self.assertRaises(ValueError):
            self.opt._validate_llm_dict({
                "optimized_prompt": "Valid.",
                "summary": "ok",
                "improvement_score_delta": "not-a-number",
            })

    def test_validate_change_missing_category_raises(self):
        with self.assertRaises(ValueError):
            self.opt._validate_llm_dict({
                "optimized_prompt": "Valid.",
                "summary": "ok",
                "improvement_score_delta": 5.0,
                "changes": [{"description": "No category."}],
            })

    def test_validate_preserved_missing_reason_raises(self):
        with self.assertRaises(ValueError):
            self.opt._validate_llm_dict({
                "optimized_prompt": "Valid.",
                "summary": "ok",
                "improvement_score_delta": 5.0,
                "preserved_requirements": [{"requirement": "X"}],
            })

    def test_validate_valid_dict_passes(self):
        # Should not raise
        self.opt._validate_llm_dict({
            "optimized_prompt": "Valid improved prompt.",
            "summary": "Improved clarity.",
            "changes": [{"category": "clarity", "description": "Fixed phrasing."}],
            "preserved_requirements": [{"requirement": "X", "reason": "Y"}],
            "placeholders_inserted": ["[FRAMEWORK]"],
            "improvement_score_delta": 12.0,
        })


if __name__ == "__main__":
    unittest.main()
