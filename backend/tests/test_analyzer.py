"""Tests for PromptAnalyzer core class (Step 15).

All tests use MockLLMProvider injected via set_llm_service() for test isolation.
No LLM weights are downloaded; no commercial APIs are called.
"""
import unittest

from app.analyzer.analyzer import PromptAnalyzer
from app.analyzer.schemas import AnalysisMode, AnalyzerAnalysis
from app.llm.model import MockLLMProvider
from app.llm.schemas import LLMHealthStatus
from app.services.analyzer_service import set_analyzer_service
from app.services.llm_service import LLMService, set_llm_service


def _make_llm_service(mode: str = "valid") -> LLMService:
    """Create an LLMService backed by MockLLMProvider in specified mode."""
    from app.llm.config import LLMConfig

    config = LLMConfig(enabled=True, provider="mock", max_retries=2)
    provider = MockLLMProvider(mode=mode)
    return LLMService(config=config, provider=provider)


# ---------------------------------------------------------------------------
# We need MockLLMProvider to return analyzer-compatible JSON, not Step 14 JSON.
# Build a subclass that returns the richer analyzer schema.
# ---------------------------------------------------------------------------
import json


class AnalyzerMockProvider(MockLLMProvider):
    """Mock provider returning analyzer-compatible JSON."""

    def _valid_payload(self) -> str:
        return json.dumps({
            "interpreted_goal": "Build a REST API endpoint for user authentication.",
            "context_summary": "Developer task in a web backend context.",
            "intent_label": "code_generation",
            "intent_confidence": 0.88,
            "ambiguities": [
                {"issue": "Authentication protocol not specified.", "evidence": "No mention of JWT or OAuth.", "confidence": 0.75}
            ],
            "missing_information": [
                {"item": "Target language/framework", "why_it_matters": "Implementation differs by stack.", "confidence": 0.80}
            ],
            "contradictions": [],
            "instruction_quality": {
                "score": 70.0,
                "reason": "Clear goal but missing implementation constraints.",
            },
            "strengths": [
                {"strength": "Explicit functional requirement.", "evidence": "Endpoint purpose is stated."}
            ],
            "weaknesses": [
                {"weakness": "No error handling mentioned.", "evidence": "Prompt lacks constraint language."}
            ],
            "recommendations": [
                {"recommendation": "Specify the auth protocol.", "reason": "Reduces ambiguity.", "priority": "HIGH"},
                {"recommendation": "Add error handling requirements.", "reason": "Completeness.", "priority": "MEDIUM"},
            ],
        })


class TestPromptAnalyzerUnavailableMode(unittest.TestCase):
    """Tests for UNAVAILABLE mode (LLM disabled)."""

    def setUp(self):
        # Ensure LLM is disabled
        from app.llm.config import LLMConfig

        config = LLMConfig(enabled=False, provider="transformers")
        provider = MockLLMProvider(health_status=LLMHealthStatus.LLM_DISABLED)
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

    def tearDown(self):
        set_llm_service(None)
        set_analyzer_service(None)

    def test_returns_analyzer_analysis(self):
        analyzer = PromptAnalyzer()
        result = analyzer.analyze("Write a Python function to sort a list.")
        self.assertIsInstance(result, AnalyzerAnalysis)

    def test_mode_is_unavailable(self):
        analyzer = PromptAnalyzer()
        result = analyzer.analyze("Write a Python function to sort a list.")
        self.assertEqual(result.metadata.analysis_mode, AnalysisMode.UNAVAILABLE)

    def test_llm_model_is_none(self):
        analyzer = PromptAnalyzer()
        result = analyzer.analyze("Write a Python function to sort a list.")
        self.assertEqual(result.metadata.llm_model, "none")

    def test_evidence_is_list(self):
        analyzer = PromptAnalyzer()
        result = analyzer.analyze("Write a Python function to sort a list.")
        self.assertIsInstance(result.evidence, list)

    def test_deterministic_evidence_populated(self):
        analyzer = PromptAnalyzer()
        result = analyzer.analyze(
            "You are a senior Python developer. Write a REST API endpoint. "
            "Avoid using synchronous libraries. Return JSON only."
        )
        self.assertGreater(len(result.evidence), 0)

    def test_instruction_quality_in_range(self):
        analyzer = PromptAnalyzer()
        result = analyzer.analyze("Translate the following text to French.")
        self.assertGreaterEqual(result.instruction_quality.score, 0.0)
        self.assertLessEqual(result.instruction_quality.score, 100.0)

    def test_metadata_version_set(self):
        analyzer = PromptAnalyzer()
        result = analyzer.analyze("Summarize the article.")
        self.assertIsNotNone(result.metadata.analyzer_version)
        self.assertGreater(len(result.metadata.analyzer_version), 0)

    def test_latency_ms_positive(self):
        analyzer = PromptAnalyzer()
        result = analyzer.analyze("What is the capital of France?")
        self.assertGreater(result.metadata.latency_ms, 0.0)

    def test_blank_prompt_raises(self):
        analyzer = PromptAnalyzer()
        with self.assertRaises(ValueError):
            analyzer.analyze("   ")

    def test_empty_prompt_raises(self):
        analyzer = PromptAnalyzer()
        with self.assertRaises(ValueError):
            analyzer.analyze("")


class TestPromptAnalyzerMockMode(unittest.TestCase):
    """Tests for MOCK mode (MockProvider with valid analyzer output)."""

    def setUp(self):
        from app.llm.config import LLMConfig

        config = LLMConfig(enabled=True, provider="mock", max_retries=2)
        provider = AnalyzerMockProvider(mode="valid")
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

    def tearDown(self):
        set_llm_service(None)
        set_analyzer_service(None)

    def test_mode_is_mock(self):
        analyzer = PromptAnalyzer()
        result = analyzer.analyze("Build a REST API endpoint for user authentication.")
        self.assertEqual(result.metadata.analysis_mode, AnalysisMode.MOCK)

    def test_interpreted_goal_populated(self):
        analyzer = PromptAnalyzer()
        result = analyzer.analyze("Build a REST API endpoint for user authentication.")
        self.assertGreater(len(result.interpreted_goal), 0)

    def test_intent_populated(self):
        analyzer = PromptAnalyzer()
        result = analyzer.analyze("Build a REST API endpoint for user authentication.")
        self.assertIsNotNone(result.intent)
        self.assertGreater(len(result.intent.label), 0)
        self.assertGreaterEqual(result.intent.confidence, 0.0)
        self.assertLessEqual(result.intent.confidence, 1.0)

    def test_ambiguities_populated(self):
        analyzer = PromptAnalyzer()
        result = analyzer.analyze("Build a REST API endpoint for user authentication.")
        self.assertIsInstance(result.ambiguities, list)

    def test_missing_info_populated(self):
        analyzer = PromptAnalyzer()
        result = analyzer.analyze("Build a REST API endpoint for user authentication.")
        self.assertIsInstance(result.missing_information, list)

    def test_contradictions_list(self):
        analyzer = PromptAnalyzer()
        result = analyzer.analyze("Build a REST API endpoint for user authentication.")
        self.assertIsInstance(result.contradictions, list)

    def test_recommendations_have_priorities(self):
        analyzer = PromptAnalyzer()
        result = analyzer.analyze("Build a REST API endpoint for user authentication.")
        for rec in result.recommendations:
            self.assertIn(rec.priority.value, ("HIGH", "MEDIUM", "LOW"))

    def test_evidence_includes_llm_source(self):
        from app.analyzer.schemas import EvidenceSource

        analyzer = PromptAnalyzer()
        result = analyzer.analyze("Build a REST API endpoint for user authentication.")
        sources = {e.source for e in result.evidence}
        self.assertIn(EvidenceSource.LLM, sources)

    def test_instruction_quality_in_range(self):
        analyzer = PromptAnalyzer()
        result = analyzer.analyze("Build a REST API endpoint for user authentication.")
        self.assertGreaterEqual(result.instruction_quality.score, 0.0)
        self.assertLessEqual(result.instruction_quality.score, 100.0)

    def test_strengths_and_weaknesses(self):
        analyzer = PromptAnalyzer()
        result = analyzer.analyze("Build a REST API endpoint for user authentication.")
        self.assertIsInstance(result.strengths, list)
        self.assertIsInstance(result.weaknesses, list)


class TestPromptAnalyzerRetry(unittest.TestCase):
    """Tests for retry/fallback behaviour."""

    def tearDown(self):
        set_llm_service(None)
        set_analyzer_service(None)

    def test_malformed_json_falls_back_to_unavailable(self):
        from app.llm.config import LLMConfig

        config = LLMConfig(enabled=True, provider="mock", max_retries=0)
        provider = MockLLMProvider(mode="malformed_json")
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

        analyzer = PromptAnalyzer()
        result = analyzer.analyze("Write a unit test for a Python class.")
        # After exhausting retries, should fall back to UNAVAILABLE
        self.assertEqual(result.metadata.analysis_mode, AnalysisMode.UNAVAILABLE)

    def test_error_mode_falls_back_to_unavailable(self):
        from app.llm.config import LLMConfig

        config = LLMConfig(enabled=True, provider="mock", max_retries=0)
        provider = MockLLMProvider(mode="error")
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

        analyzer = PromptAnalyzer()
        result = analyzer.analyze("Design a database schema for an e-commerce platform.")
        self.assertEqual(result.metadata.analysis_mode, AnalysisMode.UNAVAILABLE)


class TestPromptAnalyzerValidation(unittest.TestCase):
    """Tests for _validate_llm_dict internal validation."""

    def test_validate_missing_interpreted_goal_raises(self):
        analyzer = PromptAnalyzer()
        with self.assertRaises(ValueError):
            analyzer._validate_llm_dict({
                "context_summary": "ok",
                "intent_label": "other",
                "intent_confidence": 0.5,
                "instruction_quality": {"score": 50.0, "reason": "ok"},
            })

    def test_validate_score_out_of_range_raises(self):
        analyzer = PromptAnalyzer()
        with self.assertRaises(ValueError):
            analyzer._validate_llm_dict({
                "interpreted_goal": "ok",
                "context_summary": "ok",
                "intent_label": "other",
                "intent_confidence": 0.5,
                "instruction_quality": {"score": 150.0, "reason": "bad"},
            })

    def test_validate_invalid_priority_raises(self):
        analyzer = PromptAnalyzer()
        with self.assertRaises(ValueError):
            analyzer._validate_llm_dict({
                "interpreted_goal": "ok",
                "context_summary": "ok",
                "intent_label": "other",
                "intent_confidence": 0.5,
                "instruction_quality": {"score": 50.0, "reason": "ok"},
                "recommendations": [
                    {"recommendation": "X", "reason": "Y", "priority": "EXTREME"}
                ],
            })


if __name__ == "__main__":
    unittest.main()
