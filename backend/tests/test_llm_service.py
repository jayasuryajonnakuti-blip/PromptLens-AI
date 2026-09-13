import unittest

from app.llm.config import LLMConfig
from app.llm.errors import LLMDisabledError, ModelNotDownloadedError
from app.llm.model import MockLLMProvider
from app.llm.schemas import LLMHealthStatus
from app.services.llm_service import LLMService


class LLMServiceTests(unittest.TestCase):
    def test_empty_prompt_raises_value_error(self) -> None:
        service = LLMService(LLMConfig(enabled=True, provider="mock"))
        with self.assertRaises(ValueError):
            service.analyze_prompt("")

    def test_disabled_llm_raises_disabled_error(self) -> None:
        service = LLMService(LLMConfig(enabled=False, provider="mock"))
        # Force provider health to report disabled
        service.provider = MockLLMProvider(health_status=LLMHealthStatus.LLM_DISABLED)
        with self.assertRaises(LLMDisabledError):
            service.analyze_prompt("Valid prompt")

    def test_not_downloaded_model_raises_not_downloaded_error(self) -> None:
        service = LLMService(LLMConfig(enabled=True, provider="mock"))
        service.provider = MockLLMProvider(health_status=LLMHealthStatus.MODEL_NOT_DOWNLOADED)
        with self.assertRaises(ModelNotDownloadedError):
            service.analyze_prompt("Valid prompt")

    def test_successful_analysis_response_structure(self) -> None:
        service = LLMService(LLMConfig(enabled=True, provider="mock"))
        resp = service.analyze_prompt("Explain quantum computing to high school students.")

        self.assertTrue(resp.structured_output_valid)
        self.assertGreater(len(resp.analysis.interpreted_goal), 0)
        self.assertGreaterEqual(resp.analysis.instruction_quality.score, 0.0)
        self.assertLessEqual(resp.analysis.instruction_quality.score, 100.0)
        self.assertIsInstance(resp.analysis.strengths, list)
        self.assertIsInstance(resp.analysis.weaknesses, list)
        self.assertIsInstance(resp.analysis.recommendations, list)
        self.assertEqual(resp.provider, "mock")
        self.assertGreaterEqual(resp.latency_ms, 0.0)

    def test_context_summary_generation_integration(self) -> None:
        service = LLMService(LLMConfig(enabled=True, provider="mock"))
        summary = service._build_context_summary("Write a Python script to sort numbers.")

        self.assertIn("word_count", summary)
        self.assertIn("detected_intent", summary)
        self.assertIn("baseline_quality_score", summary)


if __name__ == "__main__":
    unittest.main()
