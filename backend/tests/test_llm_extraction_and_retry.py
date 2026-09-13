import unittest

from app.llm.config import LLMConfig
from app.llm.errors import InvalidJsonError, RetryExhaustedError
from app.llm.model import MockLLMProvider
from app.services.llm_service import LLMService


class LLMExtractionAndRetryTests(unittest.TestCase):
    def test_clean_json_extraction(self) -> None:
        service = LLMService(LLMConfig(enabled=True, provider="mock"))
        raw = '{"interpreted_goal": "Goal", "context": "Context"}'
        parsed = service._extract_json(raw)
        self.assertEqual(parsed["interpreted_goal"], "Goal")

    def test_markdown_fence_json_extraction(self) -> None:
        service = LLMService(LLMConfig(enabled=True, provider="mock"))
        raw = 'Here is the analysis:\n```json\n{"interpreted_goal": "Goal", "context": "Context"}\n```\nHope this helps!'
        parsed = service._extract_json(raw)
        self.assertEqual(parsed["interpreted_goal"], "Goal")

    def test_prose_surrounded_json_extraction(self) -> None:
        service = LLMService(LLMConfig(enabled=True, provider="mock"))
        raw = 'Certainly! {\n  "interpreted_goal": "Goal",\n  "context": "Context"\n}\nEnd of response.'
        parsed = service._extract_json(raw)
        self.assertEqual(parsed["interpreted_goal"], "Goal")

    def test_no_json_found_raises_invalid_json_error(self) -> None:
        service = LLMService(LLMConfig(enabled=True, provider="mock"))
        raw = "This is pure conversational prose without any brackets at all."
        with self.assertRaises(InvalidJsonError):
            service._extract_json(raw)

    def test_retry_succeeds_after_one_failure(self) -> None:
        # Mock provider will fail once then succeed
        provider = MockLLMProvider(mode="fail_once_then_succeed")
        config = LLMConfig(enabled=True, provider="mock", max_retries=2)
        service = LLMService(config=config, provider=provider)

        resp = service.analyze_prompt("Test prompt")
        self.assertTrue(resp.structured_output_valid)
        self.assertEqual(provider.call_count, 2)

    def test_retry_exhaustion_on_malformed_json(self) -> None:
        provider = MockLLMProvider(mode="malformed_json")
        config = LLMConfig(enabled=True, provider="mock", max_retries=1)
        service = LLMService(config=config, provider=provider)

        with self.assertRaises(RetryExhaustedError):
            service.analyze_prompt("Test prompt")

        # 1 initial + 1 retry = 2 attempts
        self.assertEqual(provider.call_count, 2)

    def test_retry_exhaustion_on_schema_invalid(self) -> None:
        provider = MockLLMProvider(mode="schema_invalid")
        config = LLMConfig(enabled=True, provider="mock", max_retries=1)
        service = LLMService(config=config, provider=provider)

        with self.assertRaises(RetryExhaustedError):
            service.analyze_prompt("Test prompt")

        self.assertEqual(provider.call_count, 2)


if __name__ == "__main__":
    unittest.main()
