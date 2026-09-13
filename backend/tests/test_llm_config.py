import os
import unittest

from app.llm.config import LLMConfig, load_llm_config


class LLMConfigTests(unittest.TestCase):
    def test_default_config_values(self) -> None:
        config = LLMConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.provider, "transformers")
        self.assertEqual(config.model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(config.context_length, 2048)
        self.assertEqual(config.max_tokens, 768)
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.timeout, 60.0)
        self.assertEqual(config.max_retries, 2)

    def test_invalid_max_tokens_exceeding_context_length(self) -> None:
        with self.assertRaises(ValueError):
            LLMConfig(context_length=512, max_tokens=1024)

    def test_invalid_negative_temperature(self) -> None:
        with self.assertRaises(ValueError):
            LLMConfig(temperature=-0.5)

    def test_invalid_zero_context_length(self) -> None:
        with self.assertRaises(ValueError):
            LLMConfig(context_length=0)

    def test_invalid_negative_retries(self) -> None:
        with self.assertRaises(ValueError):
            LLMConfig(max_retries=-1)

    def test_env_variable_overrides(self) -> None:
        old_env = os.environ.copy()
        try:
            os.environ["LLM_ENABLED"] = "true"
            os.environ["LLM_PROVIDER"] = "mock"
            os.environ["LLM_MODEL"] = "test-model-v1"
            os.environ["LLM_CONTEXT_LENGTH"] = "4096"
            os.environ["LLM_MAX_TOKENS"] = "1024"
            os.environ["LLM_TEMPERATURE"] = "0.2"
            os.environ["LLM_TIMEOUT"] = "45.0"
            os.environ["LLM_MAX_RETRIES"] = "3"

            loaded = load_llm_config()
            self.assertTrue(loaded.enabled)
            self.assertEqual(loaded.provider, "mock")
            self.assertEqual(loaded.model, "test-model-v1")
            self.assertEqual(loaded.context_length, 4096)
            self.assertEqual(loaded.max_tokens, 1024)
            self.assertEqual(loaded.temperature, 0.2)
            self.assertEqual(loaded.timeout, 45.0)
            self.assertEqual(loaded.max_retries, 3)
        finally:
            os.environ.clear()
            os.environ.update(old_env)


if __name__ == "__main__":
    unittest.main()
