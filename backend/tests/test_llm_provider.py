import unittest

from app.llm.config import LLMConfig
from app.llm.model import MockLLMProvider, TransformersLocalProvider, create_provider
from app.llm.schemas import LLMHealthStatus


class LLMProviderTests(unittest.TestCase):
    def test_mock_provider_metadata_and_generation(self) -> None:
        provider = MockLLMProvider(model_name="test-mock")
        meta = provider.model_metadata()

        self.assertEqual(meta["name"], "test-mock")
        self.assertEqual(meta["provider"], "mock")
        self.assertEqual(provider.health(), LLMHealthStatus.MODEL_AVAILABLE)

        res = provider.generate("Test prompt", "System prompt", max_tokens=100)
        self.assertIn("interpreted_goal", res)
        self.assertEqual(provider.call_count, 1)

    def test_mock_provider_configurable_health(self) -> None:
        not_downloaded = MockLLMProvider(health_status=LLMHealthStatus.MODEL_NOT_DOWNLOADED)
        self.assertEqual(not_downloaded.health(), LLMHealthStatus.MODEL_NOT_DOWNLOADED)

        errored = MockLLMProvider(health_status=LLMHealthStatus.MODEL_ERROR)
        self.assertEqual(errored.health(), LLMHealthStatus.MODEL_ERROR)

    def test_create_provider_factory(self) -> None:
        mock_cfg = LLMConfig(provider="mock")
        prov = create_provider(mock_cfg)
        self.assertIsInstance(prov, MockLLMProvider)

        trans_cfg = LLMConfig(provider="transformers")
        prov_trans = create_provider(trans_cfg)
        self.assertIsInstance(prov_trans, TransformersLocalProvider)

        with self.assertRaises(ValueError):
            create_provider(LLMConfig(provider="unknown_provider"))

    def test_transformers_provider_disabled_state(self) -> None:
        cfg = LLMConfig(enabled=False, provider="transformers")
        prov = TransformersLocalProvider(cfg)
        self.assertEqual(prov.health(), LLMHealthStatus.LLM_DISABLED)


if __name__ == "__main__":
    unittest.main()
