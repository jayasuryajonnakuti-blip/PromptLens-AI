import unittest

from app.ml.quality.categories import QualityLabel
from app.services.quality_service import (
    QualityModel,
    load_quality_model,
    predict_quality,
)


class QualityModelArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_quality_model()

    def test_model_metadata(self) -> None:
        self.assertEqual(self.model.metadata.name, "promptlens-quality-model")
        self.assertEqual(self.model.metadata.version, "0.1.0")

    def test_prediction_output_structure(self) -> None:
        response = self.model.predict("Write a simple Python script to reverse a string.")

        self.assertIsInstance(response.score, float)
        self.assertGreaterEqual(response.score, 0.0)
        self.assertLessEqual(response.score, 100.0)
        self.assertIsInstance(response.label, QualityLabel)
        self.assertEqual(response.model.name, "promptlens-quality-model")
        self.assertEqual(response.model.version, "0.1.0")

    def test_predict_quality_service_function(self) -> None:
        response = predict_quality("Explain machine learning with a practical example.")

        self.assertGreaterEqual(response.score, 0.0)
        self.assertLessEqual(response.score, 100.0)
        self.assertIn(response.label.value, {"POOR", "FAIR", "GOOD", "STRONG", "EXCELLENT"})

    def test_predict_quality_empty_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            predict_quality("")

    def test_predict_quality_whitespace_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            predict_quality("   \n\t  ")

    def test_multi_thousand_word_prompt_compatibility(self) -> None:
        large_prompt = "You are an enterprise system architect. Explain cloud resilience. " * 300
        response = predict_quality(large_prompt)

        self.assertGreaterEqual(response.score, 0.0)
        self.assertLessEqual(response.score, 100.0)


if __name__ == "__main__":
    unittest.main()
