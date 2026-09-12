import csv
import json
import tempfile
import unittest
from pathlib import Path

from app.ml.intent.dataset import DatasetValidationError, load_and_validate_dataset
from app.ml.intent.inference import MODEL_PATH, load_classifier
from app.schemas.intent import IntentLabel

DATASET_PATH = Path(__file__).parents[1] / "data" / "intent_dataset_v1.csv"


class IntentPipelineTests(unittest.TestCase):
    def test_dataset_covers_every_supported_intent(self) -> None:
        dataset = load_and_validate_dataset(DATASET_PATH)

        self.assertEqual(set(dataset.distribution), {intent.value for intent in IntentLabel})
        self.assertEqual(len(dataset.texts), 96)
        self.assertEqual(set(dataset.distribution.values()), {8})

    def test_unsupported_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            path.write_text("text,intent\nA prompt,NOT_SUPPORTED\n", encoding="utf-8")

            with self.assertRaises(DatasetValidationError):
                load_and_validate_dataset(path)

    def test_duplicate_prompt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.csv"
            rows = [{"text": "same", "intent": "GENERAL"}] * 2
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["text", "intent"])
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaises(DatasetValidationError):
                load_and_validate_dataset(path)

    def test_trained_model_artifact_loads(self) -> None:
        self.assertTrue(MODEL_PATH.is_file())
        classifier = load_classifier()
        result = classifier.predict("Write a Python script to parse JSON.")

        self.assertIn(result.intent, list(IntentLabel))
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertTrue(result.model.name)
        self.assertTrue(result.model.version)
        self.assertTrue(result.confidence_methodology)

    def test_inference_is_reproducible(self) -> None:
        classifier = load_classifier()
        first = classifier.predict("Summarize this research report.").model_dump()
        second = classifier.predict("Summarize this research report.").model_dump()

        self.assertEqual(first, second)

    def test_evaluation_artifact_has_metadata(self) -> None:
        artifact_path = Path(__file__).parents[1] / "artifacts" / "intent" / "evaluation.json"
        metadata = json.loads(artifact_path.read_text(encoding="utf-8"))

        self.assertIn("validation_metrics", metadata)
        self.assertIn("test_metrics", metadata)
        self.assertIn("selected_model", metadata)
        self.assertEqual(len(metadata["supported_intents"]), 12)


if __name__ == "__main__":
    unittest.main()
