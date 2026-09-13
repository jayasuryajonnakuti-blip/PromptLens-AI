import unittest

from app.ml.quality.features import (
    FEATURE_NAMES,
    extract_features,
    extract_features_batch,
)


class QualityFeaturesTests(unittest.TestCase):
    def test_feature_names_count(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), 30)

    def test_empty_prompt_returns_zeros(self) -> None:
        features = extract_features("")
        self.assertEqual(len(features), 30)
        self.assertTrue(all(val == 0.0 for val in features))

    def test_whitespace_prompt_returns_zeros(self) -> None:
        features = extract_features("   \n\t  ")
        self.assertEqual(len(features), 30)
        self.assertTrue(all(val == 0.0 for val in features))

    def test_normal_prompt_feature_structure(self) -> None:
        prompt = "Write a Python script that reads a CSV file and calculates the mean of a column."
        features = extract_features(prompt)

        self.assertEqual(len(features), 30)
        # Check non-zero for fundamental stats
        self.assertGreater(features[0], 0.0)  # character_count
        self.assertGreater(features[1], 0.0)  # word_count
        self.assertGreater(features[2], 0.0)  # sentence_count

    def test_deterministic_extraction(self) -> None:
        prompt = "Explain quantum computing in 3 bullet points with a Python code example."
        f1 = extract_features(prompt)
        f2 = extract_features(prompt)

        self.assertEqual(f1, f2)

    def test_long_prompt_extraction(self) -> None:
        long_prompt = "Write a comprehensive analysis of distributed databases. " * 200
        features = extract_features(long_prompt)

        self.assertEqual(len(features), 30)
        self.assertGreater(features[1], 1000.0)  # word_count

    def test_batch_extraction(self) -> None:
        prompts = ["First test prompt.", "Second test prompt."]
        batch = extract_features_batch(prompts)

        self.assertEqual(len(batch), 2)
        self.assertEqual(len(batch[0]), 30)
        self.assertEqual(len(batch[1]), 30)


if __name__ == "__main__":
    unittest.main()
