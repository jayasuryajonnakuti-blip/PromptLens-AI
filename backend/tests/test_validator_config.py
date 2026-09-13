"""Tests for PromptLens AI Validator configuration (Step 18)."""
import unittest

from app.validator.config import (
    MAX_EXPANSION_RATIO_FAIL,
    MAX_EXPANSION_RATIO_WARN,
    MAX_FATAL_CONTRADICTIONS,
    MAX_LOST_CRITICAL_REQUIREMENTS,
    MAX_SCORE_REGRESSION_FAIL,
    MAX_SCORE_REGRESSION_WARN,
    MAX_TRUNCATION_RATIO_WARN,
    MIN_SEMANTIC_SIMILARITY_FAIL,
    MIN_SEMANTIC_SIMILARITY_WARN,
    VALIDATOR_CONTEXT_CHAR_LIMIT,
    VALIDATOR_VERSION,
)


class TestValidatorConfig(unittest.TestCase):
    def test_version_present(self):
        self.assertEqual(VALIDATOR_VERSION, "1.0.0")

    def test_semantic_similarity_thresholds(self):
        self.assertLess(MIN_SEMANTIC_SIMILARITY_FAIL, MIN_SEMANTIC_SIMILARITY_WARN)
        self.assertGreater(MIN_SEMANTIC_SIMILARITY_FAIL, 0.0)
        self.assertLess(MIN_SEMANTIC_SIMILARITY_WARN, 1.0)

    def test_score_regression_thresholds(self):
        self.assertLess(MAX_SCORE_REGRESSION_WARN, MAX_SCORE_REGRESSION_FAIL)
        self.assertGreater(MAX_SCORE_REGRESSION_WARN, 0.0)

    def test_expansion_ratio_thresholds(self):
        self.assertLess(MAX_EXPANSION_RATIO_WARN, MAX_EXPANSION_RATIO_FAIL)
        self.assertGreater(MAX_EXPANSION_RATIO_WARN, 1.0)
        self.assertGreater(MAX_TRUNCATION_RATIO_WARN, 0.0)
        self.assertLess(MAX_TRUNCATION_RATIO_WARN, 1.0)

    def test_zero_tolerances(self):
        self.assertEqual(MAX_LOST_CRITICAL_REQUIREMENTS, 0)
        self.assertEqual(MAX_FATAL_CONTRADICTIONS, 0)

    def test_context_budget_limit(self):
        self.assertGreaterEqual(VALIDATOR_CONTEXT_CHAR_LIMIT, 1000)


if __name__ == "__main__":
    unittest.main()
