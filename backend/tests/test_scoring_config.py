import unittest

from app.scoring.config import (
    DEFAULT_SCORING_CONFIG,
    DEFAULT_SIGNAL_WEIGHTS,
    ScoringConfig,
)


class ScoringConfigTests(unittest.TestCase):
    def test_default_configured_weights_sum_to_one(self) -> None:
        weights = DEFAULT_SIGNAL_WEIGHTS
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        self.assertEqual(weights["rules"], 0.25)
        self.assertEqual(weights["nlp"], 0.15)
        self.assertEqual(weights["quality_ml"], 0.20)
        self.assertEqual(weights["llm"], 0.40)

    def test_missing_llm_effective_weights_renormalization(self) -> None:
        config = DEFAULT_SCORING_CONFIG
        available = {"rules", "nlp", "quality_ml"}
        effective = config.calculate_effective_weights(available)

        # Available signals must sum to 1.0
        self.assertAlmostEqual(sum(effective.values()), 1.0, places=5)
        self.assertEqual(effective["llm"], 0.0)

        # Ratio checks: 25/60, 15/60, 20/60
        self.assertAlmostEqual(effective["rules"], 25.0 / 60.0, places=4)
        self.assertAlmostEqual(effective["nlp"], 15.0 / 60.0, places=4)
        self.assertAlmostEqual(effective["quality_ml"], 20.0 / 60.0, places=4)

    def test_all_signals_available_weights(self) -> None:
        config = DEFAULT_SCORING_CONFIG
        all_signals = {"rules", "nlp", "quality_ml", "llm"}
        effective = config.calculate_effective_weights(all_signals)

        self.assertAlmostEqual(sum(effective.values()), 1.0, places=5)
        self.assertAlmostEqual(effective["rules"], 0.25, places=5)
        self.assertAlmostEqual(effective["nlp"], 0.15, places=5)
        self.assertAlmostEqual(effective["quality_ml"], 0.20, places=5)
        self.assertAlmostEqual(effective["llm"], 0.40, places=5)

    def test_single_available_signal_receives_full_weight(self) -> None:
        config = DEFAULT_SCORING_CONFIG
        effective = config.calculate_effective_weights({"rules"})

        self.assertAlmostEqual(effective["rules"], 1.0, places=5)
        self.assertEqual(effective["nlp"], 0.0)
        self.assertEqual(effective["quality_ml"], 0.0)
        self.assertEqual(effective["llm"], 0.0)

    def test_no_signals_available_returns_zeros(self) -> None:
        config = DEFAULT_SCORING_CONFIG
        effective = config.calculate_effective_weights(set())

        self.assertEqual(sum(effective.values()), 0.0)


if __name__ == "__main__":
    unittest.main()
