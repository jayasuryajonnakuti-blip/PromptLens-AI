"""Unit tests for Agent Loop configuration and constants (Step 19)."""
from __future__ import annotations

import unittest

from app.agent.config import (
    AGENT_DISCLAIMERS,
    AGENT_VERSION,
    DISCLAIMER_BOUNDED_LOOP,
    DISCLAIMER_PASS_TERMINATES,
    DISCLAIMER_REVIEW_TERMINATES,
    DISCLAIMER_SEMANTIC_SIMILARITY,
    DISCLAIMER_UNVALIDATED_NEVER_ACCEPTED,
    MAX_CONSECUTIVE_IDENTICAL_ERRORS,
    MAX_CONSECUTIVE_REGRESSIONS,
    MAX_OPTIMIZATION_ITERATIONS,
    MAX_PROMPT_EXPANSION_RATIO,
    MIN_CHAR_DELTA_RATIO,
    MIN_SCORE_IMPROVEMENT,
    MIN_SEMANTIC_SIMILARITY,
)


class TestAgentConfig(unittest.TestCase):
    """Test suite for Step 19 Agent Loop configuration."""

    def test_version_present(self) -> None:
        self.assertEqual(AGENT_VERSION, "1.0.0")

    def test_max_optimization_iterations(self) -> None:
        self.assertEqual(MAX_OPTIMIZATION_ITERATIONS, 3)

    def test_thresholds(self) -> None:
        self.assertGreater(MIN_SCORE_IMPROVEMENT, 0.0)
        self.assertEqual(MAX_PROMPT_EXPANSION_RATIO, 4.0)
        self.assertEqual(MIN_SEMANTIC_SIMILARITY, 0.40)
        self.assertEqual(MIN_CHAR_DELTA_RATIO, 0.02)
        self.assertEqual(MAX_CONSECUTIVE_REGRESSIONS, 2)
        self.assertEqual(MAX_CONSECUTIVE_IDENTICAL_ERRORS, 2)

    def test_mandatory_disclaimers(self) -> None:
        self.assertIn(DISCLAIMER_BOUNDED_LOOP, AGENT_DISCLAIMERS)
        self.assertIn(DISCLAIMER_UNVALIDATED_NEVER_ACCEPTED, AGENT_DISCLAIMERS)
        self.assertIn(DISCLAIMER_PASS_TERMINATES, AGENT_DISCLAIMERS)
        self.assertIn(DISCLAIMER_REVIEW_TERMINATES, AGENT_DISCLAIMERS)
        self.assertIn(DISCLAIMER_SEMANTIC_SIMILARITY, AGENT_DISCLAIMERS)
        self.assertEqual(len(AGENT_DISCLAIMERS), 5)


if __name__ == "__main__":
    unittest.main()
