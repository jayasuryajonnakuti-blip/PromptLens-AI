"""Tests for AnalyzerContext and build_analyzer_context() (Step 15).

Uses only the AnalyzerContext dataclass directly to test the compact dict
output — no external service calls needed for most tests.
"""
import unittest

from app.analyzer.context import AnalyzerContext


class TestAnalyzerContextDefaults(unittest.TestCase):
    def test_default_values(self):
        ctx = AnalyzerContext()
        self.assertEqual(ctx.word_count, 0)
        self.assertEqual(ctx.intent_label, "")
        self.assertFalse(ctx.intent_available)
        self.assertFalse(ctx.quality_available)
        self.assertFalse(ctx.step13_available)

    def test_compact_dict_empty(self):
        ctx = AnalyzerContext()
        d = ctx.to_compact_dict()
        self.assertIsInstance(d, dict)
        # word_count and sentence_count always present
        self.assertIn("word_count", d)
        self.assertIn("sentence_count", d)
        # Intent/quality/scoring absent when not available
        self.assertNotIn("detected_intent", d)
        self.assertNotIn("baseline_quality_score", d)
        self.assertNotIn("step13_overall_score", d)


class TestAnalyzerContextWithData(unittest.TestCase):
    def _make_context(self) -> AnalyzerContext:
        ctx = AnalyzerContext()
        ctx.word_count = 35
        ctx.sentence_count = 3
        ctx.output_formats = ["markdown", "json"]
        ctx.sections_detected = ["CONTEXT", "OUTPUT"]
        ctx.has_role_definition = True
        ctx.has_constraints = True
        ctx.has_examples = False
        ctx.entity_count = 2
        ctx.entity_labels = ["Python", "REST API"]
        ctx.verb_count = 5
        ctx.noun_count = 10
        ctx.avg_sentence_length = 11.7
        ctx.has_imperative = True
        ctx.intent_label = "code_generation"
        ctx.intent_confidence = 0.88
        ctx.intent_available = True
        ctx.quality_score = 72.5
        ctx.quality_category = "GOOD"
        ctx.quality_available = True
        ctx.step13_overall_score = 68.0
        ctx.step13_category = "GOOD"
        ctx.step13_available = True
        ctx.top_dimensions = [
            {"dimension": "clarity", "score": 45.0, "status": "NEEDS_WORK", "reason": "Unclear."},
            {"dimension": "specificity", "score": 55.0, "status": "FAIR", "reason": "Adequate."},
        ]
        ctx.step13_recommendations = ["Add output format", "Clarify constraints"]
        return ctx

    def test_compact_dict_includes_all_signals(self):
        ctx = self._make_context()
        d = ctx.to_compact_dict()

        self.assertEqual(d["word_count"], 35)
        self.assertEqual(d["sentence_count"], 3)
        self.assertIn("output_formats", d)
        self.assertIn("sections_detected", d)
        self.assertIn("structural_flags", d)
        self.assertIn("role_definition_present", d["structural_flags"])
        self.assertIn("constraints_present", d["structural_flags"])
        self.assertNotIn("examples_present", d["structural_flags"])
        self.assertIn("named_entities", d)
        self.assertEqual(d["verb_count"], 5)
        self.assertEqual(d["noun_count"], 10)
        self.assertAlmostEqual(d["avg_sentence_length"], 11.7, places=1)
        self.assertTrue(d.get("imperative_detected"))
        self.assertEqual(d["detected_intent"], "code_generation")
        self.assertAlmostEqual(d["intent_confidence"], 0.88, places=2)
        self.assertAlmostEqual(d["baseline_quality_score"], 72.5, places=1)
        self.assertEqual(d["quality_category"], "GOOD")
        self.assertAlmostEqual(d["step13_overall_score"], 68.0, places=1)
        self.assertIn("dimension_highlights", d)
        self.assertIn("step13_recommendations", d)

    def test_entity_labels_capped_at_ten(self):
        ctx = AnalyzerContext()
        ctx.entity_labels = [f"ENT{i}" for i in range(20)]
        d = ctx.to_compact_dict()
        self.assertLessEqual(len(d.get("named_entities", [])), 10)

    def test_step13_recommendations_capped_at_five(self):
        ctx = AnalyzerContext()
        ctx.step13_available = True
        ctx.step13_overall_score = 50.0
        ctx.step13_category = "FAIR"
        ctx.step13_recommendations = [f"Rec {i}" for i in range(10)]
        d = ctx.to_compact_dict()
        self.assertLessEqual(len(d.get("step13_recommendations", [])), 5)

    def test_imperative_absent_when_false(self):
        ctx = AnalyzerContext()
        ctx.verb_count = 3
        ctx.noun_count = 9
        ctx.has_imperative = False
        d = ctx.to_compact_dict()
        self.assertNotIn("imperative_detected", d)

    def test_no_structural_flags_when_none(self):
        ctx = AnalyzerContext()
        ctx.word_count = 10
        d = ctx.to_compact_dict()
        self.assertNotIn("structural_flags", d)


class TestBuildAnalyzerContextIntegration(unittest.TestCase):
    """Integration tests that call build_analyzer_context() with a real prompt.

    These tests verify the function runs without error and returns an
    AnalyzerContext with expected field types.  Exact numeric values are
    not asserted because they depend on loaded ML models.
    """

    def test_build_returns_analyzer_context(self):
        from app.analyzer.context import build_analyzer_context

        ctx = build_analyzer_context("Write a Python function to sort a list.")
        self.assertIsInstance(ctx, AnalyzerContext)

    def test_build_word_count_positive(self):
        from app.analyzer.context import build_analyzer_context

        ctx = build_analyzer_context("Write a Python function to sort a list.")
        self.assertGreater(ctx.word_count, 0)

    def test_build_compact_dict_is_dict(self):
        from app.analyzer.context import build_analyzer_context

        ctx = build_analyzer_context("Explain the difference between REST and GraphQL APIs.")
        d = ctx.to_compact_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("word_count", d)


if __name__ == "__main__":
    unittest.main()
