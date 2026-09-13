"""Tests for evidence generation utilities (Step 15)."""
import unittest

from app.analyzer.context import AnalyzerContext
from app.analyzer.evidence import (
    build_deterministic_evidence,
    build_llm_evidence_items,
)
from app.analyzer.schemas import EvidenceSource, EvidenceType


def _make_full_context() -> AnalyzerContext:
    ctx = AnalyzerContext()
    ctx.word_count = 40
    ctx.sentence_count = 4
    ctx.output_formats = ["json"]
    ctx.sections_detected = ["OUTPUT"]
    ctx.has_role_definition = True
    ctx.has_constraints = True
    ctx.has_examples = True
    ctx.entity_count = 3
    ctx.entity_labels = ["Python", "API", "REST"]
    ctx.verb_count = 6
    ctx.noun_count = 12
    ctx.avg_sentence_length = 10.0
    ctx.has_imperative = True
    ctx.intent_label = "code_generation"
    ctx.intent_confidence = 0.92
    ctx.intent_available = True
    ctx.quality_score = 78.0
    ctx.quality_category = "GOOD"
    ctx.quality_available = True
    ctx.step13_overall_score = 72.0
    ctx.step13_category = "GOOD"
    ctx.step13_available = True
    ctx.top_dimensions = [
        {"dimension": "clarity", "score": 40.0, "status": "NEEDS_WORK", "reason": "Unclear phrasing."},
        {"dimension": "specificity", "score": 55.0, "status": "FAIR", "reason": "Somewhat specific."},
        {"dimension": "context", "score": 80.0, "status": "GOOD", "reason": "Good context."},
    ]
    return ctx


class TestBuildDeterministicEvidence(unittest.TestCase):
    def test_returns_list(self):
        ctx = _make_full_context()
        items = build_deterministic_evidence(ctx)
        self.assertIsInstance(items, list)
        self.assertGreater(len(items), 0)

    def test_all_items_are_evidence_items(self):
        from app.analyzer.schemas import EvidenceItem

        ctx = _make_full_context()
        items = build_deterministic_evidence(ctx)
        for item in items:
            self.assertIsInstance(item, EvidenceItem)

    def test_sources_ordering(self):
        """Preprocessing evidence must appear before NLP, intent, quality, scoring."""
        ctx = _make_full_context()
        items = build_deterministic_evidence(ctx)
        sources = [item.source for item in items]
        # Find first occurrence of each source
        def first_idx(source):
            try:
                return next(i for i, s in enumerate(sources) if s == source)
            except StopIteration:
                return None

        prep_idx = first_idx(EvidenceSource.PREPROCESSING)
        nlp_idx = first_idx(EvidenceSource.NLP)
        intent_idx = first_idx(EvidenceSource.INTENT)
        quality_idx = first_idx(EvidenceSource.QUALITY_ML)
        scoring_idx = first_idx(EvidenceSource.SCORING)

        if prep_idx is not None and nlp_idx is not None:
            self.assertLess(prep_idx, nlp_idx)
        if nlp_idx is not None and intent_idx is not None:
            self.assertLess(nlp_idx, intent_idx)
        if intent_idx is not None and quality_idx is not None:
            self.assertLess(intent_idx, quality_idx)
        if quality_idx is not None and scoring_idx is not None:
            self.assertLess(quality_idx, scoring_idx)

    def test_all_confidences_in_range(self):
        ctx = _make_full_context()
        items = build_deterministic_evidence(ctx)
        for item in items:
            self.assertGreaterEqual(item.confidence, 0.0)
            self.assertLessEqual(item.confidence, 1.0)

    def test_word_count_evidence_present(self):
        ctx = _make_full_context()
        items = build_deterministic_evidence(ctx)
        preprocessing_obs = [
            i for i in items
            if i.source == EvidenceSource.PREPROCESSING
            and i.type == EvidenceType.OBSERVATION
        ]
        self.assertTrue(any("word" in i.statement.lower() for i in preprocessing_obs))

    def test_intent_high_confidence_label(self):
        ctx = AnalyzerContext()
        ctx.intent_label = "summarization"
        ctx.intent_confidence = 0.85
        ctx.intent_available = True
        items = build_deterministic_evidence(ctx)
        intent_items = [i for i in items if i.source == EvidenceSource.INTENT]
        self.assertEqual(len(intent_items), 1)
        self.assertIn("summarization", intent_items[0].statement)
        # High confidence item should not be labelled uncertain
        self.assertNotIn("uncertain", intent_items[0].statement.lower())

    def test_intent_low_confidence_labelled_uncertain(self):
        ctx = AnalyzerContext()
        ctx.intent_label = "creative_writing"
        ctx.intent_confidence = 0.30
        ctx.intent_available = True
        items = build_deterministic_evidence(ctx)
        intent_items = [i for i in items if i.source == EvidenceSource.INTENT]
        self.assertEqual(len(intent_items), 1)
        self.assertIn("uncertain", intent_items[0].statement.lower())

    def test_empty_context_returns_minimal_evidence(self):
        ctx = AnalyzerContext()
        items = build_deterministic_evidence(ctx)
        # Word count 0 → no word count evidence
        prep_word_items = [
            i for i in items
            if i.source == EvidenceSource.PREPROCESSING and "word" in i.statement.lower()
        ]
        self.assertEqual(len(prep_word_items), 0)

    def test_no_llm_source_in_deterministic_evidence(self):
        ctx = _make_full_context()
        items = build_deterministic_evidence(ctx)
        llm_items = [i for i in items if i.source == EvidenceSource.LLM]
        self.assertEqual(len(llm_items), 0)

    def test_scoring_dimension_evidence(self):
        ctx = _make_full_context()
        items = build_deterministic_evidence(ctx)
        scoring_items = [i for i in items if i.source == EvidenceSource.SCORING]
        self.assertGreater(len(scoring_items), 0)


class TestBuildLLMEvidenceItems(unittest.TestCase):
    def test_ambiguity_creates_evidence(self):
        items = build_llm_evidence_items(
            ambiguities=[{"issue": "Vague term", "evidence": "'soon'", "confidence": 0.75}],
            missing_info=[],
            contradictions=[],
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source, EvidenceSource.LLM)
        self.assertIn("Vague term", items[0].statement)

    def test_missing_info_creates_evidence(self):
        items = build_llm_evidence_items(
            ambiguities=[],
            missing_info=[{"item": "Target audience", "why_it_matters": "x", "confidence": 0.70}],
            contradictions=[],
        )
        self.assertEqual(len(items), 1)
        self.assertIn("Target audience", items[0].statement)

    def test_contradiction_creates_evidence(self):
        items = build_llm_evidence_items(
            ambiguities=[],
            missing_info=[],
            contradictions=[{"issue": "Conflicting formats", "evidence": "JSON vs text", "confidence": 0.80}],
        )
        self.assertEqual(len(items), 1)
        self.assertIn("Conflicting formats", items[0].statement)

    def test_confidence_clamped(self):
        items = build_llm_evidence_items(
            ambiguities=[{"issue": "X", "evidence": "Y", "confidence": 0.0}],
            missing_info=[],
            contradictions=[],
        )
        self.assertGreaterEqual(items[0].confidence, 0.10)

    def test_empty_issue_skipped(self):
        items = build_llm_evidence_items(
            ambiguities=[{"issue": "", "evidence": "Y", "confidence": 0.5}],
            missing_info=[],
            contradictions=[],
        )
        self.assertEqual(len(items), 0)

    def test_all_sources_are_llm(self):
        items = build_llm_evidence_items(
            ambiguities=[{"issue": "A", "evidence": "B", "confidence": 0.6}],
            missing_info=[{"item": "C", "why_it_matters": "D", "confidence": 0.6}],
            contradictions=[{"issue": "E", "evidence": "F", "confidence": 0.6}],
        )
        for item in items:
            self.assertEqual(item.source, EvidenceSource.LLM)
            self.assertEqual(item.type, EvidenceType.INFERENCE)


if __name__ == "__main__":
    unittest.main()
