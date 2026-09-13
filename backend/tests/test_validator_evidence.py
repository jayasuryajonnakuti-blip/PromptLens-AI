"""Tests for PromptLens AI Validator evidence system (Step 18)."""
import unittest

from app.validator.context import ValidatorContext
from app.validator.evidence import (
    build_llm_validation_evidence,
    build_validation_evidence,
)
from app.validator.schemas import EvidenceSource, IssueSeverity, ValidationIssue


class TestValidatorEvidence(unittest.TestCase):
    def test_build_validation_evidence_sources(self):
        ctx = ValidatorContext(
            original_word_count=15,
            optimized_word_count=22,
            expansion_ratio=1.47,
            original_formats=["json"],
            optimized_formats=["json"],
            original_intent="code_generation",
            optimized_intent="code_generation",
            semantic_similarity=0.91,
            scoring_available=True,
            original_score=70.0,
            optimized_score=82.0,
            score_delta=12.0,
            critic_provided=True,
            critic_decision="PASS",
            critic_score=85.0,
        )
        issues = [
            ValidationIssue(
                code="MINOR_EXPANSION",
                severity=IssueSeverity.INFO,
                message="Slight word count increase.",
                field_or_scope="length",
                evidence="ratio: 1.47",
            )
        ]
        evidence = build_validation_evidence(ctx, issues)
        self.assertGreater(len(evidence), 0)

        sources = {e.source for e in evidence}
        self.assertIn(EvidenceSource.DETERMINISTIC, sources)
        self.assertIn(EvidenceSource.NLP, sources)
        self.assertIn(EvidenceSource.EMBEDDING, sources)
        self.assertIn(EvidenceSource.SCORING, sources)
        self.assertIn(EvidenceSource.CRITIC, sources)

    def test_semantic_similarity_disclaimer_present(self):
        ctx = ValidatorContext(semantic_similarity=0.87)
        evidence = build_validation_evidence(ctx, [])

        emb_items = [e for e in evidence if e.source == EvidenceSource.EMBEDDING]
        self.assertGreater(len(emb_items), 0)
        desc = emb_items[0].description
        self.assertIn("Semantic similarity is supporting evidence", desc)
        self.assertIn("not, by itself, proof of intent preservation", desc)

    def test_llm_validation_evidence(self):
        llm_data = {
            "safety_score": 88.0,
            "justification": "Semantic intent and constraints are rigorously preserved.",
        }
        evidence = build_llm_validation_evidence(llm_data)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].source, EvidenceSource.LLM)
        self.assertAlmostEqual(evidence[0].confidence, 0.70)


if __name__ == "__main__":
    unittest.main()
