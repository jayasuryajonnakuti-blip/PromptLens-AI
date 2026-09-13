"""Tests for PromptLens AI Critic evidence generation (Step 17)."""
import unittest

from app.critic.context import CriticContext
from app.critic.evidence import build_deterministic_evidence, build_llm_evidence
from app.critic.schemas import CriticIssue, EvidenceSource, EvidenceType, IssueSeverity


class TestCriticEvidence(unittest.TestCase):
    def test_build_deterministic_evidence_content(self):
        ctx = CriticContext()
        ctx.word_count_original = 20
        ctx.word_count_optimized = 35
        ctx.char_count_original = 120
        ctx.char_count_optimized = 210
        ctx.formats_original = ["json"]
        ctx.formats_optimized = ["json"]
        ctx.intent_original = "code_generation"
        ctx.intent_optimized = "code_generation"
        ctx.semantic_similarity = 0.89
        ctx.scoring_available = True
        ctx.overall_score_original = 68.0
        ctx.overall_score_optimized = 80.0
        ctx.overall_score_delta = 12.0

        issues = [
            CriticIssue(
                type="test_warning",
                severity=IssueSeverity.WARNING,
                description="Minor phrasing warning",
                evidence="some evidence",
            )
        ]

        evidence = build_deterministic_evidence(ctx, "Orig", "Opt", issues)
        self.assertGreater(len(evidence), 0)

        sources = {e.source for e in evidence}
        self.assertIn(EvidenceSource.ORIGINAL, sources)
        self.assertIn(EvidenceSource.OPTIMIZED, sources)
        self.assertIn(EvidenceSource.INTENT, sources)
        self.assertIn(EvidenceSource.EMBEDDING, sources)
        self.assertIn(EvidenceSource.SCORING, sources)

    def test_semantic_similarity_disclaimer_present(self):
        ctx = CriticContext()
        ctx.semantic_similarity = 0.85
        evidence = build_deterministic_evidence(ctx, "Orig", "Opt", [])

        emb_items = [e for e in evidence if e.source == EvidenceSource.EMBEDDING]
        self.assertGreater(len(emb_items), 0)
        stmt = emb_items[0].statement
        self.assertIn("Semantic similarity is supporting evidence", stmt)
        self.assertIn("not, by itself, proof of intent preservation", stmt)

    def test_build_llm_evidence(self):
        llm_critique = {
            "issues": [
                {
                    "type": "unclear_instruction",
                    "description": "The third requirement is ambiguous.",
                }
            ]
        }
        llm_ev = build_llm_evidence(llm_critique)
        self.assertEqual(len(llm_ev), 1)
        self.assertEqual(llm_ev[0].source, EvidenceSource.LLM)
        self.assertEqual(llm_ev[0].type, EvidenceType.INFERENCE)
        self.assertAlmostEqual(llm_ev[0].confidence, 0.65)


if __name__ == "__main__":
    unittest.main()
