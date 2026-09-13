"""Tests for ValidatorContext and context construction (Step 18)."""
import unittest

from app.validator.context import ValidatorContext, build_validator_context
from app.validator.schemas import CriticResultInput


class TestValidatorContext(unittest.TestCase):
    def test_default_values(self):
        ctx = ValidatorContext()
        self.assertEqual(ctx.original_word_count, 0)
        self.assertEqual(ctx.optimized_word_count, 0)
        self.assertAlmostEqual(ctx.expansion_ratio, 1.0)
        self.assertTrue(ctx.intent_match)
        self.assertFalse(ctx.critic_provided)

    def test_compact_dict_structure(self):
        ctx = ValidatorContext()
        d = ctx.to_compact_dict()
        self.assertIn("lexical", d)
        self.assertIn("intent", d)
        self.assertIn("semantic_similarity", d)
        self.assertIn("scoring", d)
        self.assertIn("formats", d)
        self.assertIn("critic", d)

    def test_build_validator_context_integration(self):
        orig = "Write a Python script to sort items in a list."
        opt = (
            "You are a senior Python developer. Write an efficient Python script to sort items in a list. "
            "Use standard sorting algorithms and return a sorted list."
        )
        ctx = build_validator_context(orig, opt)

        self.assertIsInstance(ctx, ValidatorContext)
        self.assertGreater(ctx.original_word_count, 0)
        self.assertGreater(ctx.optimized_word_count, ctx.original_word_count)
        self.assertGreater(ctx.expansion_ratio, 1.0)
        self.assertGreaterEqual(ctx.semantic_similarity, 0.0)
        self.assertLessEqual(ctx.semantic_similarity, 1.0)

        meta_orig = ctx.to_original_metadata()
        meta_opt = ctx.to_optimized_metadata()
        self.assertEqual(meta_orig.word_count, ctx.original_word_count)
        self.assertEqual(meta_opt.word_count, ctx.optimized_word_count)

    def test_build_validator_context_with_critic_result(self):
        orig = "Explain quantum computing."
        opt = "Explain quantum computing simply with 3 key principles."
        critic_in = CriticResultInput(
            decision="PASS",
            overall_critique_score=88.0,
            issues=[],
            lost_requirements=[],
            introduced_requirements=[],
            unsupported_assumptions=[],
        )
        ctx = build_validator_context(orig, opt, critic_result=critic_in)

        self.assertTrue(ctx.critic_provided)
        self.assertEqual(ctx.critic_decision, "PASS")
        self.assertAlmostEqual(ctx.critic_score, 88.0)


if __name__ == "__main__":
    unittest.main()
