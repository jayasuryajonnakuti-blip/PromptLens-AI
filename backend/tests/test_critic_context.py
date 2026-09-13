"""Tests for CriticContext and build_critic_context (Step 17)."""
import unittest

from app.critic.context import CriticContext, build_critic_context


class TestCriticContextDefaults(unittest.TestCase):
    def test_default_values(self):
        ctx = CriticContext()
        self.assertEqual(ctx.word_count_original, 0)
        self.assertEqual(ctx.word_count_optimized, 0)
        self.assertAlmostEqual(ctx.word_count_ratio, 1.0)
        self.assertAlmostEqual(ctx.semantic_similarity, 1.0)
        self.assertTrue(ctx.intent_match)

    def test_compact_dict_structure(self):
        ctx = CriticContext()
        d = ctx.to_compact_dict()
        self.assertIn("lexical_comparison", d)
        self.assertIn("intent_comparison", d)
        self.assertIn("semantic_similarity", d)
        self.assertIn("scoring_comparison", d)
        self.assertIn("structural_comparison", d)


class TestCriticContextIntegration(unittest.TestCase):
    def test_build_critic_context_basic(self):
        orig = "Write a Python script to parse a CSV file."
        opt = (
            "You are a Python data engineer. Write a Python script to parse a CSV file. "
            "Use the standard csv module. Handle FileNotFoundError and return a dictionary."
        )
        ctx = build_critic_context(orig, opt)

        self.assertIsInstance(ctx, CriticContext)
        self.assertGreater(ctx.word_count_original, 0)
        self.assertGreater(ctx.word_count_optimized, ctx.word_count_original)
        self.assertGreater(ctx.word_count_ratio, 1.0)
        self.assertGreater(ctx.char_count_original, 0)
        self.assertGreater(ctx.char_count_optimized, ctx.char_count_original)
        self.assertGreaterEqual(ctx.semantic_similarity, 0.0)
        self.assertLessEqual(ctx.semantic_similarity, 1.0)

    def test_build_critic_context_role_detection(self):
        orig = "Write a function."
        opt = "You are an expert Python developer. Write a function."
        ctx = build_critic_context(orig, opt)
        self.assertFalse(ctx.has_role_original)
        self.assertTrue(ctx.has_role_optimized)

    def test_build_critic_context_constraint_detection(self):
        orig = "Write code. Avoid using synchronous network requests."
        opt = "Write code. Use requests library."
        ctx = build_critic_context(orig, opt)
        self.assertTrue(ctx.has_constraints_original)


if __name__ == "__main__":
    unittest.main()
