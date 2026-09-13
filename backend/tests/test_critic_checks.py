"""Tests for PromptLens AI Critic deterministic evaluation checks (Step 17)."""
import unittest

from app.critic.checks import (
    check_contradictions,
    check_format_regression,
    check_intent_regression,
    check_negative_constraints,
    check_score_regression,
    check_unsupported_technical_assumptions,
    check_verbosity,
    run_deterministic_checks,
)
from app.critic.context import CriticContext
from app.critic.schemas import CriticDecision, IssueSeverity


class TestDeterministicChecks(unittest.TestCase):
    def test_intent_regression_detected(self):
        ctx = CriticContext()
        ctx.intent_original = "code_generation"
        ctx.intent_confidence_original = 0.85
        ctx.intent_optimized = "creative_writing"
        ctx.intent_confidence_optimized = 0.80
        ctx.intent_match = False

        issues = check_intent_regression(ctx, "Write code", "Write a story")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, IssueSeverity.ERROR)
        self.assertEqual(issues[0].type, "intent_regression")

    def test_semantic_divergence_detected(self):
        ctx = CriticContext()
        ctx.similarity_available = True
        ctx.semantic_similarity = 0.25

        issues = check_intent_regression(ctx, "Orig", "Completely different topic")
        self.assertTrue(any(i.type == "semantic_divergence" for i in issues))

    def test_format_regression_detected(self):
        ctx = CriticContext()
        ctx.formats_original = ["json", "csv"]
        ctx.formats_optimized = ["csv"]  # lost json

        issues, lost_reqs = check_format_regression(ctx)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, IssueSeverity.ERROR)
        self.assertIn("Output format 'json'", lost_reqs)

    def test_format_preserved_no_issue(self):
        ctx = CriticContext()
        ctx.formats_original = ["json"]
        ctx.formats_optimized = ["json", "markdown"]

        issues, lost_reqs = check_format_regression(ctx)
        self.assertEqual(len(issues), 0)
        self.assertEqual(len(lost_reqs), 0)

    def test_negative_constraint_loss_detected(self):
        orig = "Write a login endpoint. Avoid using synchronous libraries."
        opt = "Write a login endpoint with FastAPI."  # dropped synchronous constraint

        issues, lost_reqs = check_negative_constraints(orig, opt)
        self.assertGreater(len(issues), 0)
        self.assertEqual(issues[0].severity, IssueSeverity.WARNING)
        self.assertTrue(any("synchronous" in r.lower() for r in lost_reqs))

    def test_score_regression_critical(self):
        ctx = CriticContext()
        ctx.scoring_available = True
        ctx.overall_score_original = 85.0
        ctx.overall_score_optimized = 60.0
        ctx.overall_score_delta = -25.0

        issues = check_score_regression(ctx)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, IssueSeverity.ERROR)
        self.assertEqual(issues[0].type, "score_regression_critical")

    def test_score_regression_moderate(self):
        ctx = CriticContext()
        ctx.scoring_available = True
        ctx.overall_score_original = 80.0
        ctx.overall_score_optimized = 72.0
        ctx.overall_score_delta = -8.0

        issues = check_score_regression(ctx)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, IssueSeverity.WARNING)
        self.assertEqual(issues[0].type, "score_regression_moderate")

    def test_verbosity_excessive_expansion(self):
        ctx = CriticContext()
        ctx.word_count_original = 10
        ctx.word_count_optimized = 85
        ctx.word_count_ratio = 8.5

        issues = check_verbosity(ctx)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, IssueSeverity.ERROR)
        self.assertEqual(issues[0].type, "excessive_verbosity")

    def test_contradiction_detected(self):
        orig = "Return user data."
        opt = "Return JSON only. Include a detailed step by step explanation in conversational prose."

        issues = check_contradictions(orig, opt)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, IssueSeverity.ERROR)
        self.assertEqual(issues[0].type, "introduced_contradiction")

    def test_unsupported_tech_assumption_detected(self):
        orig = "Write a web service to store documents."
        opt = "Write a web service to store documents using Django and PostgreSQL."

        issues, intro_reqs, unsupported = check_unsupported_technical_assumptions(orig, opt)
        self.assertGreater(len(issues), 0)
        self.assertTrue(any("postgresql" in r.lower() or "django" in r.lower() for r in intro_reqs))

    def test_run_deterministic_checks_pass(self):
        ctx = CriticContext()
        ctx.intent_original = "code_generation"
        ctx.intent_optimized = "code_generation"
        ctx.intent_match = True
        ctx.semantic_similarity = 0.92
        ctx.similarity_available = True
        ctx.scoring_available = True
        ctx.overall_score_original = 70.0
        ctx.overall_score_optimized = 82.0
        ctx.overall_score_delta = 12.0
        ctx.word_count_ratio = 1.3

        orig = "Write a Python script to sort numbers."
        opt = "Write a clean Python script to sort numbers in ascending order. Return a list."

        res = run_deterministic_checks(ctx, orig, opt)
        self.assertEqual(res.suggested_decision, CriticDecision.PASS)
        self.assertGreaterEqual(res.base_critique_score, 70.0)

    def test_run_deterministic_checks_fail_on_critical_issue(self):
        ctx = CriticContext()
        ctx.intent_original = "code_generation"
        ctx.intent_optimized = "creative_writing"
        ctx.intent_confidence_original = 0.88
        ctx.intent_confidence_optimized = 0.85
        ctx.intent_match = False
        ctx.formats_original = ["json"]
        ctx.formats_optimized = []

        res = run_deterministic_checks(ctx, "Write code returning JSON", "Write a poem")
        self.assertEqual(res.suggested_decision, CriticDecision.FAIL)


if __name__ == "__main__":
    unittest.main()
