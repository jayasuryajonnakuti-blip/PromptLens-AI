"""Tests for PromptLens AI Validator deterministic checks (Step 18)."""
import unittest

from app.validator.checks import (
    check_contradictions,
    check_critic_consistency,
    check_field_integrity,
    check_intent_preservation,
    check_length_bloat,
    check_negative_constraints,
    check_output_format_preservation,
    check_requirement_preservation,
    check_score_regression,
    check_unsupported_additions,
    run_deterministic_validation,
)
from app.validator.context import ValidatorContext
from app.validator.schemas import IssueSeverity, ValidatorDecision


class TestValidatorChecks(unittest.TestCase):
    def test_field_integrity_valid(self):
        ctx = ValidatorContext(
            original_prompt="Valid original prompt.",
            optimized_prompt="Valid optimized prompt.",
            original_score=75.0,
            optimized_score=85.0,
            semantic_similarity=0.90,
        )
        issues = check_field_integrity(ctx)
        self.assertEqual(len(issues), 0)

    def test_field_integrity_invalid_score(self):
        ctx = ValidatorContext(
            original_prompt="Valid",
            optimized_prompt="Valid",
            original_score=-5.0,
            optimized_score=85.0,
            semantic_similarity=0.9,
        )
        issues = check_field_integrity(ctx)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "INVALID_SCORE_RANGE")
        self.assertEqual(issues[0].severity, IssueSeverity.ERROR)

    def test_intent_preservation_pass(self):
        ctx = ValidatorContext(
            original_intent="code_generation",
            optimized_intent="code_generation",
            original_intent_confidence=0.9,
            optimized_intent_confidence=0.88,
            intent_match=True,
            similarity_available=True,
            semantic_similarity=0.92,
        )
        issues = check_intent_preservation(ctx)
        self.assertEqual(len(issues), 0)

    def test_intent_regression_detected(self):
        ctx = ValidatorContext(
            original_intent="code_generation",
            optimized_intent="creative_writing",
            original_intent_confidence=0.85,
            optimized_intent_confidence=0.82,
            intent_match=False,
            similarity_available=True,
            semantic_similarity=0.80,
        )
        issues = check_intent_preservation(ctx)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "INTENT_REGRESSION")
        self.assertEqual(issues[0].severity, IssueSeverity.ERROR)

    def test_fatal_semantic_divergence_detected(self):
        ctx = ValidatorContext(
            original_intent="code_generation",
            optimized_intent="code_generation",
            intent_match=True,
            similarity_available=True,
            semantic_similarity=0.20,
        )
        issues = check_intent_preservation(ctx)
        self.assertTrue(any(i.code == "FATAL_SEMANTIC_DIVERGENCE" for i in issues))

    def test_requirement_preservation_pass(self):
        orig = "You must include error handling and return a dictionary."
        opt = "You must include error handling for all functions and return a dictionary."
        issues = check_requirement_preservation(orig, opt)
        self.assertEqual(len(issues), 0)

    def test_requirement_preservation_lost(self):
        orig = "You must include error handling and return a dictionary."
        opt = "Write a basic script without error handling."
        issues = check_requirement_preservation(orig, opt)
        self.assertGreater(len(issues), 0)
        self.assertEqual(issues[0].code, "LOST_EXPLICIT_REQUIREMENT")
        self.assertEqual(issues[0].severity, IssueSeverity.ERROR)

    def test_negative_constraint_preservation_pass(self):
        orig = "Write a script. Avoid using synchronous network requests."
        opt = "Write a script. Avoid using synchronous network requests by using httpx."
        issues = check_negative_constraints(orig, opt)
        self.assertEqual(len(issues), 0)

    def test_negative_constraint_lost(self):
        orig = "Write an API client. Avoid using synchronous network requests."
        opt = "Write an API client using the standard urllib library."
        issues = check_negative_constraints(orig, opt)
        self.assertGreater(len(issues), 0)
        self.assertEqual(issues[0].code, "LOST_NEGATIVE_CONSTRAINT")
        self.assertEqual(issues[0].severity, IssueSeverity.ERROR)

    def test_output_format_preservation_pass(self):
        ctx = ValidatorContext(
            original_formats=["json", "table"],
            optimized_formats=["json", "table", "markdown"],
        )
        issues = check_output_format_preservation(ctx)
        self.assertEqual(len(issues), 0)

    def test_output_format_lost(self):
        ctx = ValidatorContext(
            original_formats=["json"],
            optimized_formats=["markdown"],
        )
        issues = check_output_format_preservation(ctx)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "LOST_OUTPUT_FORMAT")
        self.assertEqual(issues[0].severity, IssueSeverity.ERROR)

    def test_contradiction_detection(self):
        opt = "Return JSON only. Include a detailed conversational explanation describing your thought process."
        issues = check_contradictions(opt)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "CONTRADICTORY_FORMAT_DIRECTIVE")
        self.assertEqual(issues[0].severity, IssueSeverity.ERROR)

    def test_unsupported_tech_additions(self):
        orig = "Write a simple program to manage tasks."
        opt = "Write a program to manage tasks. You must use Django and deploy on AWS with PostgreSQL."
        issues = check_unsupported_additions(orig, opt)
        self.assertGreater(len(issues), 0)
        self.assertTrue(any(i.code == "UNSUPPORTED_MANDATORY_TECH" for i in issues))

    def test_score_regression_fail(self):
        ctx = ValidatorContext(
            original_score=85.0,
            optimized_score=68.0,
            score_delta=-17.0,
            scoring_available=True,
        )
        issues = check_score_regression(ctx)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "SEVERE_SCORE_REGRESSION")
        self.assertEqual(issues[0].severity, IssueSeverity.ERROR)

    def test_length_bloat_fail(self):
        ctx = ValidatorContext(
            original_word_count=10,
            optimized_word_count=65,
            expansion_ratio=6.5,
        )
        issues = check_length_bloat(ctx)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "EXCESSIVE_LENGTH_BLOAT")
        self.assertEqual(issues[0].severity, IssueSeverity.ERROR)

    def test_critic_fail_detected(self):
        ctx = ValidatorContext(
            critic_provided=True,
            critic_decision="FAIL",
            critic_score=40.0,
            critic_lost_requirements=["output format JSON"],
        )
        issues = check_critic_consistency(ctx)
        self.assertTrue(any(i.code == "CRITIC_FAIL_UNRESOLVED" for i in issues))
        self.assertTrue(any(i.code == "CRITIC_REPORTED_LOST_REQUIREMENT" for i in issues))

    def test_critic_needs_review_detected(self):
        ctx = ValidatorContext(
            critic_provided=True,
            critic_decision="NEEDS_REVIEW",
            critic_score=65.0,
        )
        issues = check_critic_consistency(ctx)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "CRITIC_FLAGGED_REVIEW")
        self.assertEqual(issues[0].severity, IssueSeverity.WARNING)

    def test_run_deterministic_validation_pass(self):
        ctx = ValidatorContext(
            original_prompt="Write a Python script to sort an array of integers.",
            optimized_prompt="Write a clean Python function to sort an array of integers in ascending order.",
            original_word_count=9,
            optimized_word_count=13,
            expansion_ratio=1.44,
            original_intent="code_generation",
            optimized_intent="code_generation",
            intent_match=True,
            original_intent_confidence=0.88,
            optimized_intent_confidence=0.89,
            similarity_available=True,
            semantic_similarity=0.92,
            scoring_available=True,
            original_score=72.0,
            optimized_score=84.0,
            score_delta=12.0,
        )
        res = run_deterministic_validation(ctx)
        self.assertEqual(res.decision, ValidatorDecision.PASS)
        self.assertGreaterEqual(res.safety_score, 80.0)
        self.assertEqual(len(res.failed_checks), 0)

    def test_run_deterministic_validation_fail_on_fatal_issue(self):
        ctx = ValidatorContext(
            original_prompt="Write a program and return JSON only.",
            optimized_prompt="Write a program and return JSON only, along with a detailed conversational prose breakdown.",
            original_word_count=7,
            optimized_word_count=14,
            expansion_ratio=2.0,
            intent_match=True,
            original_score=70.0,
            optimized_score=70.0,
            score_delta=0.0,
            semantic_similarity=0.88,
        )
        res = run_deterministic_validation(ctx)
        self.assertEqual(res.decision, ValidatorDecision.FAIL)
        self.assertIn("contradiction_detection", res.failed_checks)


if __name__ == "__main__":
    unittest.main()
