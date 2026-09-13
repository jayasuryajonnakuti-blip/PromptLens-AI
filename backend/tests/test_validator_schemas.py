"""Tests for PromptLens AI Validator Pydantic schemas (Step 18)."""
import unittest

from pydantic import ValidationError

from app.validator.schemas import (
    CriticResultInput,
    EvidenceSource,
    IssueSeverity,
    OptimizationMetadataInput,
    PromptMetadata,
    ValidationEvidence,
    ValidationIssue,
    ValidationMetadata,
    ValidationMode,
    ValidationResult,
    ValidatorDecision,
    ValidatorRequest,
    ValidatorResponse,
)


class TestValidatorSchemas(unittest.TestCase):
    def test_enums(self):
        self.assertEqual({d.value for d in ValidatorDecision}, {"PASS", "FAIL", "NEEDS_REVIEW"})
        self.assertEqual({m.value for m in ValidationMode}, {"LOCAL_LLM", "MOCK", "UNAVAILABLE"})
        self.assertEqual({s.value for s in IssueSeverity}, {"ERROR", "WARNING", "INFO"})
        self.assertEqual(
            {e.value for e in EvidenceSource},
            {"DETERMINISTIC", "NLP", "EMBEDDING", "SCORING", "CRITIC", "LLM"},
        )

    def test_valid_request(self):
        req = ValidatorRequest(
            original_prompt="Write a Python script to sort a list.",
            optimized_prompt="Write a Python function to sort a list in ascending order.",
        )
        self.assertEqual(req.original_prompt, "Write a Python script to sort a list.")
        self.assertIsNone(req.critic_result)
        self.assertIsNone(req.optimization_metadata)

    def test_request_blank_original_prompt_raises(self):
        with self.assertRaises(ValidationError):
            ValidatorRequest(original_prompt="   ", optimized_prompt="Valid prompt")

    def test_request_empty_original_prompt_raises(self):
        with self.assertRaises(ValidationError):
            ValidatorRequest(original_prompt="", optimized_prompt="Valid prompt")

    def test_request_blank_optimized_prompt_raises(self):
        with self.assertRaises(ValidationError):
            ValidatorRequest(original_prompt="Valid prompt", optimized_prompt="   ")

    def test_request_empty_optimized_prompt_raises(self):
        with self.assertRaises(ValidationError):
            ValidatorRequest(original_prompt="Valid prompt", optimized_prompt="")

    def test_request_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            ValidatorRequest(
                original_prompt="orig",
                optimized_prompt="opt",
                forbidden_extra="bad",
            )

    def test_with_critic_result(self):
        critic_in = CriticResultInput(
            decision="PASS",
            overall_critique_score=85.0,
            issues=[{"type": "minor_warning", "description": "some issue"}],
            lost_requirements=[],
            introduced_requirements=[],
            unsupported_assumptions=[],
        )
        req = ValidatorRequest(
            original_prompt="orig",
            optimized_prompt="opt",
            critic_result=critic_in,
        )
        self.assertIsNotNone(req.critic_result)
        self.assertEqual(req.critic_result.decision, "PASS")

    def test_prompt_metadata_bounds(self):
        with self.assertRaises(ValidationError):
            PromptMetadata(
                prompt_length=-1,
                word_count=5,
                sentence_count=1,
                detected_intent="CODING",
                intent_confidence=0.9,
                quality_score=75.0,
            )
        with self.assertRaises(ValidationError):
            PromptMetadata(
                prompt_length=10,
                word_count=5,
                sentence_count=1,
                detected_intent="CODING",
                intent_confidence=1.5,  # > 1.0
                quality_score=75.0,
            )
        with self.assertRaises(ValidationError):
            PromptMetadata(
                prompt_length=10,
                word_count=5,
                sentence_count=1,
                detected_intent="CODING",
                intent_confidence=0.5,
                quality_score=105.0,  # > 100.0
            )

    def test_validation_evidence_bounds(self):
        ev = ValidationEvidence(
            source=EvidenceSource.EMBEDDING,
            description="Similarity check",
            metric_name="similarity",
            metric_value=0.88,
            confidence=0.95,
        )
        self.assertEqual(ev.source, EvidenceSource.EMBEDDING)
        with self.assertRaises(ValidationError):
            ValidationEvidence(
                source=EvidenceSource.EMBEDDING,
                description="test",
                confidence=1.5,
            )

    def test_validation_result_valid(self):
        orig_meta = PromptMetadata(
            prompt_length=30,
            word_count=6,
            sentence_count=1,
            detected_intent="CODING",
            intent_confidence=0.9,
            quality_score=70.0,
        )
        opt_meta = PromptMetadata(
            prompt_length=50,
            word_count=10,
            sentence_count=1,
            detected_intent="CODING",
            intent_confidence=0.92,
            quality_score=85.0,
        )
        val_meta = ValidationMetadata(
            validator_version="1.0.0",
            validation_mode=ValidationMode.UNAVAILABLE,
            model="none",
            latency_ms=10.5,
            critic_consistency_checked=False,
        )
        res = ValidationResult(
            decision=ValidatorDecision.PASS,
            is_valid=True,
            safety_score=92.0,
            original_metadata=orig_meta,
            optimized_metadata=opt_meta,
            issues=[],
            evidence=[],
            passed_checks=["schema_field_integrity"],
            failed_checks=[],
            metadata=val_meta,
        )
        self.assertTrue(res.is_valid)
        self.assertEqual(res.decision, ValidatorDecision.PASS)

    def test_validation_result_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            ValidationResult(
                decision=ValidatorDecision.PASS,
                is_valid=True,
                safety_score=90.0,
                original_metadata=PromptMetadata(
                    prompt_length=10, word_count=2, sentence_count=1,
                    detected_intent="CODING", intent_confidence=0.8, quality_score=70.0,
                ),
                optimized_metadata=PromptMetadata(
                    prompt_length=10, word_count=2, sentence_count=1,
                    detected_intent="CODING", intent_confidence=0.8, quality_score=70.0,
                ),
                metadata=ValidationMetadata(
                    validator_version="1.0.0", validation_mode=ValidationMode.UNAVAILABLE,
                    model="none", latency_ms=5.0, critic_consistency_checked=False,
                ),
                unexpected_field="disallowed",
            )


if __name__ == "__main__":
    unittest.main()
