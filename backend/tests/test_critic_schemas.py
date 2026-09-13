"""Tests for PromptLens AI Critic Pydantic schemas (Step 17)."""
import unittest

from pydantic import ValidationError

from app.critic.schemas import (
    CriticAnalysisMode,
    CriticDecision,
    CriticEvaluation,
    CriticEvidenceItem,
    CriticIssue,
    CriticMetadata,
    CriticRequest,
    CriticResponse,
    EvidenceSource,
    EvidenceType,
    IntentPreservationResult,
    IssueSeverity,
    MetricChange,
    OptimizationMetadataInput,
    QualityStatus,
    RequirementPreservationResult,
)


class TestCriticRequest(unittest.TestCase):
    def test_valid_request(self):
        req = CriticRequest(
            original_prompt="Summarize this article.",
            optimized_prompt="Summarize this article in 3 bullet points focusing on key takeaways.",
        )
        self.assertEqual(req.original_prompt, "Summarize this article.")
        self.assertEqual(
            req.optimized_prompt,
            "Summarize this article in 3 bullet points focusing on key takeaways.",
        )
        self.assertIsNone(req.optimization_metadata)

    def test_with_optimization_metadata(self):
        req = CriticRequest(
            original_prompt="Write a Python script.",
            optimized_prompt="Write a Python script to sort records.",
            optimization_metadata=OptimizationMetadataInput(
                optimization_mode="analytical",
                changes=[{"category": "structure", "description": "Added sort requirement."}],
                preserved_requirements=[{"requirement": "Python script", "reason": "Retained."}],
                placeholders_inserted=["[SORT_KEY]"],
            ),
        )
        self.assertIsNotNone(req.optimization_metadata)
        self.assertEqual(req.optimization_metadata.optimization_mode, "analytical")

    def test_blank_original_prompt_raises(self):
        with self.assertRaises(ValidationError):
            CriticRequest(original_prompt="   ", optimized_prompt="Valid prompt")

    def test_empty_original_prompt_raises(self):
        with self.assertRaises(ValidationError):
            CriticRequest(original_prompt="", optimized_prompt="Valid prompt")

    def test_blank_optimized_prompt_raises(self):
        with self.assertRaises(ValidationError):
            CriticRequest(original_prompt="Valid prompt", optimized_prompt="   ")

    def test_empty_optimized_prompt_raises(self):
        with self.assertRaises(ValidationError):
            CriticRequest(original_prompt="Valid prompt", optimized_prompt="")

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            CriticRequest(
                original_prompt="orig",
                optimized_prompt="opt",
                unexpected_field="disallowed",
            )


class TestCriticEnums(unittest.TestCase):
    def test_decision_enum_values(self):
        self.assertEqual(
            {d.value for d in CriticDecision},
            {"PASS", "FAIL", "NEEDS_REVIEW"},
        )

    def test_severity_enum_values(self):
        self.assertEqual(
            {s.value for s in IssueSeverity},
            {"ERROR", "WARNING", "INFO"},
        )

    def test_quality_status_enum_values(self):
        self.assertEqual(
            {q.value for q in QualityStatus},
            {"EXCELLENT", "STRONG", "GOOD", "FAIR", "POOR"},
        )

    def test_analysis_mode_enum_values(self):
        self.assertEqual(
            {m.value for m in CriticAnalysisMode},
            {"LOCAL_LLM", "MOCK", "UNAVAILABLE"},
        )


class TestMetricChange(unittest.TestCase):
    def test_valid_metric_change(self):
        mc = MetricChange(
            original_score=50.0,
            optimized_score=75.0,
            delta=25.0,
            assessment="Improved significantly.",
        )
        self.assertAlmostEqual(mc.delta, 25.0)

    def test_score_out_of_bounds_raises(self):
        with self.assertRaises(ValidationError):
            MetricChange(
                original_score=-5.0,
                optimized_score=75.0,
                delta=80.0,
                assessment="Bad",
            )
        with self.assertRaises(ValidationError):
            MetricChange(
                original_score=50.0,
                optimized_score=105.0,
                delta=55.0,
                assessment="Bad",
            )

    def test_delta_out_of_bounds_raises(self):
        with self.assertRaises(ValidationError):
            MetricChange(
                original_score=0.0,
                optimized_score=100.0,
                delta=105.0,
                assessment="Bad",
            )

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            MetricChange(
                original_score=50.0,
                optimized_score=70.0,
                delta=20.0,
                assessment="ok",
                extra="forbidden",
            )


class TestCriticIssue(unittest.TestCase):
    def test_valid_issue(self):
        issue = CriticIssue(
            type="lost_constraint",
            severity=IssueSeverity.WARNING,
            description="Negative constraint omitted.",
            evidence="Do not use sync libraries missing.",
        )
        self.assertEqual(issue.severity, IssueSeverity.WARNING)

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            CriticIssue(
                type="test",
                severity=IssueSeverity.ERROR,
                description="desc",
                evidence="ev",
                extra="forbidden",
            )


class TestCriticEvidenceItem(unittest.TestCase):
    def test_valid_evidence(self):
        ev = CriticEvidenceItem(
            source=EvidenceSource.EMBEDDING,
            type=EvidenceType.COMPARISON,
            statement="Semantic similarity is 0.88.",
            confidence=0.90,
        )
        self.assertEqual(ev.source, EvidenceSource.EMBEDDING)
        self.assertAlmostEqual(ev.confidence, 0.90)

    def test_confidence_bounds_enforced(self):
        with self.assertRaises(ValidationError):
            CriticEvidenceItem(
                source=EvidenceSource.ORIGINAL,
                type=EvidenceType.OBSERVATION,
                statement="test",
                confidence=-0.1,
            )
        with self.assertRaises(ValidationError):
            CriticEvidenceItem(
                source=EvidenceSource.ORIGINAL,
                type=EvidenceType.OBSERVATION,
                statement="test",
                confidence=1.1,
            )


class TestCriticEvaluation(unittest.TestCase):
    def _make_valid_evaluation(self) -> CriticEvaluation:
        return CriticEvaluation(
            decision=CriticDecision.PASS,
            overall_critique_score=82.0,
            intent_preservation=IntentPreservationResult(
                score=90.0,
                status=QualityStatus.EXCELLENT,
                reason="Intent preserved verbatim.",
            ),
            requirement_preservation=RequirementPreservationResult(
                score=85.0,
                status=QualityStatus.STRONG,
                reason="All constraints preserved.",
            ),
            clarity_change=MetricChange(
                original_score=60.0,
                optimized_score=80.0,
                delta=20.0,
                assessment="Improved.",
            ),
            specificity_change=MetricChange(
                original_score=55.0,
                optimized_score=75.0,
                delta=20.0,
                assessment="Improved.",
            ),
            ambiguity_change=MetricChange(
                original_score=65.0,
                optimized_score=85.0,
                delta=20.0,
                assessment="Improved.",
            ),
            completeness_change=MetricChange(
                original_score=50.0,
                optimized_score=70.0,
                delta=20.0,
                assessment="Improved.",
            ),
            issues=[],
            preserved_requirements=["Python endpoint", "JSON format"],
            lost_requirements=[],
            introduced_requirements=[],
            unsupported_assumptions=[],
            strengths=["Clearer structure"],
            weaknesses=[],
            recommendations=[],
            evidence=[
                CriticEvidenceItem(
                    source=EvidenceSource.ORIGINAL,
                    type=EvidenceType.OBSERVATION,
                    statement="Original prompt has 10 words.",
                    confidence=1.0,
                )
            ],
            metadata=CriticMetadata(
                critic_version="1.0.0",
                analysis_mode=CriticAnalysisMode.UNAVAILABLE,
                model="none",
                semantic_similarity=0.92,
                latency_ms=12.5,
            ),
        )

    def test_valid_evaluation(self):
        ev = self._make_valid_evaluation()
        self.assertEqual(ev.decision, CriticDecision.PASS)
        self.assertAlmostEqual(ev.overall_critique_score, 82.0)

    def test_extra_fields_forbidden(self):
        ev_dict = self._make_valid_evaluation().model_dump()
        ev_dict["disallowed_extra"] = "bad"
        with self.assertRaises(ValidationError):
            CriticEvaluation(**ev_dict)


class TestCriticResponse(unittest.TestCase):
    def test_valid_response(self):
        ev = TestCriticEvaluation()._make_valid_evaluation()
        resp = CriticResponse(
            evaluation=ev,
            original_prompt_length=45,
            optimized_prompt_length=65,
        )
        self.assertEqual(resp.original_prompt_length, 45)
        self.assertEqual(resp.optimized_prompt_length, 65)

    def test_extra_fields_forbidden(self):
        ev = TestCriticEvaluation()._make_valid_evaluation()
        with self.assertRaises(ValidationError):
            CriticResponse(
                evaluation=ev,
                original_prompt_length=45,
                optimized_prompt_length=65,
                forbidden="bad",
            )


if __name__ == "__main__":
    unittest.main()
