"""Tests for PromptLens Analyzer schemas (Step 15)."""
import unittest

from pydantic import ValidationError

from app.analyzer.schemas import (
    AnalysisMode,
    AnalyzerAnalysis,
    AnalyzerMetadata,
    AnalyzerRequest,
    AnalyzerResponse,
    AmbiguityItem,
    ContradictionItem,
    EvidenceItem,
    EvidenceSource,
    EvidenceType,
    InstructionQualityResult,
    IntentResult,
    MissingInfoItem,
    RecommendationItem,
    RecommendationPriority,
    StrengthItem,
    WeaknessItem,
)


class TestAnalyzerRequest(unittest.TestCase):
    def test_valid_prompt(self):
        req = AnalyzerRequest(prompt="Analyze this prompt for quality.")
        self.assertEqual(req.prompt, "Analyze this prompt for quality.")

    def test_blank_prompt_raises(self):
        with self.assertRaises(ValidationError):
            AnalyzerRequest(prompt="   ")

    def test_empty_prompt_raises(self):
        with self.assertRaises(ValidationError):
            AnalyzerRequest(prompt="")

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            AnalyzerRequest(prompt="test", unknown_field="bad")


class TestEvidenceItem(unittest.TestCase):
    def test_valid_evidence(self):
        ev = EvidenceItem(
            source=EvidenceSource.PREPROCESSING,
            type=EvidenceType.OBSERVATION,
            statement="Word count is 50.",
            confidence=0.95,
        )
        self.assertEqual(ev.source, EvidenceSource.PREPROCESSING)
        self.assertEqual(ev.type, EvidenceType.OBSERVATION)
        self.assertAlmostEqual(ev.confidence, 0.95)

    def test_confidence_below_zero_raises(self):
        with self.assertRaises(ValidationError):
            EvidenceItem(
                source=EvidenceSource.NLP,
                type=EvidenceType.INFERENCE,
                statement="Invalid.",
                confidence=-0.1,
            )

    def test_confidence_above_one_raises(self):
        with self.assertRaises(ValidationError):
            EvidenceItem(
                source=EvidenceSource.LLM,
                type=EvidenceType.INFERENCE,
                statement="Invalid.",
                confidence=1.1,
            )

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            EvidenceItem(
                source=EvidenceSource.SCORING,
                type=EvidenceType.OBSERVATION,
                statement="ok",
                confidence=0.5,
                extra="bad",
            )


class TestSubItems(unittest.TestCase):
    def test_ambiguity_item(self):
        a = AmbiguityItem(issue="Vague term", evidence="'soon' is ambiguous", confidence=0.7)
        self.assertEqual(a.issue, "Vague term")

    def test_missing_info_item(self):
        m = MissingInfoItem(item="Target format", why_it_matters="Format affects output", confidence=0.65)
        self.assertEqual(m.item, "Target format")

    def test_contradiction_item(self):
        c = ContradictionItem(issue="Conflict", evidence="Instruction A vs B", confidence=0.80)
        self.assertEqual(c.issue, "Conflict")

    def test_strength_item(self):
        s = StrengthItem(strength="Clear goal", evidence="Explicit task statement.")
        self.assertEqual(s.strength, "Clear goal")

    def test_weakness_item(self):
        w = WeaknessItem(weakness="No constraints", evidence="Preprocessing: no constraint language.")
        self.assertEqual(w.weakness, "No constraints")

    def test_recommendation_item_high(self):
        r = RecommendationItem(
            recommendation="Add examples",
            reason="Improves specificity",
            priority=RecommendationPriority.HIGH,
        )
        self.assertEqual(r.priority, RecommendationPriority.HIGH)

    def test_recommendation_priorities(self):
        for p in ("HIGH", "MEDIUM", "LOW"):
            r = RecommendationItem(
                recommendation="X",
                reason="Y",
                priority=RecommendationPriority(p),
            )
            self.assertEqual(r.priority.value, p)


class TestInstructionQuality(unittest.TestCase):
    def test_valid_score(self):
        iq = InstructionQualityResult(score=72.0, reason="Good phrasing.")
        self.assertAlmostEqual(iq.score, 72.0)

    def test_score_below_zero_raises(self):
        with self.assertRaises(ValidationError):
            InstructionQualityResult(score=-1.0, reason="Bad")

    def test_score_above_100_raises(self):
        with self.assertRaises(ValidationError):
            InstructionQualityResult(score=101.0, reason="Bad")


class TestAnalyzerMetadata(unittest.TestCase):
    def test_all_modes(self):
        for mode in AnalysisMode:
            meta = AnalyzerMetadata(
                analyzer_version="1.0.0",
                llm_model="test-model",
                analysis_mode=mode,
            )
            self.assertEqual(meta.analysis_mode, mode)

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            AnalyzerMetadata(
                analyzer_version="1.0.0",
                llm_model="test",
                analysis_mode=AnalysisMode.UNAVAILABLE,
                extra="bad",
            )


class TestAnalyzerAnalysis(unittest.TestCase):
    def _make_analysis(self, mode=AnalysisMode.UNAVAILABLE) -> AnalyzerAnalysis:
        return AnalyzerAnalysis(
            interpreted_goal="Test goal.",
            intent=IntentResult(label="code_generation", confidence=0.80),
            context_summary="A test prompt.",
            instruction_quality=InstructionQualityResult(score=65.0, reason="Adequate."),
            evidence=[
                EvidenceItem(
                    source=EvidenceSource.PREPROCESSING,
                    type=EvidenceType.OBSERVATION,
                    statement="Word count: 10.",
                    confidence=0.95,
                )
            ],
            metadata=AnalyzerMetadata(
                analyzer_version="1.0.0",
                llm_model="none",
                analysis_mode=mode,
            ),
        )

    def test_minimal_valid_analysis(self):
        a = self._make_analysis()
        self.assertEqual(a.interpreted_goal, "Test goal.")
        self.assertEqual(a.analysis_mode if hasattr(a, "analysis_mode") else a.metadata.analysis_mode, AnalysisMode.UNAVAILABLE)

    def test_empty_lists_default(self):
        a = self._make_analysis()
        self.assertEqual(a.ambiguities, [])
        self.assertEqual(a.missing_information, [])
        self.assertEqual(a.contradictions, [])
        self.assertEqual(a.strengths, [])
        self.assertEqual(a.weaknesses, [])
        self.assertEqual(a.recommendations, [])

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            AnalyzerAnalysis(
                interpreted_goal="X",
                intent=IntentResult(label="other", confidence=0.5),
                context_summary="X",
                instruction_quality=InstructionQualityResult(score=50.0, reason="ok"),
                evidence=[],
                metadata=AnalyzerMetadata(
                    analyzer_version="1.0.0",
                    llm_model="none",
                    analysis_mode=AnalysisMode.UNAVAILABLE,
                ),
                unknown_extra="bad",
            )


class TestAnalyzerResponse(unittest.TestCase):
    def _make_response(self) -> AnalyzerResponse:
        analysis = AnalyzerAnalysis(
            interpreted_goal="Goal.",
            intent=IntentResult(label="summarization", confidence=0.75),
            context_summary="Context.",
            instruction_quality=InstructionQualityResult(score=60.0, reason="Fair."),
            evidence=[],
            metadata=AnalyzerMetadata(
                analyzer_version="1.0.0",
                llm_model="none",
                analysis_mode=AnalysisMode.UNAVAILABLE,
            ),
        )
        return AnalyzerResponse(
            analysis=analysis,
            prompt_length=42,
            step13_overall_score=55.3,
            step13_quality_category="FAIR",
        )

    def test_response_fields(self):
        resp = self._make_response()
        self.assertEqual(resp.prompt_length, 42)
        self.assertAlmostEqual(resp.step13_overall_score, 55.3, places=1)
        self.assertEqual(resp.step13_quality_category, "FAIR")
        self.assertIsInstance(resp.analysis, AnalyzerAnalysis)

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            AnalyzerResponse(
                analysis=self._make_response().analysis,
                prompt_length=10,
                step13_overall_score=50.0,
                step13_quality_category="GOOD",
                extra="bad",
            )


if __name__ == "__main__":
    unittest.main()
