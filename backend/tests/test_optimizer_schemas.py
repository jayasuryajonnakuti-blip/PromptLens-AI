"""Tests for PromptLens Optimizer schemas (Step 16)."""
import unittest

from pydantic import ValidationError

from app.optimizer.schemas import (
    ChangeRecord,
    OptimizationMode,
    OptimizerMetadata,
    OptimizerMode,
    OptimizerRequest,
    OptimizerResponse,
    OptimizerResult,
    PreservedRequirement,
)


class TestOptimizerRequest(unittest.TestCase):
    def test_valid_prompt_default_mode(self):
        req = OptimizerRequest(prompt="Write a Python function to sort a list.")
        self.assertEqual(req.prompt, "Write a Python function to sort a list.")
        self.assertEqual(req.mode, OptimizationMode.BALANCED)

    def test_valid_prompt_all_modes(self):
        for mode in OptimizationMode:
            req = OptimizerRequest(prompt="Test", mode=mode)
            self.assertEqual(req.mode, mode)

    def test_blank_prompt_raises(self):
        with self.assertRaises(ValidationError):
            OptimizerRequest(prompt="   ")

    def test_empty_prompt_raises(self):
        with self.assertRaises(ValidationError):
            OptimizerRequest(prompt="")

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            OptimizerRequest(prompt="test", unknown="bad")

    def test_with_analyzer_result(self):
        req = OptimizerRequest(
            prompt="Test prompt",
            mode=OptimizationMode.ANALYTICAL,
            analyzer_result={"interpreted_goal": "Test"},
        )
        self.assertEqual(req.analyzer_result, {"interpreted_goal": "Test"})

    def test_without_analyzer_result_defaults_none(self):
        req = OptimizerRequest(prompt="Test")
        self.assertIsNone(req.analyzer_result)


class TestOptimizationModeEnum(unittest.TestCase):
    def test_all_modes_present(self):
        modes = {m.value for m in OptimizationMode}
        self.assertEqual(modes, {"balanced", "analytical", "creative", "expert"})

    def test_mode_string_values(self):
        self.assertEqual(OptimizationMode.BALANCED.value, "balanced")
        self.assertEqual(OptimizationMode.ANALYTICAL.value, "analytical")
        self.assertEqual(OptimizationMode.CREATIVE.value, "creative")
        self.assertEqual(OptimizationMode.EXPERT.value, "expert")


class TestOptimizerModeEnum(unittest.TestCase):
    def test_all_modes_present(self):
        modes = {m.value for m in OptimizerMode}
        self.assertEqual(modes, {"LOCAL_LLM", "MOCK", "UNAVAILABLE"})


class TestChangeRecord(unittest.TestCase):
    def test_valid_change(self):
        ch = ChangeRecord(category="clarity", description="Removed vague phrasing.")
        self.assertEqual(ch.category, "clarity")

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            ChangeRecord(category="clarity", description="ok", extra="bad")


class TestPreservedRequirement(unittest.TestCase):
    def test_valid(self):
        pr = PreservedRequirement(
            requirement="Return JSON only.",
            reason="Explicit constraint stated by user.",
        )
        self.assertEqual(pr.requirement, "Return JSON only.")

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            PreservedRequirement(requirement="X", reason="Y", extra="bad")


class TestOptimizerMetadata(unittest.TestCase):
    def test_all_optimizer_modes(self):
        for omode in OptimizerMode:
            meta = OptimizerMetadata(
                optimizer_version="1.0.0",
                llm_model="test",
                optimizer_mode=omode,
                optimization_mode=OptimizationMode.BALANCED,
            )
            self.assertEqual(meta.optimizer_mode, omode)

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            OptimizerMetadata(
                optimizer_version="1.0.0",
                llm_model="test",
                optimizer_mode=OptimizerMode.UNAVAILABLE,
                optimization_mode=OptimizationMode.BALANCED,
                extra="bad",
            )

    def test_default_latency(self):
        meta = OptimizerMetadata(
            optimizer_version="1.0.0",
            llm_model="none",
            optimizer_mode=OptimizerMode.UNAVAILABLE,
            optimization_mode=OptimizationMode.BALANCED,
        )
        self.assertEqual(meta.latency_ms, 0.0)


class TestOptimizerResult(unittest.TestCase):
    def _make_result(self) -> OptimizerResult:
        return OptimizerResult(
            optimized_prompt="Write a Python function to sort a list in ascending order.",
            summary="Added explicit ordering requirement.",
            changes=[ChangeRecord(category="specificity", description="Added ordering.")],
            preserved_requirements=[
                PreservedRequirement(requirement="Python function", reason="Original task.")
            ],
            placeholders_inserted=[],
            improvement_score_delta=8.0,
            metadata=OptimizerMetadata(
                optimizer_version="1.0.0",
                llm_model="none",
                optimizer_mode=OptimizerMode.MOCK,
                optimization_mode=OptimizationMode.BALANCED,
            ),
        )

    def test_valid_result(self):
        result = self._make_result()
        self.assertGreater(len(result.optimized_prompt), 0)
        self.assertAlmostEqual(result.improvement_score_delta, 8.0)

    def test_empty_lists_default(self):
        result = OptimizerResult(
            optimized_prompt="Test.",
            summary="No changes.",
            improvement_score_delta=0.0,
            metadata=OptimizerMetadata(
                optimizer_version="1.0.0",
                llm_model="none",
                optimizer_mode=OptimizerMode.UNAVAILABLE,
                optimization_mode=OptimizationMode.BALANCED,
            ),
        )
        self.assertEqual(result.changes, [])
        self.assertEqual(result.preserved_requirements, [])
        self.assertEqual(result.placeholders_inserted, [])

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            OptimizerResult(
                optimized_prompt="X",
                summary="Y",
                improvement_score_delta=0.0,
                metadata=OptimizerMetadata(
                    optimizer_version="1.0.0",
                    llm_model="none",
                    optimizer_mode=OptimizerMode.UNAVAILABLE,
                    optimization_mode=OptimizationMode.BALANCED,
                ),
                extra="bad",
            )


class TestOptimizerResponse(unittest.TestCase):
    def _make_response(self) -> OptimizerResponse:
        result = OptimizerResult(
            optimized_prompt="Improved prompt text.",
            summary="Two improvements applied.",
            improvement_score_delta=10.0,
            metadata=OptimizerMetadata(
                optimizer_version="1.0.0",
                llm_model="mock",
                optimizer_mode=OptimizerMode.MOCK,
                optimization_mode=OptimizationMode.EXPERT,
            ),
        )
        return OptimizerResponse(
            result=result,
            original_prompt_length=25,
            optimized_prompt_length=20,
            optimization_mode=OptimizationMode.EXPERT,
        )

    def test_response_fields(self):
        resp = self._make_response()
        self.assertEqual(resp.optimization_mode, OptimizationMode.EXPERT)
        self.assertEqual(resp.original_prompt_length, 25)
        self.assertEqual(resp.optimized_prompt_length, 20)
        self.assertIsInstance(resp.result, OptimizerResult)

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            resp = self._make_response()
            OptimizerResponse(
                result=resp.result,
                original_prompt_length=10,
                optimized_prompt_length=10,
                optimization_mode=OptimizationMode.BALANCED,
                extra="bad",
            )


if __name__ == "__main__":
    unittest.main()
