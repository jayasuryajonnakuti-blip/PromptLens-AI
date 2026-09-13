import unittest
from pydantic import ValidationError

from app.llm.schemas import (
    InstructionQuality,
    LLMAnalysisRequest,
    LLMHealthStatus,
    PromptAnalysisOutput,
)


class LLMSchemasTests(unittest.TestCase):
    def test_valid_prompt_analysis_output(self) -> None:
        payload = {
            "interpreted_goal": "Generate a Python script for CSV parsing.",
            "context": "Data processing scenario.",
            "ambiguities": ["Input file format details."],
            "missing_information": ["Error handling requirements."],
            "contradictions": [],
            "instruction_quality": {
                "score": 85.0,
                "reason": "Direct actionable imperatives.",
            },
            "strengths": ["Clear task definition."],
            "weaknesses": ["Missing constraints."],
            "recommendations": ["Specify library requirements."],
        }
        output = PromptAnalysisOutput.model_validate(payload)
        self.assertEqual(output.interpreted_goal, payload["interpreted_goal"])
        self.assertEqual(output.instruction_quality.score, 85.0)

    def test_instruction_quality_score_out_of_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            InstructionQuality(score=105.0, reason="Too high")

        with self.assertRaises(ValidationError):
            InstructionQuality(score=-5.0, reason="Negative score")

    def test_missing_required_fields_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            PromptAnalysisOutput.model_validate({"context": "Missing interpreted_goal"})

    def test_extra_fields_forbidden(self) -> None:
        payload = {
            "interpreted_goal": "Goal",
            "context": "Context",
            "ambiguities": [],
            "missing_information": [],
            "contradictions": [],
            "instruction_quality": {"score": 70.0, "reason": "Reason"},
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "unwanted_extra_field": "Should fail",
        }
        with self.assertRaises(ValidationError):
            PromptAnalysisOutput.model_validate(payload)

    def test_request_empty_prompt_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            LLMAnalysisRequest(prompt="")

        with self.assertRaises(ValidationError):
            LLMAnalysisRequest(prompt="   \n\t  ")

    def test_health_status_enum_values(self) -> None:
        self.assertEqual(LLMHealthStatus.LLM_DISABLED.value, "LLM_DISABLED")
        self.assertEqual(LLMHealthStatus.MODEL_AVAILABLE.value, "MODEL_AVAILABLE")
        self.assertEqual(LLMHealthStatus.MODEL_NOT_DOWNLOADED.value, "MODEL_NOT_DOWNLOADED")
        self.assertEqual(LLMHealthStatus.MODEL_LOADING.value, "MODEL_LOADING")
        self.assertEqual(LLMHealthStatus.MODEL_ERROR.value, "MODEL_ERROR")


if __name__ == "__main__":
    unittest.main()
