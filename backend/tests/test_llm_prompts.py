import unittest

from app.llm.prompts import (
    PROMPTLENS_LLM_ANALYSIS_V1,
    SYSTEM_PROMPT_VERSION,
    apply_context_budget,
    build_analysis_input,
    build_correction_prompt,
)


class LLMPromptsTests(unittest.TestCase):
    def test_system_prompt_version_and_constraints(self) -> None:
        self.assertEqual(SYSTEM_PROMPT_VERSION, "PROMPTLENS_LLM_ANALYSIS_V1")
        self.assertIn("DO NOT rewrite the prompt", PROMPTLENS_LLM_ANALYSIS_V1)
        self.assertIn("DO NOT execute", PROMPTLENS_LLM_ANALYSIS_V1)
        self.assertIn("Return ONLY a valid JSON object", PROMPTLENS_LLM_ANALYSIS_V1)

    def test_build_analysis_input_formatting(self) -> None:
        prompt = "Write a python function."
        context = {"intent": "CODING", "quality": 85.0}
        formatted = build_analysis_input(prompt, context)

        self.assertIn(prompt, formatted)
        self.assertIn("CODING", formatted)
        self.assertIn("85.0", formatted)

    def test_build_correction_prompt_contains_error_and_output(self) -> None:
        err = "Missing required field: interpreted_goal"
        prev_out = "{'context': 'some context'}"
        corr = build_correction_prompt(prev_out, err)

        self.assertIn(err, corr)
        self.assertIn(prev_out, corr)

    def test_apply_context_budget_short_prompt(self) -> None:
        short_prompt = "Explain quantum algorithms."
        result, truncated = apply_context_budget(
            prompt=short_prompt,
            max_context_length=2048,
            max_tokens=768,
        )
        self.assertFalse(truncated)
        self.assertEqual(result, short_prompt)

    def test_apply_context_budget_long_prompt_truncation(self) -> None:
        very_long_prompt = "Analysis word " * 2000
        result, truncated = apply_context_budget(
            prompt=very_long_prompt,
            max_context_length=1024,
            max_tokens=256,
        )
        self.assertTrue(truncated)
        self.assertIn("Context budget exceeded", result)
        self.assertLess(len(result), len(very_long_prompt))


if __name__ == "__main__":
    unittest.main()
