import unittest

from app.engines.nlp import analyze_prompt
from app.engines.preprocessing import preprocess_prompt
from app.ml.quality.categories import QualityLabel
from app.scoring.dimensions import DIMENSION_NAMES, evaluate_dimensions


class ScoringDimensionsTests(unittest.TestCase):
    def test_all_twelve_dimensions_present(self) -> None:
        prompt = "Explain quantum computing to high school students in 3 paragraphs."
        prep = preprocess_prompt(prompt)
        nlp = analyze_prompt(prompt)
        dims = evaluate_dimensions(prep, nlp)

        self.assertEqual(len(dims), 12)
        for expected_name in DIMENSION_NAMES:
            self.assertIn(expected_name, dims)

    def test_dimension_properties_and_bounds(self) -> None:
        prompt = "Act as a security engineer. Review the following Python code for SQL injection vulnerabilities. Return JSON."
        prep = preprocess_prompt(prompt)
        nlp = analyze_prompt(prompt)
        dims = evaluate_dimensions(prep, nlp)

        for name, eval_data in dims.items():
            self.assertGreaterEqual(eval_data.score, 0.0, f"Score under 0 for {name}")
            self.assertLessEqual(eval_data.score, 100.0, f"Score over 100 for {name}")
            self.assertIsInstance(eval_data.status, QualityLabel)
            self.assertTrue(len(eval_data.reason) > 0, f"Empty reason for {name}")
            self.assertTrue(len(eval_data.recommendation) > 0, f"Empty recommendation for {name}")

    def test_empty_prompt_dimensions(self) -> None:
        prep = preprocess_prompt("")
        nlp = analyze_prompt("")
        dims = evaluate_dimensions(prep, nlp)

        self.assertEqual(len(dims), 12)
        for name, eval_data in dims.items():
            self.assertEqual(eval_data.score, 0.0)
            self.assertEqual(eval_data.status, QualityLabel.POOR)

    def test_ambiguity_inverted_scoring(self) -> None:
        vague_prep = preprocess_prompt("Do it.")
        vague_nlp = analyze_prompt("Do it.")
        vague_dims = evaluate_dimensions(vague_prep, vague_nlp)

        clear_prompt = "Write a Python script that reads data.csv and calculates the average of column 'price'. Return code only."
        clear_prep = preprocess_prompt(clear_prompt)
        clear_nlp = analyze_prompt(clear_prompt)
        clear_dims = evaluate_dimensions(clear_prep, clear_nlp)

        # Higher ambiguity dimension score means LESS ambiguity (higher quality)
        self.assertGreater(clear_dims["ambiguity"].score, vague_dims["ambiguity"].score)

    def test_output_format_detection_boosts_dimension(self) -> None:
        no_fmt_prep = preprocess_prompt("Explain gravity.")
        no_fmt_nlp = analyze_prompt("Explain gravity.")
        no_fmt_dims = evaluate_dimensions(no_fmt_prep, no_fmt_nlp)

        fmt_prep = preprocess_prompt("Explain gravity. Return the output as JSON.")
        fmt_nlp = analyze_prompt("Explain gravity. Return the output as JSON.")
        fmt_dims = evaluate_dimensions(fmt_prep, fmt_nlp)

        self.assertGreater(fmt_dims["output_format"].score, no_fmt_dims["output_format"].score)

    def test_role_persona_detection_boosts_dimension(self) -> None:
        no_role_prep = preprocess_prompt("Write a test case.")
        no_role_nlp = analyze_prompt("Write a test case.")
        no_role_dims = evaluate_dimensions(no_role_prep, no_role_nlp)

        role_prep = preprocess_prompt("You are a QA automation architect. Write a test case.")
        role_nlp = analyze_prompt("You are a QA automation architect. Write a test case.")
        role_dims = evaluate_dimensions(role_prep, role_nlp)

        self.assertGreater(role_dims["role_persona"].score, no_role_dims["role_persona"].score)


if __name__ == "__main__":
    unittest.main()
