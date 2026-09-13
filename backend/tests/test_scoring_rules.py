import unittest

from app.engines.preprocessing import preprocess_prompt
from app.scoring.signals import compute_rule_score


class ScoringRulesTests(unittest.TestCase):
    def test_empty_prompt_score_is_zero(self) -> None:
        prep = preprocess_prompt("")
        score = compute_rule_score(prep)
        self.assertEqual(score, 0.0)

    def test_near_empty_prompt_score(self) -> None:
        prep = preprocess_prompt("Hi there")
        score = compute_rule_score(prep)
        self.assertLess(score, 30.0)

    def test_structured_prompt_scores_higher_than_simple(self) -> None:
        simple_prep = preprocess_prompt("Write code.")
        structured_prep = preprocess_prompt(
            "You are a backend architect. Write a Python FastAPI endpoint for user registration. "
            "Include email validation, password hashing, and return response as JSON with status 201."
        )
        simple_score = compute_rule_score(simple_prep)
        structured_score = compute_rule_score(structured_prep)

        self.assertGreater(structured_score, simple_score)
        self.assertGreaterEqual(structured_score, 60.0)

    def test_repetition_penalizes_score(self) -> None:
        normal_prep = preprocess_prompt("Create a clean data pipeline for telemetry events.")
        repetitive_prep = preprocess_prompt(
            "make it fast and make it fast and make it fast and make it fast and make it fast"
        )
        normal_score = compute_rule_score(normal_prep)
        rep_score = compute_rule_score(repetitive_prep)

        self.assertGreater(normal_score, rep_score)

    def test_rule_score_is_deterministic(self) -> None:
        prompt = "Explain photosynthesis to high school students in 3 numbered points."
        prep1 = preprocess_prompt(prompt)
        prep2 = preprocess_prompt(prompt)

        self.assertEqual(compute_rule_score(prep1), compute_rule_score(prep2))

    def test_rule_score_bounds(self) -> None:
        prompts = [
            "Hi",
            "Write an essay.",
            "Write a Python script. " * 50,
            "You are a developer. Create a function. Return as JSON.",
        ]
        for p in prompts:
            score = compute_rule_score(preprocess_prompt(p))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)


if __name__ == "__main__":
    unittest.main()
