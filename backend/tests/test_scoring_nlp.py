import unittest

from app.engines.nlp import analyze_prompt
from app.scoring.signals import compute_nlp_score


class ScoringNlpTests(unittest.TestCase):
    def test_empty_prompt_nlp_score(self) -> None:
        nlp = analyze_prompt("")
        score = compute_nlp_score(nlp)
        self.assertEqual(score, 0.0)

    def test_minimal_prompt_nlp_score(self) -> None:
        nlp = analyze_prompt("Hi")
        score = compute_nlp_score(nlp)
        self.assertLessEqual(score, 25.0)

    def test_well_formed_sentences_score_well(self) -> None:
        nlp = analyze_prompt(
            "Analyze the following performance metrics and identify potential bottlenecks in the data pipeline."
        )
        score = compute_nlp_score(nlp)
        self.assertGreaterEqual(score, 65.0)
        self.assertLessEqual(score, 100.0)

    def test_nlp_score_is_deterministic(self) -> None:
        prompt = "Explain quantum computing algorithms using clear analogies for undergraduate students."
        nlp1 = analyze_prompt(prompt)
        nlp2 = analyze_prompt(prompt)

        self.assertEqual(compute_nlp_score(nlp1), compute_nlp_score(nlp2))

    def test_nlp_score_bounds(self) -> None:
        prompts = [
            "x",
            "Explain machine learning.",
            "Write a function. Include unit tests. Return JSON.",
            "word " * 100,
        ]
        for p in prompts:
            score = compute_nlp_score(analyze_prompt(p))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)


if __name__ == "__main__":
    unittest.main()
