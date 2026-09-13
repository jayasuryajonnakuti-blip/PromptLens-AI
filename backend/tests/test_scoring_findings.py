import unittest

from app.engines.nlp import analyze_prompt
from app.engines.preprocessing import preprocess_prompt
from app.scoring.dimensions import evaluate_dimensions
from app.scoring.findings import (
    FindingSeverity,
    generate_findings,
    generate_recommendations,
)


class ScoringFindingsTests(unittest.TestCase):
    def test_empty_prompt_produces_error_finding(self) -> None:
        prep = preprocess_prompt("")
        nlp = analyze_prompt("")
        dims = evaluate_dimensions(prep, nlp)
        findings = generate_findings(prep, nlp, dims)

        self.assertTrue(any(f.severity == FindingSeverity.ERROR for f in findings))
        self.assertEqual(findings[0].type, "empty_prompt")

    def test_near_empty_prompt_produces_warning(self) -> None:
        prep = preprocess_prompt("Help me")
        nlp = analyze_prompt("Help me")
        dims = evaluate_dimensions(prep, nlp)
        findings = generate_findings(prep, nlp, dims)

        self.assertTrue(any(f.type == "near_empty_prompt" for f in findings))

    def test_missing_format_produces_info_finding(self) -> None:
        prep = preprocess_prompt("Explain the theory of relativity.")
        nlp = analyze_prompt("Explain the theory of relativity.")
        dims = evaluate_dimensions(prep, nlp)
        findings = generate_findings(prep, nlp, dims)

        self.assertTrue(any(f.type == "missing_output_format" for f in findings))

    def test_repetition_produces_warning(self) -> None:
        prompt = "make it clear and make it clear and make it clear and make it clear"
        prep = preprocess_prompt(prompt)
        nlp = analyze_prompt(prompt)
        dims = evaluate_dimensions(prep, nlp)
        findings = generate_findings(prep, nlp, dims)

        self.assertTrue(any(f.type == "excessive_repetition" for f in findings))

    def test_recommendations_generation_is_deterministic(self) -> None:
        prompt = "Write code."
        prep = preprocess_prompt(prompt)
        nlp = analyze_prompt(prompt)
        dims = evaluate_dimensions(prep, nlp)
        findings = generate_findings(prep, nlp, dims)

        rec1 = generate_recommendations(dims, findings)
        rec2 = generate_recommendations(dims, findings)

        self.assertEqual(rec1, rec2)
        self.assertGreaterEqual(len(rec1), 1)
        self.assertLessEqual(len(rec1), 4)


if __name__ == "__main__":
    unittest.main()
