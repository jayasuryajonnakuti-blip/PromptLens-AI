import unittest

from app.engines.nlp import analyze_prompt
from app.engines.preprocessing import preprocess_prompt
from app.ml.quality.categories import score_to_label
from app.scoring.config import DEFAULT_SCORING_CONFIG
from app.scoring.fusion import fuse_signals
from app.services.quality_service import predict_quality


class ScoringFusionTests(unittest.TestCase):
    def test_fusion_weight_renormalization_excludes_missing_llm(self) -> None:
        prompt = "Explain quantum computing algorithms in 3 paragraphs for students."
        prep = preprocess_prompt(prompt)
        nlp = analyze_prompt(prompt)
        qual = predict_quality(prompt)

        result = fuse_signals(prompt, prep, nlp, qual)

        # Verify LLM is marked unavailable and weight is 0.0
        self.assertFalse(result.signals["llm"].available)
        self.assertIsNone(result.signals["llm"].raw_score)
        self.assertEqual(result.signals["llm"].effective_weight, 0.0)

        # Verify available signals have positive effective weights summing to 1.0
        available_weights = [
            result.signals["rules"].effective_weight,
            result.signals["nlp"].effective_weight,
            result.signals["quality_ml"].effective_weight,
        ]
        self.assertAlmostEqual(sum(available_weights), 1.0, places=5)

    def test_fusion_score_calculation(self) -> None:
        prompt = "Write a Python script to sort an array."
        prep = preprocess_prompt(prompt)
        nlp = analyze_prompt(prompt)
        qual = predict_quality(prompt)

        result = fuse_signals(prompt, prep, nlp, qual)

        r_score = result.signals["rules"].raw_score
        r_w = result.signals["rules"].effective_weight
        n_score = result.signals["nlp"].raw_score
        n_w = result.signals["nlp"].effective_weight
        q_score = result.signals["quality_ml"].raw_score
        q_w = result.signals["quality_ml"].effective_weight

        expected = round(r_score * r_w + n_score * n_w + q_score * q_w, 2)
        self.assertAlmostEqual(result.overall_score.score, expected, places=2)
        self.assertEqual(result.overall_score.category, score_to_label(result.overall_score.score))
        self.assertEqual(result.overall_score.status, result.overall_score.category.value)

    def test_fusion_is_deterministic(self) -> None:
        prompt = "Design a REST API for a bookstore with endpoints to list, create, and search books."
        prep = preprocess_prompt(prompt)
        nlp = analyze_prompt(prompt)
        qual = predict_quality(prompt)

        res1 = fuse_signals(prompt, prep, nlp, qual)
        res2 = fuse_signals(prompt, prep, nlp, qual)

        self.assertEqual(res1.overall_score.score, res2.overall_score.score)
        self.assertEqual(res1.overall_score.category, res2.overall_score.category)

    def test_fusion_score_bounds(self) -> None:
        test_prompts = [
            "x",
            "Explain machine learning.",
            "You are a cloud architect. Write a Terraform script to deploy an EKS cluster. Return JSON.",
            "Write a script. " * 80,
        ]
        for p in test_prompts:
            prep = preprocess_prompt(p)
            nlp = analyze_prompt(p)
            qual = predict_quality(p)
            res = fuse_signals(p, prep, nlp, qual)
            self.assertGreaterEqual(res.overall_score.score, 0.0)
            self.assertLessEqual(res.overall_score.score, 100.0)


if __name__ == "__main__":
    unittest.main()
