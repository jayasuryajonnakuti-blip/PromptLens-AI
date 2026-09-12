import unittest

import numpy as np

from app.engines.embeddings.chunking import chunk_text
from app.engines.embeddings.similarity import (
    EmbeddingDimensionError,
    cosine_similarity,
    is_near_duplicate,
    pairwise_similarity,
)
from app.services.embedding_service import get_embedding_service


class EmbeddingUtilityTests(unittest.TestCase):
    def test_cosine_similarity_and_zero_vectors(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0)

    def test_incompatible_dimensions_are_rejected(self) -> None:
        with self.assertRaises(EmbeddingDimensionError):
            cosine_similarity([1.0, 0.0], [1.0])

    def test_pairwise_similarity(self) -> None:
        matrix = pairwise_similarity([[1.0, 0.0], [0.0, 1.0]])

        self.assertEqual(matrix.shape, (2, 2))
        self.assertAlmostEqual(float(matrix[0, 0]), 1.0)
        self.assertAlmostEqual(float(matrix[0, 1]), 0.0)

    def test_near_duplicate_threshold(self) -> None:
        self.assertTrue(is_near_duplicate(0.95, 0.9))
        self.assertFalse(is_near_duplicate(0.85, 0.9))

    def test_chunking_preserves_nonempty_content(self) -> None:
        tokenizer = lambda value, add_special_tokens=False: value.split()
        decoder = lambda tokens: " ".join(tokens)
        text = "First paragraph has words.\n\nSecond paragraph has more words."

        chunks = chunk_text(text, tokenizer, decoder, max_tokens=4)

        self.assertGreater(len(chunks), 1)
        self.assertIn("First", " ".join(chunks))
        self.assertIn("Second", " ".join(chunks))


class EmbeddingModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = get_embedding_service()

    def test_embedding_shape_and_normalization(self) -> None:
        embedding = self.service.encode("Explain machine learning in simple terms.")

        self.assertEqual(embedding.shape, (384,))
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=3)
        self.assertTrue(self.service.metadata.normalized)

    def test_repeated_embedding_is_deterministic(self) -> None:
        first = self.service.encode("Explain machine learning.")
        second = self.service.encode("Explain machine learning.")

        self.assertTrue(np.allclose(first, second, atol=1e-6))

    def test_related_similarity_is_higher_than_unrelated(self) -> None:
        related = self.service.similarity(
            "Explain machine learning.",
            "Describe the basics of machine learning.",
        )
        unrelated = self.service.similarity(
            "Explain machine learning.",
            "What is the weather forecast today?",
        )

        self.assertGreaterEqual(related, unrelated)
        self.assertGreater(self.service.similarity("same prompt", "same prompt"), 0.99)

    def test_batch_encoding(self) -> None:
        embeddings = self.service.encode_batch(["First prompt.", "Second prompt."])

        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0].shape, embeddings[1].shape)

    def test_long_prompt_is_chunked_and_embedded(self) -> None:
        embedding = self.service.encode(
            "This is a long sentence about semantic prompt embeddings. " * 150
        )

        self.assertEqual(embedding.shape, (384,))
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()