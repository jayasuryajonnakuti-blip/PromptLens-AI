from collections.abc import Sequence

import numpy as np


class EmbeddingDimensionError(ValueError):
    """Raised when vectors cannot be compared because their dimensions differ."""


def _as_vector(values: Sequence[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError("embedding must be a one-dimensional vector")
    return vector


def cosine_similarity(
    embedding_a: Sequence[float] | np.ndarray,
    embedding_b: Sequence[float] | np.ndarray,
) -> float:
    vector_a = _as_vector(embedding_a)
    vector_b = _as_vector(embedding_b)
    if vector_a.shape != vector_b.shape:
        raise EmbeddingDimensionError(
            f"embedding dimensions differ: {vector_a.size} != {vector_b.size}"
        )
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    value = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    return max(-1.0, min(1.0, value))


def pairwise_similarity(embeddings: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    result = (matrix / safe_norms) @ (matrix / safe_norms).T
    result[(norms == 0.0).ravel(), :] = 0.0
    result[:, (norms == 0.0).ravel()] = 0.0
    return np.clip(result, -1.0, 1.0)


def is_near_duplicate(similarity: float, threshold: float = 0.90) -> bool:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("near-duplicate threshold must be between 0 and 1")
    return similarity >= threshold