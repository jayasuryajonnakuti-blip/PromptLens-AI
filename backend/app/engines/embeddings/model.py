from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

# Keep the default CPU runtime stable on Windows environments with OpenMP DLLs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from app.engines.embeddings.config import SETTINGS


class EmbeddingModelUnavailableError(RuntimeError):
    """Raised when the local sentence-transformer model cannot be loaded."""


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise EmbeddingModelUnavailableError(
            "sentence-transformers is unavailable. Install backend requirements first."
        ) from error
    try:
        return SentenceTransformer(SETTINGS.model_name, device="cpu")
    except Exception as error:
        raise EmbeddingModelUnavailableError(
            f"Unable to load local embedding model '{SETTINGS.model_name}'. "
            "The first run downloads it to the local model cache."
        ) from error


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).copy()
    norm = np.linalg.norm(vector)
    return vector if norm == 0.0 else vector / norm