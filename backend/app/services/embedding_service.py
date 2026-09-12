from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from app.engines.embeddings.chunking import chunk_text
from app.engines.embeddings.config import SETTINGS
from app.engines.embeddings.model import load_embedding_model, normalize_embedding
from app.engines.embeddings.similarity import cosine_similarity
from app.schemas.embeddings import EmbeddingModelMetadata


class EmbeddingService:
    def __init__(self) -> None:
        self._model = load_embedding_model()
        self._dimension = int(self._model.get_embedding_dimension())
        self._max_tokens = min(
            SETTINGS.max_chunk_tokens,
            max(1, int(self._model.max_seq_length) - 2),
        )

    @property
    def metadata(self) -> EmbeddingModelMetadata:
        return EmbeddingModelMetadata(
            name=SETTINGS.model_name,
            version=SETTINGS.model_version,
            dimension=self._dimension,
            normalized=SETTINGS.normalized,
        )

    def encode(self, prompt: str) -> np.ndarray:
        return self.encode_batch([prompt])[0]

    def encode_batch(self, prompts: Sequence[str]) -> list[np.ndarray]:
        if not prompts:
            return []
        if any(not prompt.strip() for prompt in prompts):
            raise ValueError("prompts must not be empty or whitespace-only")
        tokenizer = self._model.tokenizer.encode
        decoder = self._model.tokenizer.decode
        prompt_chunks = [
            chunk_text(prompt, tokenizer, decoder, self._max_tokens)
            for prompt in prompts
        ]
        all_chunks = [chunk for chunks in prompt_chunks for chunk in chunks]
        raw_chunks = self._model.encode(
            all_chunks,
            batch_size=min(32, len(all_chunks)),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        embeddings: list[np.ndarray] = []
        offset = 0
        for chunks in prompt_chunks:
            count = len(chunks)
            vectors = np.asarray(raw_chunks[offset:offset + count], dtype=np.float32)
            weights = np.asarray(
                [max(1, len(tokenizer(chunk, add_special_tokens=False))) for chunk in chunks],
                dtype=np.float32,
            )
            aggregate = np.average(vectors, axis=0, weights=weights)
            embeddings.append(normalize_embedding(aggregate) if SETTINGS.normalized else aggregate)
            offset += count
        return embeddings

    def similarity(self, text_a: str, text_b: str) -> float:
        return cosine_similarity(self.encode(text_a), self.encode(text_b))


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()