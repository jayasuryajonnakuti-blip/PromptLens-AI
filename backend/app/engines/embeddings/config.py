from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingSettings:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_version: str = "all-MiniLM-L6-v2"
    application_version: str = "0.1.0"
    normalized: bool = True
    similarity_threshold: float = 0.90
    max_chunk_tokens: int = 254


SETTINGS = EmbeddingSettings()