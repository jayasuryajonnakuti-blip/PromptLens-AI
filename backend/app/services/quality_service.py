"""Quality inference service for the PromptLens quality ML model (Step 12).

Loads the persisted model artifact once at startup and reuses it for all
inference requests. Training is never performed during an API request.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.ml.quality.categories import score_to_label
from app.ml.quality.features import extract_features
from app.schemas.quality import QualityModelMetadata, QualityPredictionResponse

MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "quality" / "quality_model.joblib"
)


class QualityModelUnavailableError(RuntimeError):
    """Raised when the trained quality model artifact is not available."""


class QualityModel:
    """Wrapper around the persisted quality regression pipeline."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        self._pipeline = artifact["pipeline"]
        meta = artifact["metadata"]
        self._metadata = QualityModelMetadata(
            name=meta["model_name"],
            version=meta["model_version"],
        )

    @property
    def metadata(self) -> QualityModelMetadata:
        return self._metadata

    def predict(self, prompt: str) -> QualityPredictionResponse:
        """Predict the quality score and label for a prompt.

        Args:
            prompt: Raw, non-empty prompt text.

        Returns:
            QualityPredictionResponse with score, label, and model metadata.
        """
        feature_vector = extract_features(prompt)
        X = np.asarray([feature_vector], dtype=np.float64)
        raw_score = float(self._pipeline.predict(X)[0])
        # Clamp only if the regressor produces out-of-range values
        clamped_score = max(0.0, min(100.0, raw_score))
        label = score_to_label(clamped_score)
        return QualityPredictionResponse(
            score=round(clamped_score, 2),
            label=label,
            model=self._metadata,
        )


@lru_cache(maxsize=1)
def load_quality_model() -> QualityModel:
    """Load and cache the quality model artifact.

    Raises:
        QualityModelUnavailableError: When the artifact file does not exist.
    """
    if not MODEL_PATH.is_file():
        raise QualityModelUnavailableError(
            f"Quality model artifact not found at {MODEL_PATH}. "
            "Train it with: python -m app.ml.quality.train"
        )
    return QualityModel(joblib.load(MODEL_PATH))


_MODEL: QualityModel | None = None


def predict_quality(prompt: str) -> QualityPredictionResponse:
    """Public interface for quality prediction used by the API layer.

    Args:
        prompt: Non-empty prompt string.

    Returns:
        QualityPredictionResponse.

    Raises:
        ValueError: When prompt is empty or whitespace-only.
        QualityModelUnavailableError: When the model artifact is unavailable.
    """
    global _MODEL
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty or whitespace-only")
    if _MODEL is None:
        _MODEL = load_quality_model()
    return _MODEL.predict(prompt)
