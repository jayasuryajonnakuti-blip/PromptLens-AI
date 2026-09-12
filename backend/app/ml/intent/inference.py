from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from app.schemas.intent import IntentClassificationResponse, IntentLabel, IntentModelMetadata

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "intent" / "intent_classifier.joblib"


class IntentModelUnavailableError(RuntimeError):
    """Raised when the trained intent artifact is not available."""


class IntentClassifier:
    def __init__(self, artifact: dict[str, Any]) -> None:
        self.pipeline = artifact["pipeline"]
        self.metadata = artifact["metadata"]

    def predict(self, prompt: str) -> IntentClassificationResponse:
        probabilities = self.pipeline.predict_proba([prompt])[0]
        classes = list(self.pipeline.classes_)
        best_index = int(probabilities.argmax())
        intent = IntentLabel(classes[best_index])
        return IntentClassificationResponse(
            intent=intent,
            confidence=round(float(probabilities[best_index]), 4),
            model=IntentModelMetadata(
                name=self.metadata["model_name"], version=self.metadata["model_version"]
            ),
            confidence_methodology=self.metadata["confidence_methodology"],
        )


@lru_cache(maxsize=1)
def load_classifier() -> IntentClassifier:
    if not MODEL_PATH.is_file():
        raise IntentModelUnavailableError(
            f"Intent model artifact is unavailable at {MODEL_PATH}. "
            "Train it with: python -m app.ml.intent.train"
        )
    return IntentClassifier(joblib.load(MODEL_PATH))