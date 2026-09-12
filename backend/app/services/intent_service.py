from app.ml.intent.inference import IntentClassifier, load_classifier
from app.schemas.intent import IntentClassificationResponse


_CLASSIFIER: IntentClassifier | None = None


def _get_classifier() -> IntentClassifier:
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _CLASSIFIER = load_classifier()
    return _CLASSIFIER


def classify_prompt(prompt: str) -> IntentClassificationResponse:
    if not prompt.strip():
        raise ValueError("prompt must not be empty or whitespace-only")
    return _get_classifier().predict(prompt)