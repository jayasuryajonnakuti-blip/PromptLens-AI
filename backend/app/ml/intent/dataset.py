import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.schemas.intent import IntentLabel


SUPPORTED_INTENTS = tuple(intent.value for intent in IntentLabel)
MIN_EXAMPLES_PER_CLASS = 5


class DatasetValidationError(ValueError):
    """Raised when the intent dataset cannot be used for training."""


@dataclass(frozen=True)
class IntentDataset:
    texts: tuple[str, ...]
    labels: tuple[str, ...]
    version: str
    distribution: dict[str, int]


def load_and_validate_dataset(path: Path, version: str = "0.1.0") -> IntentDataset:
    if not path.is_file():
        raise DatasetValidationError(f"Intent dataset does not exist: {path}")

    texts: list[str] = []
    labels: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["text", "intent"]:
            raise DatasetValidationError("Dataset must contain exactly: text,intent")
        for row_number, row in enumerate(reader, start=2):
            text = (row.get("text") or "").strip()
            intent = (row.get("intent") or "").strip()
            if not text:
                raise DatasetValidationError(f"Empty prompt text at row {row_number}")
            if intent not in SUPPORTED_INTENTS:
                raise DatasetValidationError(
                    f"Unsupported intent '{intent}' at row {row_number}"
                )
            normalized_text = text.casefold()
            if normalized_text in seen:
                raise DatasetValidationError(
                    f"Duplicate prompt text at row {row_number}: {text}"
                )
            seen.add(normalized_text)
            texts.append(text)
            labels.append(intent)

    distribution = dict(Counter(labels))
    missing = set(SUPPORTED_INTENTS) - distribution.keys()
    if missing:
        raise DatasetValidationError(f"Missing intent classes: {sorted(missing)}")
    undersized = {
        intent: count
        for intent, count in distribution.items()
        if count < MIN_EXAMPLES_PER_CLASS
    }
    if undersized:
        raise DatasetValidationError(
            f"Each intent needs at least {MIN_EXAMPLES_PER_CLASS} examples: {undersized}"
        )
    return IntentDataset(tuple(texts), tuple(labels), version, distribution)