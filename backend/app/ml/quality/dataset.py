"""
Dataset loader and validator for the PromptLens quality ML baseline (Step 12).

The quality dataset is a development baseline. It is heuristically annotated
and does not represent production-quality ground truth. A larger, manually
validated dataset is required for production use.

Dataset format: prompt_quality_dataset_v1.csv
Required columns:
    prompt, quality_score, quality_label,
    clarity, specificity, context, goal_definition, constraints,
    output_format, role_persona, audience, ambiguity, completeness,
    actionability, consistency

Dimension scores (clarity..consistency) are stored for analytical reference
and are NOT used as model input features to prevent data leakage.
"""
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.ml.quality.categories import SUPPORTED_LABELS, label_score_is_consistent

REQUIRED_COLUMNS: tuple[str, ...] = (
    "prompt",
    "quality_score",
    "quality_label",
    "clarity",
    "specificity",
    "context",
    "goal_definition",
    "constraints",
    "output_format",
    "role_persona",
    "audience",
    "ambiguity",
    "completeness",
    "actionability",
    "consistency",
)

DIMENSION_COLUMNS: tuple[str, ...] = (
    "clarity",
    "specificity",
    "context",
    "goal_definition",
    "constraints",
    "output_format",
    "role_persona",
    "audience",
    "ambiguity",
    "completeness",
    "actionability",
    "consistency",
)

MIN_EXAMPLES_PER_CLASS = 5


class DatasetValidationError(ValueError):
    """Raised when the quality dataset cannot be used for training."""


@dataclass(frozen=True)
class QualityDataset:
    prompts: tuple[str, ...]
    scores: tuple[float, ...]
    labels: tuple[str, ...]
    version: str
    distribution: dict[str, int]


def load_and_validate_dataset(path: Path, version: str = "0.1.0") -> QualityDataset:
    """Load and strictly validate the quality dataset CSV.

    Raises:
        DatasetValidationError: On any schema, value, consistency, or
            duplicate violation. Does not silently repair invalid data.
    """
    if not path.is_file():
        raise DatasetValidationError(f"Quality dataset does not exist: {path}")

    prompts: list[str] = []
    scores: list[float] = []
    labels: list[str] = []
    seen: set[str] = set()

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        # Column check
        if reader.fieldnames is None:
            raise DatasetValidationError("Dataset file appears to be empty or missing a header.")
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing_cols:
            raise DatasetValidationError(
                f"Dataset is missing required columns: {missing_cols}"
            )

        for row_number, row in enumerate(reader, start=2):
            # --- Prompt ---
            prompt = (row.get("prompt") or "").strip()
            if not prompt:
                raise DatasetValidationError(f"Empty or missing prompt at row {row_number}")

            normalized = prompt.casefold()
            if normalized in seen:
                raise DatasetValidationError(
                    f"Duplicate prompt at row {row_number}: '{prompt[:60]}...'"
                )
            seen.add(normalized)

            # --- Quality score ---
            raw_score = (row.get("quality_score") or "").strip()
            if not raw_score:
                raise DatasetValidationError(
                    f"Missing quality_score at row {row_number}"
                )
            try:
                score = float(raw_score)
            except ValueError:
                raise DatasetValidationError(
                    f"quality_score is not numeric at row {row_number}: '{raw_score}'"
                )
            if not (0.0 <= score <= 100.0):
                raise DatasetValidationError(
                    f"quality_score {score} out of [0, 100] at row {row_number}"
                )

            # --- Quality label ---
            label = (row.get("quality_label") or "").strip()
            if not label:
                raise DatasetValidationError(
                    f"Missing quality_label at row {row_number}"
                )
            if label not in SUPPORTED_LABELS:
                raise DatasetValidationError(
                    f"Unsupported quality_label '{label}' at row {row_number}. "
                    f"Must be one of: {sorted(SUPPORTED_LABELS)}"
                )

            # --- Label/score consistency ---
            if not label_score_is_consistent(score, label):
                raise DatasetValidationError(
                    f"quality_label '{label}' is inconsistent with quality_score {score} "
                    f"at row {row_number}"
                )

            # --- Dimension scores ---
            for col in DIMENSION_COLUMNS:
                raw_dim = (row.get(col) or "").strip()
                if not raw_dim:
                    raise DatasetValidationError(
                        f"Missing value for dimension '{col}' at row {row_number}"
                    )
                try:
                    dim_score = float(raw_dim)
                except ValueError:
                    raise DatasetValidationError(
                        f"Dimension '{col}' is not numeric at row {row_number}: '{raw_dim}'"
                    )
                if not (0.0 <= dim_score <= 100.0):
                    raise DatasetValidationError(
                        f"Dimension '{col}' value {dim_score} out of [0, 100] at row {row_number}"
                    )

            prompts.append(prompt)
            scores.append(score)
            labels.append(label)

    if not prompts:
        raise DatasetValidationError("Dataset contains no valid rows.")

    distribution = dict(Counter(labels))
    missing_classes = SUPPORTED_LABELS - distribution.keys()
    if missing_classes:
        raise DatasetValidationError(
            f"Dataset is missing quality label classes: {sorted(missing_classes)}"
        )
    undersized = {
        label: count
        for label, count in distribution.items()
        if count < MIN_EXAMPLES_PER_CLASS
    }
    if undersized:
        raise DatasetValidationError(
            f"Each quality label needs at least {MIN_EXAMPLES_PER_CLASS} examples. "
            f"Undersized classes: {undersized}"
        )

    return QualityDataset(
        prompts=tuple(prompts),
        scores=tuple(scores),
        labels=tuple(labels),
        version=version,
        distribution=distribution,
    )
