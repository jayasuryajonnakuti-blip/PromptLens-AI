import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib

from app.ml.intent.dataset import IntentDataset, load_and_validate_dataset
from app.ml.intent.evaluate import classification_metrics
from app.ml.intent.features import build_vectorizer

MODEL_NAME = "promptlens-intent-classifier"
MODEL_VERSION = "0.1.0"
RANDOM_STATE = 42
LOGGER = logging.getLogger(__name__)


def _split_dataset(dataset: IntentDataset) -> tuple[list[str], ...]:
    train_texts, holdout_texts, train_labels, holdout_labels = train_test_split(
        list(dataset.texts),
        list(dataset.labels),
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=list(dataset.labels),
    )
    validation_texts, test_texts, validation_labels, test_labels = train_test_split(
        holdout_texts,
        holdout_labels,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=holdout_labels,
    )
    return train_texts, train_labels, validation_texts, validation_labels, test_texts, test_labels


def _candidates() -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2_000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "linear_svm": CalibratedClassifierCV(
            estimator=LinearSVC(random_state=RANDOM_STATE, class_weight="balanced"),
            method="sigmoid",
            cv=3,
        ),
    }


def train_intent_models(
    dataset_path: Path,
    model_dir: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    dataset = load_and_validate_dataset(dataset_path)
    train_texts, train_labels, validation_texts, validation_labels, test_texts, test_labels = _split_dataset(dataset)
    labels = tuple(sorted(dataset.distribution))
    validation_results: dict[str, Any] = {}
    candidates: dict[str, Pipeline] = {}

    for name, classifier in _candidates().items():
        pipeline = Pipeline([("tfidf", build_vectorizer()), ("classifier", classifier)])
        pipeline.fit(train_texts, train_labels)
        predictions = pipeline.predict(validation_texts).tolist()
        validation_results[name] = classification_metrics(validation_labels, predictions, labels)
        candidates[name] = pipeline

    selected_name = max(
        validation_results,
        key=lambda name: (validation_results[name]["macro_f1"], validation_results[name]["accuracy"]),
    )
    selected_pipeline = candidates[selected_name]
    final_train_texts = train_texts + validation_texts
    final_train_labels = train_labels + validation_labels
    selected_pipeline.fit(final_train_texts, final_train_labels)
    test_predictions = selected_pipeline.predict(test_texts).tolist()
    test_metrics = classification_metrics(test_labels, test_predictions, labels)

    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "intent_classifier.joblib"
    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "selected_model": selected_name,
        "dataset_version": dataset.version,
        "training_date_utc": datetime.now(UTC).isoformat(),
        "supported_intents": list(labels),
        "class_distribution": dataset.distribution,
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "random_state": RANDOM_STATE,
        "selection_metric": "validation_macro_f1",
        "validation_metrics": validation_results,
        "test_metrics": test_metrics,
        "confidence_methodology": (
            "Logistic regression uses predict_proba; calibrated LinearSVC uses sigmoid calibration. "
            "Confidence is model confidence, not certainty."
        ),
        "model_path": str(model_path),
    }
    joblib.dump({"pipeline": selected_pipeline, "metadata": metadata}, model_path)
    (artifact_dir / "evaluation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Selected %s with validation macro F1 %.4f", selected_name, validation_results[selected_name]["macro_f1"])
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PromptLens intent baseline.")
    parser.add_argument("--dataset", type=Path, default=Path("data/intent_dataset_v1.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/intent"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/intent"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    train_intent_models(args.dataset, args.model_dir, args.artifact_dir)


if __name__ == "__main__":
    main()