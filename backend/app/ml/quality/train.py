"""
Training pipeline for the PromptLens quality ML model (Step 12).

Benchmarks four regression algorithms and selects the one with the lowest
validation MAE.  The held-out test set is evaluated only once, after model
selection, to prevent test-set leakage.

Leakage prevention:
- StandardScaler is fitted on the training split only.
- Test data is never seen during feature fitting or model selection.
- Dimension scores from the dataset CSV are not used as features.
- Model selection is based on validation performance, not test performance.

Usage:
    python -m app.ml.quality.train
    python -m app.ml.quality.train --dataset data/prompt_quality_dataset_v1.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.quality.dataset import load_and_validate_dataset
from app.ml.quality.evaluate import (
    classification_metrics_from_scores,
    regression_metrics,
)
from app.ml.quality.features import FEATURE_NAMES, extract_features_batch

MODEL_NAME = "promptlens-quality-model"
MODEL_VERSION = "0.1.0"
RANDOM_STATE = 42
LOGGER = logging.getLogger(__name__)


def _candidates() -> dict[str, Any]:
    """Return the benchmark regressor candidates (without the scaler)."""
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
        ),
        "ridge": Ridge(alpha=1.0),
    }


def _build_pipeline(regressor: Any) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", regressor),
    ])


def train_quality_model(
    dataset_path: Path,
    model_dir: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Train, benchmark, and persist the quality regression model.

    Returns:
        Metadata dictionary saved to the evaluation JSON artifact.
    """
    dataset = load_and_validate_dataset(dataset_path)
    LOGGER.info(
        "Loaded dataset: %d prompts, distribution: %s",
        len(dataset.prompts),
        dataset.distribution,
    )

    # Build feature matrix
    LOGGER.info("Extracting features for %d prompts...", len(dataset.prompts))
    t0_feat = time.perf_counter()
    X = np.asarray(extract_features_batch(list(dataset.prompts)), dtype=np.float64)
    y = np.asarray(dataset.scores, dtype=np.float64)
    feat_time = time.perf_counter() - t0_feat
    LOGGER.info("Feature extraction took %.1fs", feat_time)

    # Stratified split: 70% train / 15% validation / 15% test
    X_train_val, X_test, y_train_val, y_test, labels_train_val, labels_test = train_test_split(
        X, y, list(dataset.labels),
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=list(dataset.labels),
    )
    X_train, X_val, y_train, y_val, labels_train, labels_val = train_test_split(
        X_train_val, y_train_val, labels_train_val,
        test_size=0.15 / 0.85,  # ~15% of total
        random_state=RANDOM_STATE,
        stratify=labels_train_val,
    )
    LOGGER.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    # Benchmark candidates
    validation_results: dict[str, dict[str, Any]] = {}
    trained_pipelines: dict[str, Pipeline] = {}

    for name, regressor in _candidates().items():
        pipeline = _build_pipeline(regressor)
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        val_predicted = pipeline.predict(X_val).tolist()
        val_metrics = regression_metrics(y_val.tolist(), val_predicted)
        val_metrics["training_time_s"] = round(train_time, 3)

        LOGGER.info(
            "%-30s | val_mae=%.3f | val_rmse=%.3f | val_r2=%.3f | time=%.1fs",
            name, val_metrics["mae"], val_metrics["rmse"], val_metrics["r2"], train_time,
        )
        validation_results[name] = val_metrics
        trained_pipelines[name] = pipeline

    # Select best model by validation MAE (lower is better)
    selected_name = min(validation_results, key=lambda n: validation_results[n]["mae"])
    LOGGER.info("Selected model: %s (val_mae=%.4f)", selected_name, validation_results[selected_name]["mae"])

    # Refit selected model on train + validation combined
    selected_pipeline = _build_pipeline(_candidates()[selected_name])
    t0_final = time.perf_counter()
    selected_pipeline.fit(X_train_val, y_train_val)
    final_train_time = time.perf_counter() - t0_final

    # Evaluate on held-out test set (only once, after selection)
    t0_inf = time.perf_counter()
    test_predicted = selected_pipeline.predict(X_test).tolist()
    inference_time_ms = (time.perf_counter() - t0_inf) * 1000 / max(1, len(X_test))

    test_regression = regression_metrics(y_test.tolist(), test_predicted)
    test_classification = classification_metrics_from_scores(y_test.tolist(), test_predicted)

    # Persist model
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "quality_model.joblib"
    joblib.dump(
        {
            "pipeline": selected_pipeline,
            "metadata": {
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "selected_model": selected_name,
                "feature_names": list(FEATURE_NAMES),
            },
        },
        model_path,
    )
    model_size_kb = round(model_path.stat().st_size / 1024, 1)

    metadata: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "dataset_version": dataset.version,
        "dataset_size": len(dataset.prompts),
        "dataset_distribution": dataset.distribution,
        "training_date_utc": datetime.now(UTC).isoformat(),
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "random_state": RANDOM_STATE,
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "selected_model": selected_name,
        "selection_metric": "validation_mae",
        "validation_metrics": {
            name: {k: v for k, v in metrics.items()}
            for name, metrics in validation_results.items()
        },
        "test_regression_metrics": test_regression,
        "test_classification_metrics": test_classification,
        "training_time_s": round(final_train_time, 3),
        "feature_extraction_time_s": round(feat_time, 3),
        "inference_time_per_sample_ms": round(inference_time_ms, 3),
        "model_path": str(model_path),
        "model_size_kb": model_size_kb,
        "disclaimer": (
            "The quality model is a learned baseline estimate and does not represent "
            "ground-truth prompt quality. Production performance requires a substantially "
            "larger, manually validated dataset and ongoing evaluation."
        ),
    }

    (artifact_dir / "evaluation.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    LOGGER.info(
        "Model saved to %s (%.1f KB). Test MAE=%.4f, R2=%.4f",
        model_path, model_size_kb,
        test_regression["mae"], test_regression["r2"],
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PromptLens quality baseline model.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/prompt_quality_dataset_v1.csv"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/quality"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/quality"),
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    result = train_quality_model(args.dataset, args.model_dir, args.artifact_dir)
    print(json.dumps({
        "selected_model": result["selected_model"],
        "test_mae": result["test_regression_metrics"]["mae"],
        "test_rmse": result["test_regression_metrics"]["rmse"],
        "test_r2": result["test_regression_metrics"]["r2"],
        "test_accuracy": result["test_classification_metrics"]["accuracy"],
        "test_macro_f1": result["test_classification_metrics"]["macro_f1"],
    }, indent=2))


if __name__ == "__main__":
    main()
