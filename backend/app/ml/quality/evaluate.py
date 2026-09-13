"""
Evaluation utilities for the PromptLens quality ML model (Step 12).

Computes regression and secondary classification metrics.
"""
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

from app.ml.quality.categories import QualityLabel, score_to_label


def regression_metrics(
    actual: list[float],
    predicted: list[float],
) -> dict[str, Any]:
    """Compute regression evaluation metrics.

    Args:
        actual: Ground-truth quality scores.
        predicted: Model-predicted quality scores.

    Returns:
        Dictionary with MAE, RMSE, R2, Pearson r, Spearman rho,
        mean actual, mean predicted, and error distribution statistics.
    """
    actual_arr = np.asarray(actual, dtype=np.float64)
    predicted_arr = np.asarray(predicted, dtype=np.float64)
    errors = predicted_arr - actual_arr

    mae = float(mean_absolute_error(actual_arr, predicted_arr))
    rmse = float(np.sqrt(mean_squared_error(actual_arr, predicted_arr)))
    r2 = float(r2_score(actual_arr, predicted_arr))

    pearson_r, pearson_p = pearsonr(actual_arr, predicted_arr)
    spearman_rho, spearman_p = spearmanr(actual_arr, predicted_arr)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p": round(float(pearson_p), 6),
        "spearman_rho": round(float(spearman_rho), 4),
        "spearman_p": round(float(spearman_p), 6),
        "mean_actual": round(float(actual_arr.mean()), 4),
        "mean_predicted": round(float(predicted_arr.mean()), 4),
        "error_std": round(float(errors.std()), 4),
        "error_min": round(float(errors.min()), 4),
        "error_max": round(float(errors.max()), 4),
        "n_samples": len(actual),
    }


def classification_metrics_from_scores(
    actual_scores: list[float],
    predicted_scores: list[float],
) -> dict[str, Any]:
    """Compute secondary classification metrics by mapping scores to labels.

    Predicted scores are converted to quality labels using the centralized
    score_to_label mapping.  This is a secondary diagnostic view only;
    the primary model target is the numeric score.

    Args:
        actual_scores: Ground-truth quality scores.
        predicted_scores: Model-predicted quality scores.

    Returns:
        Dictionary with accuracy, macro F1, weighted F1, and per-class metrics.
    """
    label_order = [label.value for label in QualityLabel]
    actual_labels = [score_to_label(s).value for s in actual_scores]
    predicted_labels = [score_to_label(s).value for s in predicted_scores]

    precision, recall, f1, support = precision_recall_fscore_support(
        actual_labels,
        predicted_labels,
        labels=label_order,
        zero_division=0,
    )

    return {
        "accuracy": round(float(accuracy_score(actual_labels, predicted_labels)), 4),
        "macro_f1": round(float(f1_score(actual_labels, predicted_labels, average="macro", labels=label_order, zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(actual_labels, predicted_labels, average="weighted", labels=label_order, zero_division=0)), 4),
        "per_class": {
            label: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i, label in enumerate(label_order)
        },
    }
