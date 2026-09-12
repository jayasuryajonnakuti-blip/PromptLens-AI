from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(
    expected: list[str], predicted: list[str], labels: tuple[str, ...]
) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected, predicted, labels=list(labels), zero_division=0
    )
    _, _, _, support = precision_recall_fscore_support(
        expected, predicted, labels=list(labels), zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        expected, predicted, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        expected, predicted, average="weighted", zero_division=0
    )
    matrix = confusion_matrix(expected, predicted, labels=list(labels)).tolist()
    return {
        "accuracy": round(float(accuracy_score(expected, predicted)), 4),
        "precision": {label: round(float(value), 4) for label, value in zip(labels, precision)},
        "recall": {label: round(float(value), 4) for label, value in zip(labels, recall)},
        "f1": {label: round(float(value), 4) for label, value in zip(labels, f1)},
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_precision), 4),
        "weighted_recall": round(float(weighted_recall), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "support": {label: int(value) for label, value in zip(labels, support)},
        "confusion_matrix": matrix,
        "confusion_matrix_labels": list(labels),
    }