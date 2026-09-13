"""
Centralized PromptLens quality category mapping.

These are PromptLens heuristic quality categories and do not represent
scientifically universal quality thresholds.

    POOR      = 0-39
    FAIR      = 40-59
    GOOD      = 60-74
    STRONG    = 75-89
    EXCELLENT = 90-100
"""
from enum import StrEnum


class QualityLabel(StrEnum):
    POOR = "POOR"
    FAIR = "FAIR"
    GOOD = "GOOD"
    STRONG = "STRONG"
    EXCELLENT = "EXCELLENT"


# Ordered from highest threshold to lowest so the first match wins.
_THRESHOLDS: tuple[tuple[int, QualityLabel], ...] = (
    (90, QualityLabel.EXCELLENT),
    (75, QualityLabel.STRONG),
    (60, QualityLabel.GOOD),
    (40, QualityLabel.FAIR),
    (0, QualityLabel.POOR),
)

SUPPORTED_LABELS: frozenset[str] = frozenset(label.value for label in QualityLabel)


def score_to_label(score: float) -> QualityLabel:
    """Map a 0-100 quality score to the corresponding PromptLens quality label.

    Boundaries are inclusive on the lower end:
        [90, 100] -> EXCELLENT
        [75,  89] -> STRONG
        [60,  74] -> GOOD
        [40,  59] -> FAIR
        [0,   39] -> POOR

    Args:
        score: A numeric quality score. Values outside [0, 100] are accepted
               but clamped to the nearest boundary label.

    Returns:
        The matching QualityLabel.
    """
    clamped = max(0.0, min(100.0, float(score)))
    for threshold, label in _THRESHOLDS:
        if clamped >= threshold:
            return label
    return QualityLabel.POOR


def label_score_is_consistent(score: float, label: str) -> bool:
    """Return True when label matches the category that score maps to.

    Used during dataset validation to catch annotation errors.
    """
    return score_to_label(score).value == label
