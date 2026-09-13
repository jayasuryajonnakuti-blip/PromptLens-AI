"""Finding and recommendation generator for PromptLens (Step 13).

Produces traceable, rule-based findings and prioritized recommendations
from preprocessing signals, NLP indicators, and dimension scores.
"""
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app.schemas.nlp import NlpAnalysisResult
from app.schemas.preprocessing import PromptPreprocessingResult
from app.scoring.dimensions import DimensionEvaluation


class FindingSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class ScoringFinding:
    type: str
    severity: FindingSeverity
    dimension: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "severity": self.severity.value,
            "dimension": self.dimension,
            "message": self.message,
        }


def generate_findings(
    prep: PromptPreprocessingResult,
    nlp: NlpAnalysisResult,
    dimensions: dict[str, DimensionEvaluation],
) -> list[ScoringFinding]:
    """Generate rule-traceable findings from the analysis signals and dimensions."""
    findings: list[ScoringFinding] = []

    if prep.flags.is_empty:
        findings.append(
            ScoringFinding(
                type="empty_prompt",
                severity=FindingSeverity.ERROR,
                dimension="completeness",
                message="Prompt is empty or contains only whitespace.",
            )
        )
        return findings

    if prep.flags.is_near_empty:
        findings.append(
            ScoringFinding(
                type="near_empty_prompt",
                severity=FindingSeverity.WARNING,
                dimension="completeness",
                message="Prompt is extremely brief (< 3 words) and lacks necessary parameters or instructions.",
            )
        )

    if not prep.output_formats and "output_format" not in prep.sections:
        findings.append(
            ScoringFinding(
                type="missing_output_format",
                severity=FindingSeverity.INFO,
                dimension="output_format",
                message="No explicit deliverable format specified (e.g. JSON, table, or markdown).",
            )
        )

    if "constraints" not in prep.sections:
        findings.append(
            ScoringFinding(
                type="missing_constraints",
                severity=FindingSeverity.INFO,
                dimension="constraints",
                message="No negative constraints, length limits, or operational boundaries specified.",
            )
        )

    if "audience" not in prep.sections:
        findings.append(
            ScoringFinding(
                type="missing_audience",
                severity=FindingSeverity.INFO,
                dimension="audience",
                message="Target audience proficiency level is not explicitly declared.",
            )
        )

    if nlp.pos_counts.verbs == 0:
        findings.append(
            ScoringFinding(
                type="no_action_verbs",
                severity=FindingSeverity.WARNING,
                dimension="actionability",
                message="No active verbs detected; task objective may be ambiguous to execute.",
            )
        )

    if len(prep.repeated_phrases) > 2:
        findings.append(
            ScoringFinding(
                type="excessive_repetition",
                severity=FindingSeverity.WARNING,
                dimension="clarity",
                message=f"Excessive phrase repetition detected ({len(prep.repeated_phrases)} repeated phrases).",
            )
        )

    if prep.flags.is_very_long and len(prep.sections) <= 1:
        findings.append(
            ScoringFinding(
                type="unstructured_long_text",
                severity=FindingSeverity.WARNING,
                dimension="clarity",
                message="Prompt exceeds 2,000 words without structured section headings or bullet points.",
            )
        )

    if "JSON" in prep.output_formats and "plain_text" in prep.output_formats:
        findings.append(
            ScoringFinding(
                type="conflicting_formats",
                severity=FindingSeverity.WARNING,
                dimension="consistency",
                message="Potentially conflicting output formats detected: JSON and plain text.",
            )
        )

    return findings


def generate_recommendations(
    dimensions: dict[str, DimensionEvaluation],
    findings: list[ScoringFinding],
    limit: int = 4,
) -> list[str]:
    """Generate prioritized, actionable recommendations from the weakest dimensions."""
    # Sort dimensions by lowest score first
    sorted_dims = sorted(dimensions.values(), key=lambda d: d.score)

    recommendations: list[str] = []
    seen: set[str] = set()

    for dim in sorted_dims:
        if dim.score < 75.0 and dim.recommendation not in seen:
            seen.add(dim.recommendation)
            recommendations.append(dim.recommendation)
            if len(recommendations) >= limit:
                break

    if not recommendations:
        recommendations.append("Prompt is well-structured and comprehensive across evaluated dimensions.")

    return recommendations
