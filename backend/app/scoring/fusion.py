"""Central scoring and fusion engine for PromptLens (Step 13).

Combines:
1. Deterministic rules (Step 8)
2. spaCy NLP analysis (Step 9)
3. Quality ML regression prediction (Step 12)
4. Missing LLM handling via dynamic weight renormalization

Computes:
- Overall score in [0.0, 100.0]
- Quality status category
- All 12 quality dimensions
- Traceable findings
- Prioritized recommendations
"""
from dataclasses import dataclass
from typing import Any

from app.ml.quality.categories import QualityLabel, score_to_label
from app.schemas.nlp import NlpAnalysisResult
from app.schemas.preprocessing import PromptPreprocessingResult
from app.schemas.quality import QualityPredictionResponse
from app.scoring.config import DEFAULT_SCORING_CONFIG, ScoringConfig
from app.scoring.dimensions import DimensionEvaluation, evaluate_dimensions
from app.scoring.findings import ScoringFinding, generate_findings, generate_recommendations
from app.scoring.signals import ScoringSignal, compute_nlp_score, compute_rule_score


@dataclass(frozen=True)
class OverallScoreData:
    score: float
    category: QualityLabel
    status: str
    explanation: str


@dataclass(frozen=True)
class FusedScoringResult:
    overall_score: OverallScoreData
    dimensions: dict[str, DimensionEvaluation]
    signals: dict[str, ScoringSignal]
    findings: list[ScoringFinding]
    recommendations: list[str]
    scoring_version: str


def fuse_signals(
    prompt: str,
    prep: PromptPreprocessingResult,
    nlp: NlpAnalysisResult,
    quality_pred: QualityPredictionResponse,
    config: ScoringConfig = DEFAULT_SCORING_CONFIG,
) -> FusedScoringResult:
    """Fuse all available analysis signals into a unified scoring evaluation."""
    # Compute component scores
    rule_score = compute_rule_score(prep)
    nlp_score = compute_nlp_score(nlp)
    quality_ml_score = quality_pred.score

    # Determine available signals (LLM is unavailable in Step 13)
    available_signal_names = {"rules", "nlp", "quality_ml"}
    effective_weights = config.calculate_effective_weights(available_signal_names)

    # Calculate overall fused score
    raw_fused = (
        rule_score * effective_weights["rules"]
        + nlp_score * effective_weights["nlp"]
        + quality_ml_score * effective_weights["quality_ml"]
    )
    overall_score = max(0.0, min(100.0, round(raw_fused, 2)))
    overall_category = score_to_label(overall_score)

    # Build signal models
    signals: dict[str, ScoringSignal] = {
        "rules": ScoringSignal(
            source="rules",
            raw_score=rule_score,
            configured_weight=config.weights["rules"],
            effective_weight=effective_weights["rules"],
            available=True,
            explanation=f"Deterministic rule score based on structural elements and directives: {rule_score}/100.",
        ),
        "nlp": ScoringSignal(
            source="nlp",
            raw_score=nlp_score,
            configured_weight=config.weights["nlp"],
            effective_weight=effective_weights["nlp"],
            available=True,
            explanation=f"Linguistic syntactic score based on POS balance and complexity: {nlp_score}/100.",
        ),
        "quality_ml": ScoringSignal(
            source="quality_ml",
            raw_score=quality_ml_score,
            configured_weight=config.weights["quality_ml"],
            effective_weight=effective_weights["quality_ml"],
            available=True,
            explanation=f"Learned baseline regression prediction: {quality_ml_score}/100.",
        ),
        "llm": ScoringSignal(
            source="llm",
            raw_score=None,
            configured_weight=config.weights["llm"],
            effective_weight=0.0,
            available=False,
            explanation="Local LLM signal is not yet configured or enabled in Step 13.",
        ),
    }

    # Evaluate 12 dimensions
    dimensions = evaluate_dimensions(prep, nlp, quality_ml_score)

    # Generate findings and recommendations
    findings = generate_findings(prep, nlp, dimensions)
    recommendations = generate_recommendations(dimensions, findings)

    # Overall explanation
    explanation = (
        f"Fused score {overall_score}/100 ({overall_category.value}) derived from "
        f"Rules ({rule_score} × {effective_weights['rules']:.2%}), "
        f"NLP ({nlp_score} × {effective_weights['nlp']:.2%}), and "
        f"Quality ML ({quality_ml_score} × {effective_weights['quality_ml']:.2%}). "
        f"LLM signal is currently unavailable."
    )

    return FusedScoringResult(
        overall_score=OverallScoreData(
            score=overall_score,
            category=overall_category,
            status=overall_category.value,
            explanation=explanation,
        ),
        dimensions=dimensions,
        signals=signals,
        findings=findings,
        recommendations=recommendations,
        scoring_version=config.version,
    )
