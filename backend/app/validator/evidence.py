"""Evidence assembly for PromptLens AI Validator (Step 18).

Provides traceable validation evidence across:
- Deterministic checks
- NLP & Preprocessing
- Semantic embeddings (with required disclaimer)
- Step 13 Scoring
- Step 17 Critic findings
- LLM validation inferences
"""
from __future__ import annotations

from app.validator.context import ValidatorContext
from app.validator.schemas import (
    EvidenceSource,
    ValidationEvidence,
    ValidationIssue,
)


def build_validation_evidence(
    ctx: ValidatorContext,
    issues: list[ValidationIssue],
) -> list[ValidationEvidence]:
    """Assemble prioritized validation evidence items."""
    evidence_list: list[ValidationEvidence] = []

    # 1. Lexical and length evidence
    evidence_list.append(
        ValidationEvidence(
            source=EvidenceSource.DETERMINISTIC,
            description=(
                f"Original word count: {ctx.original_word_count}; "
                f"Optimized word count: {ctx.optimized_word_count} "
                f"(expansion ratio: {ctx.expansion_ratio:.2f}x)."
            ),
            metric_name="expansion_ratio",
            metric_value=round(ctx.expansion_ratio, 2),
            confidence=1.0,
        )
    )

    # 2. Output format evidence
    if ctx.original_formats:
        evidence_list.append(
            ValidationEvidence(
                source=EvidenceSource.NLP,
                description=f"Original output formats: {ctx.original_formats}; Optimized formats: {ctx.optimized_formats}.",
                metric_name="format_count",
                metric_value=float(len(ctx.original_formats)),
                confidence=0.95,
            )
        )

    # 3. Intent classification evidence
    evidence_list.append(
        ValidationEvidence(
            source=EvidenceSource.NLP,
            description=(
                f"Intent classification: original '{ctx.original_intent}' (conf: {ctx.original_intent_confidence:.2f}) "
                f"vs optimized '{ctx.optimized_intent}' (conf: {ctx.optimized_intent_confidence:.2f})."
            ),
            metric_name="intent_confidence",
            metric_value=round(ctx.original_intent_confidence, 2),
            confidence=0.90,
        )
    )

    # 4. Semantic similarity with explicit mandatory disclaimer
    evidence_list.append(
        ValidationEvidence(
            source=EvidenceSource.EMBEDDING,
            description=(
                f"Semantic embedding similarity: {ctx.semantic_similarity:.4f}. "
                "(Semantic similarity is supporting evidence and is not, by itself, proof of intent preservation.)"
            ),
            metric_name="semantic_similarity",
            metric_value=round(ctx.semantic_similarity, 4),
            confidence=0.90,
        )
    )

    # 5. Scoring evidence
    if ctx.scoring_available:
        delta_sign = "+" if ctx.score_delta >= 0 else ""
        evidence_list.append(
            ValidationEvidence(
                source=EvidenceSource.SCORING,
                description=(
                    f"Overall quality score changed from {ctx.original_score:.1f} "
                    f"to {ctx.optimized_score:.1f} ({delta_sign}{ctx.score_delta:.1f})."
                ),
                metric_name="score_delta",
                metric_value=round(ctx.score_delta, 1),
                confidence=0.88,
            )
        )

    # 6. Critic findings evidence
    if ctx.critic_provided:
        evidence_list.append(
            ValidationEvidence(
                source=EvidenceSource.CRITIC,
                description=f"Step 17 Critic returned decision '{ctx.critic_decision}' with critique score {ctx.critic_score}.",
                metric_name="critic_score",
                metric_value=ctx.critic_score,
                confidence=0.92,
            )
        )

    # 7. Issues evidence
    for issue in issues:
        evidence_list.append(
            ValidationEvidence(
                source=EvidenceSource.DETERMINISTIC,
                description=f"[{issue.severity.value}] {issue.code}: {issue.message}",
                metric_name=issue.code,
                metric_value=None,
                confidence=0.95 if issue.severity.value == "ERROR" else 0.85,
            )
        )

    return evidence_list


def build_llm_validation_evidence(llm_data: dict) -> list[ValidationEvidence]:
    """Extract LLM-derived validation evidence items."""
    evidence_list: list[ValidationEvidence] = []
    justification = llm_data.get("justification")
    if justification:
        evidence_list.append(
            ValidationEvidence(
                source=EvidenceSource.LLM,
                description=f"LLM validation evaluation: {justification}",
                metric_name="llm_safety_assessment",
                metric_value=float(llm_data.get("safety_score", 70.0)),
                confidence=0.70,
            )
        )
    return evidence_list
