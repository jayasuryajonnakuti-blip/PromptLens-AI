"""Evidence generation for PromptLens AI Critic (Step 17).

Follows evidence-first principles:
1. Explicit prompt text observations
2. Preprocessing & lexical comparisons
3. Semantic embedding similarity with explicit caveat:
   "Semantic similarity is supporting evidence and is not, by itself, proof of intent preservation."
4. Intent classification comparison
5. Scoring & dimension comparisons
6. Deterministic check findings
7. LLM inference findings
"""
from __future__ import annotations

from app.critic.context import CriticContext
from app.critic.schemas import (
    CriticEvidenceItem,
    CriticIssue,
    EvidenceSource,
    EvidenceType,
)


def build_deterministic_evidence(
    ctx: CriticContext,
    orig_prompt: str,
    opt_prompt: str,
    issues: list[CriticIssue],
) -> list[CriticEvidenceItem]:
    """Assemble prioritized evidence items from prompts, pipeline signals, and deterministic checks."""
    items: list[CriticEvidenceItem] = []

    # 1. Direct text observations
    items.append(
        CriticEvidenceItem(
            source=EvidenceSource.ORIGINAL,
            type=EvidenceType.OBSERVATION,
            statement=f"Original prompt has {ctx.word_count_original} words ({ctx.char_count_original} characters).",
            confidence=1.0,
        )
    )
    items.append(
        CriticEvidenceItem(
            source=EvidenceSource.OPTIMIZED,
            type=EvidenceType.OBSERVATION,
            statement=f"Optimized prompt has {ctx.word_count_optimized} words ({ctx.char_count_optimized} characters).",
            confidence=1.0,
        )
    )

    # 2. Preprocessing & structure comparison
    if ctx.formats_original:
        items.append(
            CriticEvidenceItem(
                source=EvidenceSource.PREPROCESSING,
                type=EvidenceType.COMPARISON,
                statement=f"Original formats: {ctx.formats_original}; Optimized formats: {ctx.formats_optimized}.",
                confidence=0.92,
            )
        )

    if ctx.has_role_original or ctx.has_role_optimized:
        role_state = (
            "preserved" if ctx.has_role_original == ctx.has_role_optimized else "altered"
        )
        items.append(
            CriticEvidenceItem(
                source=EvidenceSource.PREPROCESSING,
                type=EvidenceType.COMPARISON,
                statement=f"Role/persona definition is {role_state} in the optimized prompt.",
                confidence=0.88,
            )
        )

    # 3. Intent comparison
    items.append(
        CriticEvidenceItem(
            source=EvidenceSource.INTENT,
            type=EvidenceType.COMPARISON,
            statement=(
                f"Intent classification: original '{ctx.intent_original}' "
                f"({ctx.intent_confidence_original:.2f}) vs optimized '{ctx.intent_optimized}' "
                f"({ctx.intent_confidence_optimized:.2f})."
            ),
            confidence=0.85,
        )
    )

    # 4. Semantic similarity with explicit documentation caveat
    items.append(
        CriticEvidenceItem(
            source=EvidenceSource.EMBEDDING,
            type=EvidenceType.COMPARISON,
            statement=(
                f"Semantic cosine similarity: {ctx.semantic_similarity:.4f}. "
                "(Semantic similarity is supporting evidence and is not, by itself, proof of intent preservation.)"
            ),
            confidence=0.90,
        )
    )

    # 5. Scoring & dimension comparison
    if ctx.scoring_available:
        delta_sign = "+" if ctx.overall_score_delta >= 0 else ""
        items.append(
            CriticEvidenceItem(
                source=EvidenceSource.SCORING,
                type=EvidenceType.COMPARISON,
                statement=(
                    f"Overall quality score changed from {ctx.overall_score_original:.1f} "
                    f"to {ctx.overall_score_optimized:.1f} ({delta_sign}{ctx.overall_score_delta:.1f})."
                ),
                confidence=0.85,
            )
        )

        items.append(
            CriticEvidenceItem(
                source=EvidenceSource.SCORING,
                type=EvidenceType.COMPARISON,
                statement=(
                    f"Dimension deltas: clarity ({ctx.clarity_delta:+.1f}), "
                    f"specificity ({ctx.specificity_delta:+.1f}), "
                    f"ambiguity ({ctx.ambiguity_delta:+.1f}), "
                    f"completeness ({ctx.completeness_delta:+.1f})."
                ),
                confidence=0.82,
            )
        )

    # 6. Issue evidence
    for issue in issues:
        items.append(
            CriticEvidenceItem(
                source=EvidenceSource.OPTIMIZED,
                type=EvidenceType.COMPARISON,
                statement=f"[{issue.severity.value}] {issue.type}: {issue.description}",
                confidence=0.88 if issue.severity.value == "ERROR" else 0.78,
            )
        )

    return items


def build_llm_evidence(
    llm_critique: dict,
) -> list[CriticEvidenceItem]:
    """Extract LLM-derived evidence items marked as INFERENCE with conservative confidence."""
    items: list[CriticEvidenceItem] = []

    for issue in llm_critique.get("issues", []):
        if isinstance(issue, dict):
            desc = issue.get("description", "")
            itype = issue.get("type", "llm_finding")
            if desc:
                items.append(
                    CriticEvidenceItem(
                        source=EvidenceSource.LLM,
                        type=EvidenceType.INFERENCE,
                        statement=f"LLM observation ({itype}): {desc}",
                        confidence=0.65,
                    )
                )

    return items
