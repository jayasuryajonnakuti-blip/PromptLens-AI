"""Evidence generation utilities for PromptLens AI Analyzer (Step 15).

Converts deterministic upstream pipeline signals into structured EvidenceItem
instances.  Evidence items are later attached to the AnalyzerAnalysis and serve
as the audit trail connecting analysis conclusions to their sources.

Evidence priority order (high → low):
  1. Explicit prompt text
  2. Preprocessing (Step 8)
  3. spaCy NLP (Step 9)
  4. Intent Classification (Step 10)
  5. Quality ML (Step 12)
  6. Scoring & Fusion (Step 13)
  7. LLM inference (Step 14)
"""
from __future__ import annotations

from app.analyzer.config import INTENT_CONFIDENCE_THRESHOLD
from app.analyzer.context import AnalyzerContext
from app.analyzer.schemas import EvidenceItem, EvidenceSource, EvidenceType


def build_deterministic_evidence(ctx: AnalyzerContext) -> list[EvidenceItem]:
    """Build an ordered list of EvidenceItems from deterministic pipeline signals.

    Items are ordered by evidence priority (preprocessing first, scoring last).
    The caller appends any LLM-derived evidence after this list.
    """
    items: list[EvidenceItem] = []

    # --- Step 8: Preprocessing observations ---
    if ctx.word_count > 0:
        items.append(
            EvidenceItem(
                source=EvidenceSource.PREPROCESSING,
                type=EvidenceType.OBSERVATION,
                statement=f"Prompt contains {ctx.word_count} words across {ctx.sentence_count} sentence(s).",
                confidence=0.95,
            )
        )

    if ctx.output_formats:
        items.append(
            EvidenceItem(
                source=EvidenceSource.PREPROCESSING,
                type=EvidenceType.OBSERVATION,
                statement=f"Output format specifier(s) detected: {', '.join(ctx.output_formats)}.",
                confidence=0.90,
            )
        )

    if ctx.sections_detected:
        items.append(
            EvidenceItem(
                source=EvidenceSource.PREPROCESSING,
                type=EvidenceType.OBSERVATION,
                statement=f"Structural sections found: {', '.join(ctx.sections_detected)}.",
                confidence=0.90,
            )
        )

    if ctx.has_role_definition:
        items.append(
            EvidenceItem(
                source=EvidenceSource.PREPROCESSING,
                type=EvidenceType.OBSERVATION,
                statement="Role-definition phrase detected (e.g. 'You are', 'Act as').",
                confidence=0.88,
            )
        )

    if ctx.has_constraints:
        items.append(
            EvidenceItem(
                source=EvidenceSource.PREPROCESSING,
                type=EvidenceType.OBSERVATION,
                statement="Explicit constraint or restriction language detected.",
                confidence=0.85,
            )
        )

    if ctx.has_examples:
        items.append(
            EvidenceItem(
                source=EvidenceSource.PREPROCESSING,
                type=EvidenceType.OBSERVATION,
                statement="Example or sample content is present in the prompt.",
                confidence=0.85,
            )
        )

    # --- Step 9: NLP observations ---
    if ctx.entity_count > 0:
        labels_str = ", ".join(ctx.entity_labels[:5])
        items.append(
            EvidenceItem(
                source=EvidenceSource.NLP,
                type=EvidenceType.OBSERVATION,
                statement=f"{ctx.entity_count} named entity/entities detected: {labels_str}.",
                confidence=0.80,
            )
        )

    if ctx.verb_count > 0 or ctx.noun_count > 0:
        items.append(
            EvidenceItem(
                source=EvidenceSource.NLP,
                type=EvidenceType.OBSERVATION,
                statement=(
                    f"Lexical composition: {ctx.verb_count} verb(s), {ctx.noun_count} noun(s). "
                    f"Average sentence length: {ctx.avg_sentence_length:.1f} words."
                ),
                confidence=0.80,
            )
        )

    if ctx.has_imperative:
        items.append(
            EvidenceItem(
                source=EvidenceSource.NLP,
                type=EvidenceType.INFERENCE,
                statement="Imperative sentence structure suggests directive or instructional intent.",
                confidence=0.65,
            )
        )

    # --- Step 10: Intent classification ---
    if ctx.intent_available:
        conf = ctx.intent_confidence
        if conf >= INTENT_CONFIDENCE_THRESHOLD:
            items.append(
                EvidenceItem(
                    source=EvidenceSource.INTENT,
                    type=EvidenceType.INFERENCE,
                    statement=(
                        f"Intent classifier predicts '{ctx.intent_label}' "
                        f"with confidence {conf:.2f}."
                    ),
                    confidence=min(conf, 0.90),
                )
            )
        else:
            items.append(
                EvidenceItem(
                    source=EvidenceSource.INTENT,
                    type=EvidenceType.INFERENCE,
                    statement=(
                        f"Intent classifier produced low-confidence prediction: "
                        f"'{ctx.intent_label}' ({conf:.2f}) — treat as uncertain."
                    ),
                    confidence=max(conf * 0.5, 0.15),
                )
            )

    # --- Step 12: Quality ML ---
    if ctx.quality_available:
        items.append(
            EvidenceItem(
                source=EvidenceSource.QUALITY_ML,
                type=EvidenceType.INFERENCE,
                statement=(
                    f"Quality ML model scores the prompt at {ctx.quality_score:.1f}/100 "
                    f"(category: {ctx.quality_category})."
                ),
                confidence=0.75,
            )
        )

    # --- Step 13: Scoring & Fusion ---
    if ctx.step13_available:
        items.append(
            EvidenceItem(
                source=EvidenceSource.SCORING,
                type=EvidenceType.INFERENCE,
                statement=(
                    f"Scoring & Fusion engine assigns an overall score of "
                    f"{ctx.step13_overall_score:.1f}/100 ({ctx.step13_category})."
                ),
                confidence=0.80,
            )
        )
        # Add individual weak/strong dimension observations
        for dim in ctx.top_dimensions[:3]:
            dim_score = dim.get("score", 0.0)
            dim_name = dim.get("dimension", "unknown")
            dim_status = dim.get("status", "")
            direction = "low" if dim_score < 50 else "adequate"
            items.append(
                EvidenceItem(
                    source=EvidenceSource.SCORING,
                    type=EvidenceType.OBSERVATION,
                    statement=(
                        f"Dimension '{dim_name}' scores {dim_score:.1f}/100 "
                        f"(status: {dim_status}) — {direction}."
                    ),
                    confidence=0.72,
                )
            )

    return items


def build_llm_evidence_items(
    ambiguities: list[dict],
    missing_info: list[dict],
    contradictions: list[dict],
) -> list[EvidenceItem]:
    """Build LLM-sourced evidence items from the structured LLM output.

    These are always INFERENCE type and carry lower default confidence than
    deterministic observations, reflecting that LLM reasoning is model-assisted.
    """
    items: list[EvidenceItem] = []

    for amb in ambiguities:
        issue = amb.get("issue", "")
        conf = float(amb.get("confidence", 0.60))
        if issue:
            items.append(
                EvidenceItem(
                    source=EvidenceSource.LLM,
                    type=EvidenceType.INFERENCE,
                    statement=f"LLM-identified ambiguity: {issue}",
                    confidence=min(max(conf, 0.10), 0.90),
                )
            )

    for mi in missing_info:
        item_desc = mi.get("item", "")
        conf = float(mi.get("confidence", 0.60))
        if item_desc:
            items.append(
                EvidenceItem(
                    source=EvidenceSource.LLM,
                    type=EvidenceType.INFERENCE,
                    statement=f"LLM-identified missing information: {item_desc}",
                    confidence=min(max(conf, 0.10), 0.90),
                )
            )

    for contra in contradictions:
        issue = contra.get("issue", "")
        conf = float(contra.get("confidence", 0.60))
        if issue:
            items.append(
                EvidenceItem(
                    source=EvidenceSource.LLM,
                    type=EvidenceType.INFERENCE,
                    statement=f"LLM-identified contradiction: {issue}",
                    confidence=min(max(conf, 0.10), 0.90),
                )
            )

    return items
