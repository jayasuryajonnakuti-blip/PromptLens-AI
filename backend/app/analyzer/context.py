"""AnalyzerContext: aggregates upstream signals into a compact typed bundle.

This module is responsible for:
1. Calling each upstream service (Steps 8-13) and capturing results.
2. Producing a compact, serialisable context dict for the analyzer prompt.
3. Never propagating raw embedding vectors (too large for context budgets).

Each upstream call is individually guarded — a failure in any one service
degrades gracefully rather than aborting the entire analysis.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class AnalyzerContext:
    """Compact, typed bundle of upstream pipeline results."""

    # --- Step 8: Preprocessing ---
    word_count: int = 0
    sentence_count: int = 0
    output_formats: list[str] = field(default_factory=list)
    sections_detected: list[str] = field(default_factory=list)
    has_role_definition: bool = False
    has_constraints: bool = False
    has_examples: bool = False

    # --- Step 9: NLP ---
    entity_count: int = 0
    entity_labels: list[str] = field(default_factory=list)
    verb_count: int = 0
    noun_count: int = 0
    avg_sentence_length: float = 0.0
    has_imperative: bool = False

    # --- Step 10: Intent ---
    intent_label: str = ""
    intent_confidence: float = 0.0
    intent_available: bool = False

    # --- Step 12: Quality ML ---
    quality_score: float = 0.0
    quality_category: str = ""
    quality_available: bool = False

    # --- Step 13: Scoring & Fusion ---
    step13_overall_score: float = 0.0
    step13_category: str = ""
    step13_available: bool = False
    # Top N dimension summaries (compact — no raw vectors)
    top_dimensions: list[dict[str, Any]] = field(default_factory=list)
    # Step 13 recommendations (at most 5 to avoid bloating LLM context)
    step13_recommendations: list[str] = field(default_factory=list)

    def to_compact_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable compact representation for LLM context."""
        data: dict[str, Any] = {}

        # Preprocessing
        data["word_count"] = self.word_count
        data["sentence_count"] = self.sentence_count
        if self.output_formats:
            data["output_formats"] = self.output_formats
        if self.sections_detected:
            data["sections_detected"] = self.sections_detected
        structural_flags: list[str] = []
        if self.has_role_definition:
            structural_flags.append("role_definition_present")
        if self.has_constraints:
            structural_flags.append("constraints_present")
        if self.has_examples:
            structural_flags.append("examples_present")
        if structural_flags:
            data["structural_flags"] = structural_flags

        # NLP
        if self.entity_labels:
            data["named_entities"] = self.entity_labels[:10]  # cap for readability
        data["verb_count"] = self.verb_count
        data["noun_count"] = self.noun_count
        if self.avg_sentence_length > 0:
            data["avg_sentence_length"] = round(self.avg_sentence_length, 1)
        if self.has_imperative:
            data["imperative_detected"] = True

        # Intent
        if self.intent_available:
            data["detected_intent"] = self.intent_label
            data["intent_confidence"] = round(self.intent_confidence, 3)

        # Quality ML
        if self.quality_available:
            data["baseline_quality_score"] = round(self.quality_score, 1)
            data["quality_category"] = self.quality_category

        # Step 13 scoring
        if self.step13_available:
            data["step13_overall_score"] = round(self.step13_overall_score, 1)
            data["step13_category"] = self.step13_category
            if self.top_dimensions:
                data["dimension_highlights"] = self.top_dimensions
            if self.step13_recommendations:
                data["step13_recommendations"] = self.step13_recommendations[:5]

        return data


def build_analyzer_context(prompt: str) -> AnalyzerContext:
    """Run all upstream services and return a populated AnalyzerContext.

    Each upstream call is individually guarded so that a failure in one
    service does not abort the entire analysis pipeline.
    """
    ctx = AnalyzerContext()

    # --- Step 8: Preprocessing ---
    try:
        from app.services.preprocessing_service import preprocess_prompt

        prep = preprocess_prompt(prompt)
        ctx.word_count = getattr(prep.text, "word_count", 0)
        ctx.sentence_count = getattr(prep.text, "sentence_count", 0)
        ctx.output_formats = list(prep.output_formats or [])
        ctx.sections_detected = list(prep.sections or [])
        # Detect structural features from preprocessing flags
        text_lower = prompt.lower()
        ctx.has_role_definition = any(
            kw in text_lower for kw in ("you are", "act as", "your role", "as a ")
        )
        ctx.has_constraints = any(
            kw in text_lower
            for kw in ("must not", "do not", "avoid", "constraint", "limit", "restrict")
        )
        ctx.has_examples = any(
            kw in text_lower for kw in ("for example", "e.g.", "such as", "sample", "example:")
        )
    except Exception as exc:
        LOGGER.debug("Preprocessing unavailable in analyzer context: %s", exc)

    # --- Step 9: NLP ---
    try:
        from app.services.nlp_service import analyze_prompt_nlp

        nlp = analyze_prompt_nlp(prompt)
        if hasattr(nlp, "entities"):
            ctx.entity_count = len(nlp.entities or [])
            ctx.entity_labels = [
                e.get("label", "") if isinstance(e, dict) else str(e)
                for e in (nlp.entities or [])
            ][:15]
        if hasattr(nlp, "pos_tags"):
            pos_tags = nlp.pos_tags or []
            ctx.verb_count = sum(
                1 for p in pos_tags if (isinstance(p, dict) and p.get("pos") == "VERB")
            )
            ctx.noun_count = sum(
                1 for p in pos_tags if (isinstance(p, dict) and p.get("pos") in ("NOUN", "PROPN"))
            )
        if hasattr(nlp, "sentences"):
            sents = nlp.sentences or []
            if sents:
                lengths = [len(s.split()) for s in sents]
                ctx.avg_sentence_length = sum(lengths) / len(lengths)
        # Imperative heuristic: first token of each sentence is a base verb
        ctx.has_imperative = ctx.verb_count > 0 and ctx.noun_count < ctx.verb_count * 3
    except Exception as exc:
        LOGGER.debug("NLP unavailable in analyzer context: %s", exc)

    # --- Step 10: Intent ---
    try:
        from app.services.intent_service import classify_prompt

        intent_resp = classify_prompt(prompt)
        ctx.intent_label = intent_resp.intent.value
        ctx.intent_confidence = float(intent_resp.confidence)
        ctx.intent_available = True
    except Exception as exc:
        LOGGER.debug("Intent classification unavailable in analyzer context: %s", exc)

    # --- Step 12: Quality ML ---
    try:
        from app.services.quality_service import predict_quality

        quality = predict_quality(prompt)
        ctx.quality_score = float(quality.score)
        ctx.quality_category = quality.label.value
        ctx.quality_available = True
    except Exception as exc:
        LOGGER.debug("Quality ML unavailable in analyzer context: %s", exc)

    # --- Step 13: Scoring & Fusion ---
    try:
        from app.services.scoring_service import score_prompt

        score_resp = score_prompt(prompt)
        ctx.step13_overall_score = float(score_resp.overall_score.score)
        ctx.step13_category = score_resp.overall_score.category
        ctx.step13_available = True

        # Select top dimensions by score (lowest scored = most in need of improvement)
        from app.analyzer.config import MAX_DIMENSION_SUMMARIES

        dims = sorted(
            score_resp.dimensions.items(),
            key=lambda kv: kv[1].score,
        )
        ctx.top_dimensions = [
            {
                "dimension": name,
                "score": round(dim.score, 1),
                "status": dim.status,
                "reason": dim.reason,
            }
            for name, dim in dims[:MAX_DIMENSION_SUMMARIES]
        ]
        ctx.step13_recommendations = list(score_resp.recommendations or [])[:5]
    except Exception as exc:
        LOGGER.debug("Scoring service unavailable in analyzer context: %s", exc)

    return ctx
