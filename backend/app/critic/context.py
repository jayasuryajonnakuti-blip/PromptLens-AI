"""CriticContext: extracts and compares pipeline signals for original & optimized prompts.

Gathers:
- Step 8 Preprocessing metrics (word count, sentence count, formats, sections)
- Step 10 Intent classification (intent label, confidence)
- Step 11 Semantic embeddings similarity
- Step 13 Scoring & 12 Dimensions
- Calculates exact deltas for key dimensions

Each upstream call is safely guarded to prevent cascade failures.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class CriticContext:
    # --- Text / Lexical ---
    word_count_original: int = 0
    word_count_optimized: int = 0
    word_count_ratio: float = 1.0
    char_count_original: int = 0
    char_count_optimized: int = 0
    char_count_ratio: float = 1.0

    # --- Preprocessing & Structural Features ---
    formats_original: list[str] = field(default_factory=list)
    formats_optimized: list[str] = field(default_factory=list)
    has_role_original: bool = False
    has_role_optimized: bool = False
    has_constraints_original: bool = False
    has_constraints_optimized: bool = False

    # --- Step 10 Intent Classification ---
    intent_original: str = "unknown"
    intent_confidence_original: float = 0.0
    intent_optimized: str = "unknown"
    intent_confidence_optimized: float = 0.0
    intent_match: bool = True

    # --- Step 11 Semantic Similarity ---
    semantic_similarity: float = 1.0
    similarity_available: bool = False

    # --- Step 13 Scoring Overall & Dimensions ---
    overall_score_original: float = 50.0
    overall_score_optimized: float = 50.0
    overall_score_delta: float = 0.0
    scoring_available: bool = False

    # Key dimension scores (original, optimized, delta)
    clarity_original: float = 50.0
    clarity_optimized: float = 50.0
    clarity_delta: float = 0.0

    specificity_original: float = 50.0
    specificity_optimized: float = 50.0
    specificity_delta: float = 0.0

    ambiguity_original: float = 50.0
    ambiguity_optimized: float = 50.0
    ambiguity_delta: float = 0.0

    completeness_original: float = 50.0
    completeness_optimized: float = 50.0
    completeness_delta: float = 0.0

    consistency_original: float = 50.0
    consistency_optimized: float = 50.0
    consistency_delta: float = 0.0

    all_dimensions_original: dict[str, float] = field(default_factory=dict)
    all_dimensions_optimized: dict[str, float] = field(default_factory=dict)

    def to_compact_dict(self) -> dict[str, Any]:
        """Produce a compact dictionary representation for LLM prompt context."""
        return {
            "lexical_comparison": {
                "word_count_original": self.word_count_original,
                "word_count_optimized": self.word_count_optimized,
                "word_count_ratio": round(self.word_count_ratio, 2),
                "char_count_original": self.char_count_original,
                "char_count_optimized": self.char_count_optimized,
            },
            "intent_comparison": {
                "original_intent": self.intent_original,
                "original_confidence": round(self.intent_confidence_original, 3),
                "optimized_intent": self.intent_optimized,
                "optimized_confidence": round(self.intent_confidence_optimized, 3),
                "intent_match": self.intent_match,
            },
            "semantic_similarity": round(self.semantic_similarity, 4),
            "scoring_comparison": {
                "overall_score_original": round(self.overall_score_original, 1),
                "overall_score_optimized": round(self.overall_score_optimized, 1),
                "overall_delta": round(self.overall_score_delta, 1),
                "clarity": {
                    "orig": round(self.clarity_original, 1),
                    "opt": round(self.clarity_optimized, 1),
                    "delta": round(self.clarity_delta, 1),
                },
                "specificity": {
                    "orig": round(self.specificity_original, 1),
                    "opt": round(self.specificity_optimized, 1),
                    "delta": round(self.specificity_delta, 1),
                },
                "ambiguity": {
                    "orig": round(self.ambiguity_original, 1),
                    "opt": round(self.ambiguity_optimized, 1),
                    "delta": round(self.ambiguity_delta, 1),
                },
                "completeness": {
                    "orig": round(self.completeness_original, 1),
                    "opt": round(self.completeness_optimized, 1),
                    "delta": round(self.completeness_delta, 1),
                },
            },
            "structural_comparison": {
                "formats_original": self.formats_original,
                "formats_optimized": self.formats_optimized,
                "role_preserved": self.has_role_original == self.has_role_optimized,
                "constraints_present_optimized": self.has_constraints_optimized,
            },
        }


def build_critic_context(original_prompt: str, optimized_prompt: str) -> CriticContext:
    """Build a populated CriticContext comparing original and optimized prompts."""
    ctx = CriticContext()

    # 1. Lexical comparison
    words_orig = original_prompt.split()
    words_opt = optimized_prompt.split()
    ctx.word_count_original = len(words_orig)
    ctx.word_count_optimized = len(words_opt)
    ctx.word_count_ratio = (
        ctx.word_count_optimized / max(1, ctx.word_count_original)
    )

    ctx.char_count_original = len(original_prompt)
    ctx.char_count_optimized = len(optimized_prompt)
    ctx.char_count_ratio = (
        ctx.char_count_optimized / max(1, ctx.char_count_original)
    )

    # 2. Preprocessing & Structure
    try:
        from app.services.preprocessing_service import preprocess_prompt

        p_orig = preprocess_prompt(original_prompt)
        p_opt = preprocess_prompt(optimized_prompt)

        ctx.formats_original = list(p_orig.output_formats or [])
        ctx.formats_optimized = list(p_opt.output_formats or [])

        orig_lower = original_prompt.lower()
        opt_lower = optimized_prompt.lower()

        role_keywords = ("you are", "act as", "your role", "as a ")
        ctx.has_role_original = any(kw in orig_lower for kw in role_keywords)
        ctx.has_role_optimized = any(kw in opt_lower for kw in role_keywords)

        constraint_keywords = ("must not", "do not", "avoid", "constraint", "limit", "restrict", "never")
        ctx.has_constraints_original = any(kw in orig_lower for kw in constraint_keywords)
        ctx.has_constraints_optimized = any(kw in opt_lower for kw in constraint_keywords)

    except Exception as exc:
        LOGGER.debug("Preprocessing service failed in critic context: %s", exc)

    # 3. Intent classification comparison
    try:
        from app.services.intent_service import classify_prompt

        i_orig = classify_prompt(original_prompt)
        i_opt = classify_prompt(optimized_prompt)

        ctx.intent_original = i_orig.intent.value
        ctx.intent_confidence_original = float(i_orig.confidence)

        ctx.intent_optimized = i_opt.intent.value
        ctx.intent_confidence_optimized = float(i_opt.confidence)

        ctx.intent_match = (ctx.intent_original == ctx.intent_optimized)
    except Exception as exc:
        LOGGER.debug("Intent classification failed in critic context: %s", exc)

    # 4. Semantic similarity via embeddings
    try:
        from app.services.embedding_service import get_embedding_service

        emb_service = get_embedding_service()
        emb_orig = emb_service.generate_embedding(original_prompt)
        emb_opt = emb_service.generate_embedding(optimized_prompt)
        sim = emb_service.compute_similarity(emb_orig, emb_opt)

        ctx.semantic_similarity = max(0.0, min(1.0, float(sim)))
        ctx.similarity_available = True
    except Exception as exc:
        LOGGER.debug("Embedding similarity failed in critic context: %s", exc)
        ctx.semantic_similarity = 0.85  # neutral fallback
        ctx.similarity_available = False

    # 5. Step 13 Scoring & Dimension comparison
    try:
        from app.services.scoring_service import score_prompt

        s_orig = score_prompt(original_prompt)
        s_opt = score_prompt(optimized_prompt)

        ctx.overall_score_original = float(s_orig.overall_score.score)
        ctx.overall_score_optimized = float(s_opt.overall_score.score)
        ctx.overall_score_delta = ctx.overall_score_optimized - ctx.overall_score_original
        ctx.scoring_available = True

        for name, dim in s_orig.dimensions.items():
            ctx.all_dimensions_original[name] = float(dim.score)
        for name, dim in s_opt.dimensions.items():
            ctx.all_dimensions_optimized[name] = float(dim.score)

        def _get_dim(dim_dict: dict[str, float], key: str, default: float = 50.0) -> float:
            return dim_dict.get(key, default)

        ctx.clarity_original = _get_dim(ctx.all_dimensions_original, "clarity")
        ctx.clarity_optimized = _get_dim(ctx.all_dimensions_optimized, "clarity")
        ctx.clarity_delta = ctx.clarity_optimized - ctx.clarity_original

        ctx.specificity_original = _get_dim(ctx.all_dimensions_original, "specificity")
        ctx.specificity_optimized = _get_dim(ctx.all_dimensions_optimized, "specificity")
        ctx.specificity_delta = ctx.specificity_optimized - ctx.specificity_original

        ctx.ambiguity_original = _get_dim(ctx.all_dimensions_original, "ambiguity")
        ctx.ambiguity_optimized = _get_dim(ctx.all_dimensions_optimized, "ambiguity")
        ctx.ambiguity_delta = ctx.ambiguity_optimized - ctx.ambiguity_original

        ctx.completeness_original = _get_dim(ctx.all_dimensions_original, "completeness")
        ctx.completeness_optimized = _get_dim(ctx.all_dimensions_optimized, "completeness")
        ctx.completeness_delta = ctx.completeness_optimized - ctx.completeness_original

        ctx.consistency_original = _get_dim(ctx.all_dimensions_original, "consistency")
        ctx.consistency_optimized = _get_dim(ctx.all_dimensions_optimized, "consistency")
        ctx.consistency_delta = ctx.consistency_optimized - ctx.consistency_original

    except Exception as exc:
        LOGGER.debug("Scoring service failed in critic context: %s", exc)

    return ctx
