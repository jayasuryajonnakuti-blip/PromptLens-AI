"""ValidatorContext: builds comparative signals between original and optimized prompts.

Extracts:
- Step 8 Preprocessing (word counts, sentence counts, format requirements)
- Step 10 Intent classification (intent labels, confidence, match)
- Step 11 Semantic embeddings similarity
- Step 13 Scoring & 12 Dimensions
- Integrates optional Step 17 Critic findings
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.validator.schemas import CriticResultInput, PromptMetadata

LOGGER = logging.getLogger(__name__)


@dataclass
class ValidatorContext:
    original_prompt: str = ""
    optimized_prompt: str = ""

    # --- Lexical Metrics ---
    original_length: int = 0
    optimized_length: int = 0
    original_word_count: int = 0
    optimized_word_count: int = 0
    original_sentence_count: int = 0
    optimized_sentence_count: int = 0
    expansion_ratio: float = 1.0

    # --- Structural / Format Features ---
    original_formats: list[str] = field(default_factory=list)
    optimized_formats: list[str] = field(default_factory=list)
    original_constraints: list[str] = field(default_factory=list)
    optimized_constraints: list[str] = field(default_factory=list)

    # --- Intent Classification ---
    original_intent: str = "unknown"
    original_intent_confidence: float = 0.0
    optimized_intent: str = "unknown"
    optimized_intent_confidence: float = 0.0
    intent_match: bool = True

    # --- Semantic Similarity ---
    semantic_similarity: float = 1.0
    similarity_available: bool = False

    # --- Step 13 Scoring ---
    original_score: float = 50.0
    optimized_score: float = 50.0
    score_delta: float = 0.0
    scoring_available: bool = False

    # Key dimension deltas
    clarity_delta: float = 0.0
    specificity_delta: float = 0.0
    ambiguity_delta: float = 0.0
    completeness_delta: float = 0.0
    consistency_delta: float = 0.0

    # --- Optional Step 17 Critic Findings ---
    critic_provided: bool = False
    critic_decision: str | None = None
    critic_score: float | None = None
    critic_lost_requirements: list[str] = field(default_factory=list)
    critic_introduced_requirements: list[str] = field(default_factory=list)
    critic_issues_count: int = 0

    def to_original_metadata(self) -> PromptMetadata:
        return PromptMetadata(
            prompt_length=self.original_length,
            word_count=self.original_word_count,
            sentence_count=self.original_sentence_count,
            detected_intent=self.original_intent,
            intent_confidence=self.original_intent_confidence,
            output_formats=self.original_formats,
            quality_score=self.original_score,
        )

    def to_optimized_metadata(self) -> PromptMetadata:
        return PromptMetadata(
            prompt_length=self.optimized_length,
            word_count=self.optimized_word_count,
            sentence_count=self.optimized_sentence_count,
            detected_intent=self.optimized_intent,
            intent_confidence=self.optimized_intent_confidence,
            output_formats=self.optimized_formats,
            quality_score=self.optimized_score,
        )

    def to_compact_dict(self) -> dict[str, Any]:
        """Compact dictionary structure for LLM validator context."""
        return {
            "lexical": {
                "original_words": self.original_word_count,
                "optimized_words": self.optimized_word_count,
                "expansion_ratio": round(self.expansion_ratio, 2),
            },
            "intent": {
                "original_intent": self.original_intent,
                "optimized_intent": self.optimized_intent,
                "intent_match": self.intent_match,
                "confidence": round(self.original_intent_confidence, 3),
            },
            "semantic_similarity": round(self.semantic_similarity, 4),
            "scoring": {
                "original_score": round(self.original_score, 1),
                "optimized_score": round(self.optimized_score, 1),
                "score_delta": round(self.score_delta, 1),
            },
            "formats": {
                "original": self.original_formats,
                "optimized": self.optimized_formats,
            },
            "critic": {
                "provided": self.critic_provided,
                "decision": self.critic_decision,
                "score": self.critic_score,
                "reported_lost_requirements": self.critic_lost_requirements,
            },
        }


def build_validator_context(
    original_prompt: str,
    optimized_prompt: str,
    critic_result: CriticResultInput | None = None,
) -> ValidatorContext:
    """Build a populated ValidatorContext comparing original and optimized prompts."""
    ctx = ValidatorContext(
        original_prompt=original_prompt,
        optimized_prompt=optimized_prompt,
        original_length=len(original_prompt),
        optimized_length=len(optimized_prompt),
    )

    # 1. Preprocessing & Lexical analysis
    try:
        from app.services.preprocessing_service import preprocess_prompt

        prep_orig = preprocess_prompt(original_prompt)
        prep_opt = preprocess_prompt(optimized_prompt)

        ctx.original_word_count = getattr(prep_orig.text, "word_count", len(original_prompt.split()))
        ctx.optimized_word_count = getattr(prep_opt.text, "word_count", len(optimized_prompt.split()))
        ctx.original_sentence_count = getattr(prep_orig.text, "sentence_count", 1)
        ctx.optimized_sentence_count = getattr(prep_opt.text, "sentence_count", 1)

        ctx.original_formats = list(prep_orig.output_formats or [])
        ctx.optimized_formats = list(prep_opt.output_formats or [])
    except Exception as exc:
        LOGGER.debug("Preprocessing service failed in validator context: %s", exc)
        ctx.original_word_count = len(original_prompt.split())
        ctx.optimized_word_count = len(optimized_prompt.split())
        ctx.original_sentence_count = 1
        ctx.optimized_sentence_count = 1

    ctx.expansion_ratio = ctx.optimized_word_count / max(1, ctx.original_word_count)

    # 2. Intent Classification
    try:
        from app.services.intent_service import classify_prompt

        i_orig = classify_prompt(original_prompt)
        i_opt = classify_prompt(optimized_prompt)

        ctx.original_intent = i_orig.intent.value
        ctx.original_intent_confidence = float(i_orig.confidence)
        ctx.optimized_intent = i_opt.intent.value
        ctx.optimized_intent_confidence = float(i_opt.confidence)
        ctx.intent_match = (ctx.original_intent == ctx.optimized_intent)
    except Exception as exc:
        LOGGER.debug("Intent classification failed in validator context: %s", exc)

    # 3. Semantic Similarity
    try:
        from app.services.embedding_service import get_embedding_service

        emb_service = get_embedding_service()
        emb_orig = emb_service.generate_embedding(original_prompt)
        emb_opt = emb_service.generate_embedding(optimized_prompt)
        sim = emb_service.compute_similarity(emb_orig, emb_opt)
        ctx.semantic_similarity = max(0.0, min(1.0, float(sim)))
        ctx.similarity_available = True
    except Exception as exc:
        LOGGER.debug("Embedding service failed in validator context: %s", exc)
        ctx.semantic_similarity = 0.85
        ctx.similarity_available = False

    # 4. Step 13 Scoring Comparison
    try:
        from app.services.scoring_service import score_prompt

        s_orig = score_prompt(original_prompt)
        s_opt = score_prompt(optimized_prompt)

        ctx.original_score = float(s_orig.overall_score.score)
        ctx.optimized_score = float(s_opt.overall_score.score)
        ctx.score_delta = ctx.optimized_score - ctx.original_score
        ctx.scoring_available = True

        def _get_score(score_obj: Any, key: str) -> float:
            dim = score_obj.dimensions.get(key)
            return float(dim.score) if dim else 50.0

        ctx.clarity_delta = _get_score(s_opt, "clarity") - _get_score(s_orig, "clarity")
        ctx.specificity_delta = _get_score(s_opt, "specificity") - _get_score(s_orig, "specificity")
        ctx.ambiguity_delta = _get_score(s_opt, "ambiguity") - _get_score(s_orig, "ambiguity")
        ctx.completeness_delta = _get_score(s_opt, "completeness") - _get_score(s_orig, "completeness")
        ctx.consistency_delta = _get_score(s_opt, "consistency") - _get_score(s_orig, "consistency")
    except Exception as exc:
        LOGGER.debug("Scoring service failed in validator context: %s", exc)

    # 5. Critic result integration
    if critic_result:
        ctx.critic_provided = True
        ctx.critic_decision = critic_result.decision
        ctx.critic_score = critic_result.overall_critique_score
        ctx.critic_lost_requirements = list(critic_result.lost_requirements)
        ctx.critic_introduced_requirements = list(critic_result.introduced_requirements)
        ctx.critic_issues_count = len(critic_result.issues)

    return ctx
