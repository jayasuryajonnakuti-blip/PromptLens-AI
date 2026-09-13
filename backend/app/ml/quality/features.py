"""
Feature extraction for the PromptLens quality ML model (Step 12).

Features are derived exclusively from the existing Step 8 (preprocessing) and
Step 9 (NLP) engines. No engine logic is duplicated here.

The dimension scores stored in the quality dataset CSV (clarity, specificity,
etc.) are NEVER used as features here. They are analysis-only annotations that
document the heuristic scoring rationale. Using them as features would create
a direct data leakage path from the annotation process to the model target.

Feature list (~30 structured numeric features):
    Preprocessing features (Step 8):
        character_count, word_count, sentence_count, paragraph_count,
        average_word_length, vocabulary_diversity, question_count,
        instruction_count, bullet_count, heading_count, has_code_block,
        has_url, placeholder_count, output_format_count, section_count,
        repeated_phrase_count, is_near_empty

    NLP features (Step 9):
        unique_token_count, noun_count, verb_count, adjective_count,
        noun_phrase_count, verb_density, stopword_ratio,
        avg_dependency_depth, subordinate_clause_count,
        technical_term_count, named_entity_count, avg_sentence_length

    Derived:
        question_to_instruction_ratio
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.engines.nlp import analyze_prompt
from app.engines.preprocessing import preprocess_prompt

if TYPE_CHECKING:
    pass

FEATURE_NAMES: tuple[str, ...] = (
    # --- Preprocessing features ---
    "character_count",
    "word_count",
    "sentence_count",
    "paragraph_count",
    "average_word_length",
    "vocabulary_diversity",
    "question_count",
    "instruction_count",
    "bullet_count",
    "heading_count",
    "has_code_block",
    "has_url",
    "placeholder_count",
    "output_format_count",
    "section_count",
    "repeated_phrase_count",
    "is_near_empty",
    # --- NLP features ---
    "unique_token_count",
    "noun_count",
    "verb_count",
    "adjective_count",
    "noun_phrase_count",
    "verb_density",
    "stopword_ratio",
    "avg_dependency_depth",
    "subordinate_clause_count",
    "technical_term_count",
    "named_entity_count",
    "avg_sentence_length",
    # --- Derived ---
    "question_to_instruction_ratio",
)


def extract_features(prompt: str) -> list[float]:
    """Extract the full quality feature vector from a raw prompt string.

    Calls the Step 8 preprocessing engine and Step 9 NLP engine.  Returns
    a list of floats in the order defined by FEATURE_NAMES.

    An empty or whitespace-only prompt returns a zero-vector of the correct
    length so that edge cases are handled gracefully at inference time.

    Args:
        prompt: The raw prompt string (may be empty).

    Returns:
        A list of float values of length len(FEATURE_NAMES).
    """
    if not prompt or not prompt.strip():
        return [0.0] * len(FEATURE_NAMES)

    prep = preprocess_prompt(prompt)
    nlp = analyze_prompt(prompt)

    instruction_count = prep.structure.instruction_count_estimated
    question_count = prep.structure.question_count

    vector: list[float] = [
        # --- Preprocessing ---
        float(prep.text.character_count),
        float(prep.text.word_count),
        float(prep.text.sentence_count),
        float(prep.text.paragraph_count),
        float(prep.text.average_word_length),
        float(prep.text.vocabulary_diversity),
        float(question_count),
        float(instruction_count),
        float(prep.structure.bullet_count),
        float(prep.structure.heading_count),
        float(int(prep.structure.has_code_block)),
        float(int(prep.structure.has_url)),
        float(len(prep.placeholders)),
        float(len(prep.output_formats)),
        float(len(prep.sections)),
        float(len(prep.repeated_phrases)),
        float(int(prep.flags.is_near_empty)),
        # --- NLP ---
        float(nlp.unique_token_count),
        float(nlp.pos_counts.nouns),
        float(nlp.pos_counts.verbs),
        float(nlp.pos_counts.adjectives),
        float(nlp.noun_phrase_count),
        float(nlp.verb_statistics.verb_density),
        float(nlp.stopword_ratio),
        float(nlp.complexity.average_dependency_depth),
        float(nlp.complexity.subordinate_clause_count_estimated),
        float(nlp.technical_terms.technical_term_count_estimated),
        float(len(nlp.named_entities)),
        float(nlp.sentence_statistics.average_sentence_length),
        # --- Derived ---
        float(question_count) / max(1.0, float(instruction_count)),
    ]

    assert len(vector) == len(FEATURE_NAMES), (
        f"Feature vector length mismatch: {len(vector)} != {len(FEATURE_NAMES)}"
    )
    return vector


def extract_features_batch(prompts: list[str]) -> list[list[float]]:
    """Extract feature vectors for a list of prompts.

    Args:
        prompts: List of raw prompt strings.

    Returns:
        List of feature vectors, one per prompt.
    """
    return [extract_features(p) for p in prompts]
