"""Signal adapters and score calculators for PromptLens (Step 13).

Provides deterministic scoring functions for:
- Rule-based quality score (reusing Step 8 preprocessing)
- Linguistic NLP-based quality score (reusing Step 9 spaCy analysis)
- Quality ML adapter (reusing Step 12 quality prediction)
- Handling unavailable signals (e.g. LLM)
"""
from dataclasses import dataclass
from typing import Any

from app.schemas.nlp import NlpAnalysisResult
from app.schemas.preprocessing import PromptPreprocessingResult


@dataclass(frozen=True)
class ScoringSignal:
    source: str
    raw_score: float | None
    configured_weight: float
    effective_weight: float
    available: bool
    explanation: str | None = None


def compute_rule_score(prep: PromptPreprocessingResult) -> float:
    """Compute deterministic rule-based score in [0.0, 100.0] from Step 8 results."""
    if prep.flags.is_empty:
        return 0.0

    if prep.flags.is_near_empty:
        return 12.0

    score = 50.0

    # Length & vocabulary bonuses/penalties
    word_count = prep.text.word_count
    if word_count < 6:
        score -= 20.0
    elif word_count < 12:
        score -= 10.0
    elif 15 <= word_count <= 250:
        score += 8.0

    if word_count >= 15 and prep.text.vocabulary_diversity >= 0.60:
        score += 5.0

    # Instructions and goals
    instructions = prep.structure.instruction_count_estimated
    if instructions >= 2:
        score += 15.0
    elif instructions == 1:
        score += 8.0
    else:
        score -= 10.0

    # Questions
    if prep.structure.question_count > 0:
        score += 5.0

    # Output format
    if prep.output_formats:
        score += min(15.0, len(prep.output_formats) * 7.5)

    # Detected structural sections (task_goal, context, constraints, audience, etc.)
    if prep.sections:
        score += min(15.0, len(prep.sections) * 5.0)

    # Placeholders, headings, bullets, code blocks
    if prep.placeholders:
        score += 5.0
    if prep.structure.bullet_count > 0:
        score += 5.0
    if prep.structure.heading_count > 0:
        score += 5.0
    if prep.structure.has_code_block:
        score += 5.0

    # Penalties for excessive repetition
    if len(prep.repeated_phrases) > 4:
        score -= 15.0
    elif len(prep.repeated_phrases) > 1:
        score -= 5.0

    # Oversized rambling without sections
    if prep.flags.is_very_long and len(prep.sections) <= 1:
        score -= 15.0

    return max(0.0, min(100.0, round(score, 2)))


def compute_nlp_score(nlp: NlpAnalysisResult) -> float:
    """Compute deterministic NLP-based score in [0.0, 100.0] from Step 9 results."""
    token_count = nlp.token_count
    if token_count == 0:
        return 0.0

    if token_count < 4:
        return 15.0

    score = 50.0

    # Sentence length characteristics
    avg_sentence_len = nlp.sentence_statistics.average_sentence_length
    if 8.0 <= avg_sentence_len <= 30.0:
        score += 10.0
    elif avg_sentence_len > 45.0:
        score -= 10.0
    elif avg_sentence_len < 4.0:
        score -= 8.0

    # Part of speech balance
    nouns = nlp.pos_counts.nouns
    verbs = nlp.pos_counts.verbs
    adjectives = nlp.pos_counts.adjectives

    if nouns >= 2 and verbs >= 1:
        score += 10.0
    elif verbs == 0:
        score -= 15.0

    if adjectives >= 1:
        score += 5.0

    # Verb density & actionability
    verb_density = nlp.verb_statistics.verb_density
    if 0.10 <= verb_density <= 0.35:
        score += 8.0

    # Syntactic complexity & depth
    dep_depth = nlp.complexity.average_dependency_depth
    if 2.5 <= dep_depth <= 6.5:
        score += 8.0
    elif dep_depth > 8.0:
        score -= 5.0

    # Subordinate clauses (contextual qualifications)
    if nlp.complexity.subordinate_clause_count_estimated > 0:
        score += 5.0

    # Instructions estimated in sentences
    if nlp.complexity.instruction_count_estimated >= 1:
        score += 6.0

    # Technical terms
    if nlp.technical_terms.technical_term_count_estimated >= 1:
        score += 5.0

    # Stopword ratio balance
    if 0.20 <= nlp.stopword_ratio <= 0.60:
        score += 5.0
    elif nlp.stopword_ratio > 0.75:
        score -= 8.0

    return max(0.0, min(100.0, round(score, 2)))
