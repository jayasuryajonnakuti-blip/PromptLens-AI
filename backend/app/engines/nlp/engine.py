import re
from collections import Counter
from functools import lru_cache

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from app.schemas.nlp import (
    ComplexityIndicators,
    NamedEntity,
    NlpAnalysisResult,
    PosCounts,
    PunctuationStatistics,
    SentenceStatistics,
    TechnicalTermIndicators,
    VerbStatistics,
)

NLP_MODEL_NAME = "en_core_web_sm"
_WORDLIKE_TOKEN = re.compile(r"[\w-]+", re.UNICODE)
_URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s]+", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_IMPERATIVE_LEMMAS = {
    "add",
    "analyze",
    "compare",
    "create",
    "describe",
    "design",
    "determine",
    "draft",
    "explain",
    "generate",
    "identify",
    "include",
    "list",
    "provide",
    "rewrite",
    "summarize",
    "translate",
    "use",
    "write",
}
_SUBORDINATE_DEPENDENCIES = {"advcl", "ccomp", "xcomp", "acl", "relcl"}
_SUBORDINATE_MARKERS = {
    "after",
    "although",
    "because",
    "before",
    "if",
    "since",
    "that",
    "though",
    "unless",
    "when",
    "while",
}


class NlpModelUnavailableError(RuntimeError):
    """Raised when the configured spaCy model is not installed or cannot load."""


@lru_cache(maxsize=1)
def _load_model() -> Language:
    try:
        return spacy.load(NLP_MODEL_NAME)
    except (OSError, ImportError) as error:
        raise NlpModelUnavailableError(
            f"The spaCy model '{NLP_MODEL_NAME}' is unavailable. "
            "Install it with: python -m spacy download en_core_web_sm"
        ) from error


def analyze_prompt(prompt: str) -> NlpAnalysisResult:
    nlp = _load_model()
    if len(prompt) >= nlp.max_length:
        nlp.max_length = len(prompt) + 1
    document = nlp(prompt)
    tokens = [token for token in document if not token.is_space]
    wordlike_tokens = [token for token in tokens if _is_wordlike(token)]
    normalized_tokens = [token.lower_ for token in wordlike_tokens]
    token_count = len(tokens)
    word_count = len(wordlike_tokens)

    return NlpAnalysisResult(
        model_name=NLP_MODEL_NAME,
        token_count=token_count,
        unique_token_count=len(set(normalized_tokens)),
        lemma_count=sum(1 for token in wordlike_tokens if token.lemma_),
        vocabulary_diversity=_ratio(len(set(normalized_tokens)), word_count),
        pos_counts=_count_parts_of_speech(wordlike_tokens),
        named_entities=[
            NamedEntity(text=entity.text, label=entity.label_)
            for entity in document.ents
        ],
        sentence_statistics=_sentence_statistics(document),
        complexity=_complexity_indicators(document),
        noun_phrase_count=len(list(document.noun_chunks)),
        verb_statistics=VerbStatistics(
            verb_count=sum(1 for token in wordlike_tokens if token.pos_ in {"VERB", "AUX"}),
            verb_density=_ratio(
                sum(1 for token in wordlike_tokens if token.pos_ in {"VERB", "AUX"}),
                word_count,
            ),
        ),
        technical_terms=_technical_terms(wordlike_tokens),
        stopword_ratio=_ratio(
            sum(1 for token in wordlike_tokens if token.is_stop),
            word_count,
        ),
        punctuation=_punctuation_statistics(document),
        uppercase_token_ratio=_ratio(
            sum(1 for token in wordlike_tokens if token.text.isupper()),
            word_count,
        ),
        number_token_count=sum(1 for token in tokens if token.like_num),
        url_tokens=_URL_PATTERN.findall(prompt),
        email_tokens=_EMAIL_PATTERN.findall(prompt),
    )


def _is_wordlike(token: Token) -> bool:
    return bool(_WORDLIKE_TOKEN.fullmatch(token.text)) and not token.is_punct


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 3) if denominator else 0.0


def _count_parts_of_speech(tokens: list[Token]) -> PosCounts:
    counts = Counter(token.pos_ for token in tokens)
    return PosCounts(
        nouns=counts["NOUN"] + counts["PROPN"],
        verbs=counts["VERB"] + counts["AUX"],
        adjectives=counts["ADJ"],
        adverbs=counts["ADV"],
        pronouns=counts["PRON"],
        conjunctions=counts["CCONJ"] + counts["SCONJ"],
        prepositions_or_adpositions=counts["ADP"],
        determiners=counts["DET"],
    )


def _sentence_statistics(document: Doc) -> SentenceStatistics:
    sentences = list(document.sents)
    lengths = [sum(1 for token in sentence if _is_wordlike(token)) for sentence in sentences]
    return SentenceStatistics(
        sentence_count=len(sentences),
        average_sentence_length=round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        longest_sentence_length=max(lengths, default=0),
    )


def _complexity_indicators(document: Doc) -> ComplexityIndicators:
    sentences = list(document.sents)
    depths = [_sentence_dependency_depth(sentence) for sentence in sentences]
    return ComplexityIndicators(
        average_dependency_depth=round(sum(depths) / len(depths), 2) if depths else 0.0,
        subordinate_clause_count_estimated=sum(
            1
            for token in document
            if token.dep_ in _SUBORDINATE_DEPENDENCIES
            or token.lower_ in _SUBORDINATE_MARKERS
        ),
        question_count=sum(1 for sentence in sentences if sentence.text.rstrip().endswith("?")),
        instruction_count_estimated=sum(
            1 for sentence in sentences if _looks_like_instruction(sentence)
        ),
    )


def _sentence_dependency_depth(sentence: Span) -> int:
    sentence_token_indices = {token.i for token in sentence}
    depth_cache: dict[int, int] = {}
    for token in sentence:
        path: list[int] = []
        current = token
        visited_indices: set[int] = set()
        while (
            current.i not in depth_cache
            and current.head is not current
            and current.head.i in sentence_token_indices
            and current.i not in visited_indices
        ):
            visited_indices.add(current.i)
            path.append(current.i)
            current = current.head
        depth = depth_cache.get(current.i, 0)
        for token_index in reversed(path):
            depth += 1
            depth_cache[token_index] = depth
    depths = [depth_cache.get(token.i, 0) for token in sentence]
    return round(sum(depths) / len(depths), 2) if depths else 0.0


def _looks_like_instruction(sentence: Span) -> bool:
    first_word = next((token for token in sentence if _is_wordlike(token)), None)
    return bool(
        first_word
        and (
            first_word.lemma_.lower() in _IMPERATIVE_LEMMAS
            or first_word.lower_ == "please"
            or first_word.tag_ == "VB"
        )
    )


def _technical_terms(tokens: list[Token]) -> TechnicalTermIndicators:
    terms = [
        token.text
        for token in tokens
        if token.pos_ in {"NOUN", "PROPN"}
        and token.is_alpha
        and len(token.text) >= 8
    ]
    unique_terms = list(dict.fromkeys(terms))
    return TechnicalTermIndicators(
        technical_term_count_estimated=len(terms),
        technical_terms=unique_terms[:50],
    )


def _punctuation_statistics(document: Doc) -> PunctuationStatistics:
    punctuation = Counter(token.text for token in document if token.is_punct)
    return PunctuationStatistics(
        total_count=sum(punctuation.values()),
        comma_count=punctuation[","],
        period_count=punctuation["."],
        question_mark_count=punctuation["?"],
        exclamation_mark_count=punctuation["!"],
        colon_count=punctuation[":"],
        semicolon_count=punctuation[";"],
    )