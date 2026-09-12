import re
from collections import Counter

from app.schemas.preprocessing import (
    OutputFormat,
    PlaceholderMatch,
    PreprocessingFlags,
    PromptPreprocessingResult,
    RepeatedPhrase,
    SectionName,
    StructureStatistics,
    TextStatistics,
)

_WORD_PATTERN = re.compile(r"\b[\w]+(?:['-][\w]+)*\b", re.UNICODE)
_SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
_BULLET_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_NUMBERED_LINE_PATTERN = re.compile(r"^\s*\d+[.)]\s+")
_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+\S+")
_URL_PATTERN = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PLACEHOLDER_PATTERNS = (
    ("square_bracket", re.compile(r"\[[A-Z][A-Z0-9_ -]*\]")),
    ("curly_brace", re.compile(r"\{[A-Za-z_][\w.-]*\}")),
    ("angle_bracket", re.compile(r"<[A-Za-z_][\w.-]*>")),
)
_IMPERATIVE_STARTS = {
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
    "improve",
    "include",
    "list",
    "provide",
    "rewrite",
    "summarize",
    "translate",
    "use",
    "write",
}
_SECTION_TERMS: tuple[tuple[SectionName, tuple[str, ...]], ...] = (
    ("role_persona", ("role", "persona", "act as", "you are")),
    ("task_goal", ("task", "goal", "objective", "your job", "purpose", "explain", "write", "create")),
    ("context", ("context", "background", "given that", "assume that")),
    ("constraints", ("constraint", "must", "do not", "avoid", "limit", "only")),
    ("audience", ("audience", "reader", "user", "for beginners", "for experts", "student", "high school")),
    ("output_format", ("output", "format", "return as", "respond with", "deliverable")),
)

_VERY_LONG_WORD_THRESHOLD = 2_000
_NEAR_EMPTY_WORD_THRESHOLD = 3
_MAX_REPEATED_PHRASES = 20


def preprocess_prompt(prompt: str) -> PromptPreprocessingResult:
    words = _WORD_PATTERN.findall(prompt)
    normalized_words = [word.lower() for word in words]
    non_empty_lines = [line for line in prompt.splitlines() if line.strip()]
    sentences = _SENTENCE_PATTERN.findall(prompt)
    sentence_count = len([sentence for sentence in sentences if sentence.strip()])
    if prompt.strip() and sentence_count == 0:
        sentence_count = 1

    word_count = len(words)
    character_count = len(prompt)
    average_word_length = round(
        sum(len(word) for word in words) / word_count, 2
    ) if word_count else 0.0
    vocabulary_diversity = round(
        len(set(normalized_words)) / word_count, 3
    ) if word_count else 0.0

    bullet_count = sum(1 for line in non_empty_lines if _BULLET_PATTERN.match(line))
    heading_count = sum(1 for line in non_empty_lines if _HEADING_PATTERN.match(line))
    instruction_count = _estimate_instruction_count(sentences, non_empty_lines)
    placeholders = _extract_placeholders(prompt)
    output_formats = _detect_output_formats(prompt, bullet_count)
    sections = _detect_sections(prompt)

    return PromptPreprocessingResult(
        text=TextStatistics(
            character_count=character_count,
            character_count_no_whitespace=len(re.sub(r"\s", "", prompt)),
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=_count_paragraphs(prompt),
            average_word_length=average_word_length,
            vocabulary_diversity=vocabulary_diversity,
        ),
        structure=StructureStatistics(
            question_count=prompt.count("?"),
            instruction_count_estimated=instruction_count,
            bullet_count=bullet_count,
            heading_count=heading_count,
            has_code_block="```" in prompt or "~~~" in prompt,
            has_url=bool(_URL_PATTERN.search(prompt)),
            has_email=bool(_EMAIL_PATTERN.search(prompt)),
        ),
        placeholders=placeholders,
        output_formats=output_formats,
        sections=sections,
        repeated_phrases=_find_repeated_phrases(normalized_words),
        flags=PreprocessingFlags(
            is_empty=not prompt.strip(),
            is_near_empty=bool(prompt.strip()) and word_count < _NEAR_EMPTY_WORD_THRESHOLD,
            is_very_long=word_count >= _VERY_LONG_WORD_THRESHOLD,
        ),
    )


def _count_paragraphs(prompt: str) -> int:
    return len([paragraph for paragraph in re.split(r"\n\s*\n", prompt.strip()) if paragraph.strip()])


def _estimate_instruction_count(sentences: list[str], lines: list[str]) -> int:
    candidates = list(dict.fromkeys(
        sentences + [line for line in lines if _BULLET_PATTERN.match(line)]
    ))
    count = 0
    for candidate in candidates:
        first_word_match = _WORD_PATTERN.search(candidate)
        if first_word_match and first_word_match.group().lower() in _IMPERATIVE_STARTS:
            count += 1
    return count


def _extract_placeholders(prompt: str) -> list[PlaceholderMatch]:
    matches: list[PlaceholderMatch] = []
    for kind, pattern in _PLACEHOLDER_PATTERNS:
        for match in pattern.finditer(prompt):
            matches.append(PlaceholderMatch(value=match.group(), kind=kind))
    return matches


def _detect_output_formats(prompt: str, bullet_count: int) -> list[OutputFormat]:
    lowered = prompt.lower()
    lines = prompt.splitlines()
    formats: list[OutputFormat] = []
    indicators: tuple[tuple[OutputFormat, tuple[str, ...]], ...] = (
        ("JSON", ("json", "json format")),
        ("Markdown", ("markdown", "md format")),
        ("table", ("table", "tabular")),
        ("bullets", ("bullet", "bullets", "bullet points")),
        ("numbered_list", ("numbered list", "numbered steps", "number the")),
        ("code", ("code", "program", "script", "code block")),
        ("plain_text", ("plain text", "plain-text", "prose only")),
    )
    for output_format, terms in indicators:
        if any(term in lowered for term in terms) or (
            output_format == "code" and ("```" in prompt or "~~~" in prompt)
        ):
            formats.append(output_format)
    if bullet_count and "bullets" not in formats:
        formats.append("bullets")
    if any(_NUMBERED_LINE_PATTERN.match(line) for line in lines):
        formats.append("numbered_list")
    return formats


def _detect_sections(prompt: str) -> list[SectionName]:
    lowered = prompt.lower()
    return [
        section
        for section, terms in _SECTION_TERMS
        if any(term in lowered for term in terms)
    ]


def _find_repeated_phrases(words: list[str]) -> list[RepeatedPhrase]:
    if len(words) < 2:
        return []
    phrase_counts: Counter[str] = Counter()
    for phrase_length in (2, 3, 4):
        for index in range(len(words) - phrase_length + 1):
            phrase = " ".join(words[index:index + phrase_length])
            phrase_counts[phrase] += 1
    repeated = [
        RepeatedPhrase(phrase=phrase, count=count)
        for phrase, count in phrase_counts.items()
        if count > 1
    ]
    repeated.sort(key=lambda item: (-item.count, -len(item.phrase), item.phrase))
    return repeated[:_MAX_REPEATED_PHRASES]
