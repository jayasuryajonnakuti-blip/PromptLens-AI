from collections.abc import Callable
import re


_PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def chunk_text(
    text: str,
    tokenizer: Callable[..., list[int]],
    decoder: Callable[[list[int]], str],
    max_tokens: int,
) -> list[str]:
    """Split text on paragraphs/sentences, then tokenizer windows when necessary."""
    if not text.strip():
        return []
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    units = [
        sentence.strip()
        for paragraph in _PARAGRAPH_PATTERN.split(text)
        for sentence in _SENTENCE_PATTERN.split(paragraph)
        if sentence.strip()
    ]
    chunks: list[str] = []
    current_units: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = tokenizer(unit, add_special_tokens=False)
        if len(unit_tokens) > max_tokens:
            if current_units:
                chunks.append(" ".join(current_units))
                current_units = []
                current_tokens = 0
            chunks.extend(
                decoder(unit_tokens[index:index + max_tokens]).strip()
                for index in range(0, len(unit_tokens), max_tokens)
            )
            continue
        if current_units and current_tokens + len(unit_tokens) > max_tokens:
            chunks.append(" ".join(current_units))
            current_units = []
            current_tokens = 0
        current_units.append(unit)
        current_tokens += len(unit_tokens)
    if current_units:
        chunks.append(" ".join(current_units))
    return [chunk for chunk in chunks if chunk]