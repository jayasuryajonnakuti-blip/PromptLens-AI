from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr

OutputFormat = Literal[
    "JSON",
    "Markdown",
    "table",
    "bullets",
    "numbered_list",
    "code",
    "plain_text",
]

SectionName = Literal[
    "role_persona",
    "task_goal",
    "context",
    "constraints",
    "audience",
    "output_format",
]


class PromptPreprocessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: StrictStr = Field(description="The raw prompt to inspect deterministically.")


class TextStatistics(BaseModel):
    character_count: int
    character_count_no_whitespace: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    average_word_length: float
    vocabulary_diversity: float


class StructureStatistics(BaseModel):
    question_count: int
    instruction_count_estimated: int
    bullet_count: int
    heading_count: int
    has_code_block: bool
    has_url: bool
    has_email: bool


class PlaceholderMatch(BaseModel):
    value: str
    kind: Literal["square_bracket", "curly_brace", "angle_bracket"]


class RepeatedPhrase(BaseModel):
    phrase: str
    count: int


class PreprocessingFlags(BaseModel):
    is_empty: bool
    is_near_empty: bool
    is_very_long: bool


class PromptPreprocessingResult(BaseModel):
    text: TextStatistics
    structure: StructureStatistics
    placeholders: list[PlaceholderMatch]
    output_formats: list[OutputFormat]
    sections: list[SectionName]
    repeated_phrases: list[RepeatedPhrase]
    flags: PreprocessingFlags
