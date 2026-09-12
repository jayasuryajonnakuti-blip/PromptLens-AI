from pydantic import BaseModel, ConfigDict, Field, StrictStr


class NlpAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: StrictStr = Field(description="The raw prompt to analyze with spaCy.")


class PosCounts(BaseModel):
    nouns: int
    verbs: int
    adjectives: int
    adverbs: int
    pronouns: int
    conjunctions: int
    prepositions_or_adpositions: int
    determiners: int


class NamedEntity(BaseModel):
    text: str
    label: str


class SentenceStatistics(BaseModel):
    sentence_count: int
    average_sentence_length: float
    longest_sentence_length: int


class ComplexityIndicators(BaseModel):
    average_dependency_depth: float
    subordinate_clause_count_estimated: int
    question_count: int
    instruction_count_estimated: int


class VerbStatistics(BaseModel):
    verb_count: int
    verb_density: float


class TechnicalTermIndicators(BaseModel):
    technical_term_count_estimated: int
    technical_terms: list[str]


class PunctuationStatistics(BaseModel):
    total_count: int
    comma_count: int
    period_count: int
    question_mark_count: int
    exclamation_mark_count: int
    colon_count: int
    semicolon_count: int


class NlpAnalysisResult(BaseModel):
    model_name: str
    token_count: int
    unique_token_count: int
    lemma_count: int
    vocabulary_diversity: float
    pos_counts: PosCounts
    named_entities: list[NamedEntity]
    sentence_statistics: SentenceStatistics
    complexity: ComplexityIndicators
    noun_phrase_count: int
    verb_statistics: VerbStatistics
    technical_terms: TechnicalTermIndicators
    stopword_ratio: float
    punctuation: PunctuationStatistics
    uppercase_token_ratio: float
    number_token_count: int
    url_tokens: list[str]
    email_tokens: list[str]