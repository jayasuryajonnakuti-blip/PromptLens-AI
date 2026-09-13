"""Pydantic schemas for the PromptLens Local LLM Layer (Step 14)."""
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


class LLMHealthStatus(StrEnum):
    LLM_DISABLED = "LLM_DISABLED"
    MODEL_NOT_DOWNLOADED = "MODEL_NOT_DOWNLOADED"
    MODEL_LOADING = "MODEL_LOADING"
    MODEL_AVAILABLE = "MODEL_AVAILABLE"
    MODEL_ERROR = "MODEL_ERROR"


class InstructionQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(
        ge=0.0,
        le=100.0,
        description="Instruction quality score between 0 and 100.",
    )
    reason: str = Field(description="Explanation for the assigned instruction quality score.")


class PromptAnalysisOutput(BaseModel):
    """Strict structured analysis produced by the local LLM."""
    model_config = ConfigDict(extra="forbid")

    interpreted_goal: str = Field(
        description="The underlying objective or task the user wants to accomplish."
    )
    context: str = Field(
        description="Summary of background context, scenario assumptions, or domain framing."
    )
    ambiguities: list[str] = Field(
        default_factory=list,
        description="Ambiguous phrases, vague requirements, or multiple possible interpretations.",
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="Unstated inputs, missing parameter bounds, or omitted constraints.",
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Conflicting instructions, incompatible output formats, or logical mismatches.",
    )
    instruction_quality: InstructionQuality = Field(
        description="Assessment of instruction clarity and directness."
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Identified strong prompt engineering elements in the prompt.",
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Identified deficits or weaknesses in the prompt specification.",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable, targeted recommendations to improve the prompt (not rewritten prompts).",
    )


class LLMModelMetadata(BaseModel):
    name: str
    version: str = "0.1.0"


class LLMAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: StrictStr = Field(description="The prompt to analyze with the local LLM.")

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty or whitespace-only")
        return value


class LLMAnalysisResponse(BaseModel):
    analysis: PromptAnalysisOutput
    model: LLMModelMetadata
    provider: str
    latency_ms: float
    structured_output_valid: bool
