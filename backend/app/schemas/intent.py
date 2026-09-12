from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


class IntentLabel(StrEnum):
    CODING = "CODING"
    EDUCATION = "EDUCATION"
    RESEARCH = "RESEARCH"
    WRITING = "WRITING"
    SUMMARIZATION = "SUMMARIZATION"
    TRANSLATION = "TRANSLATION"
    BUSINESS = "BUSINESS"
    MARKETING = "MARKETING"
    DATA_ANALYSIS = "DATA_ANALYSIS"
    CREATIVE = "CREATIVE"
    IMAGE_GENERATION = "IMAGE_GENERATION"
    GENERAL = "GENERAL"


class IntentClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: StrictStr = Field(min_length=1, description="Prompt to classify.")

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty or whitespace-only")
        return value


class IntentModelMetadata(BaseModel):
    name: str
    version: str


class IntentClassificationResponse(BaseModel):
    intent: IntentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    model: IntentModelMetadata
    confidence_methodology: str