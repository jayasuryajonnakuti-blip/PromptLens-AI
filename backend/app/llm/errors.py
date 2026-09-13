"""Typed error hierarchy for the PromptLens Local LLM Layer (Step 14)."""


class LLMError(Exception):
    """Base exception for all PromptLens LLM errors."""


class LLMDisabledError(LLMError):
    """Raised when an LLM operation is attempted while LLM_ENABLED is False."""


class ModelNotDownloadedError(LLMError):
    """Raised when the configured model weights are not present locally."""


class ModelLoadingError(LLMError):
    """Raised when loading model weights into memory fails."""


class ModelRuntimeError(LLMError):
    """Raised when an error occurs during inference execution."""


class InvalidGenerationError(LLMError):
    """Raised when the model returns empty, truncated, or nonsensical output."""


class InvalidJsonError(LLMError):
    """Raised when the model output cannot be parsed as valid JSON."""


class SchemaValidationError(LLMError):
    """Raised when parsed JSON does not conform to the PromptAnalysisOutput schema."""


class RetryExhaustedError(LLMError):
    """Raised when structured generation retries are exhausted without valid output."""


class ContextOverflowError(LLMError):
    """Raised when prompt and context exceed the maximum allowed token limit."""


class LLMTimeoutError(LLMError):
    """Raised when model inference exceeds the configured timeout threshold."""


class OutOfMemoryError(LLMError):
    """Raised when local inference exhausts available physical memory."""
