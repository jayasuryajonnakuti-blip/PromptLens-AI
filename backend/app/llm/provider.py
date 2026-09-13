"""Abstract provider interface for the PromptLens Local LLM Layer (Step 14)."""
from abc import ABC, abstractmethod
from typing import Any

from app.llm.config import LLMConfig
from app.llm.schemas import LLMHealthStatus


class LLMProvider(ABC):
    """Abstract interface isolating the local LLM runtime from the application."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Generate raw text response from the model."""

    @abstractmethod
    def health(self) -> LLMHealthStatus:
        """Check the current operational state of the local model."""

    @abstractmethod
    def model_metadata(self) -> dict[str, Any]:
        """Return metadata regarding the model name, runtime, and parameters."""
