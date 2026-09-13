"""Configuration for the PromptLens Local LLM Layer (Step 14).

Reads from environment variables with safe, resource-friendly defaults.
Default is LLM_ENABLED=false to protect laptop resources.
"""
import os
from dataclasses import dataclass


def _get_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "on")


def _get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError as e:
        raise ValueError(f"Invalid integer for {key}: {val}") from e


def _get_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val.strip())
    except ValueError as e:
        raise ValueError(f"Invalid float for {key}: {val}") from e


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool = False
    provider: str = "transformers"
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    context_length: int = 2048
    max_tokens: int = 768
    temperature: float = 0.0
    timeout: float = 60.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError(f"context_length must be positive, got {self.context_length}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.max_tokens > self.context_length:
            raise ValueError(
                f"max_tokens ({self.max_tokens}) cannot exceed context_length ({self.context_length})"
            )
        if self.temperature < 0.0:
            raise ValueError(f"temperature cannot be negative, got {self.temperature}")
        if self.timeout <= 0.0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries cannot be negative, got {self.max_retries}")


def load_llm_config() -> LLMConfig:
    """Load and validate LLM configuration from environment variables."""
    return LLMConfig(
        enabled=_get_bool("LLM_ENABLED", False),
        provider=os.getenv("LLM_PROVIDER", "transformers").strip().lower(),
        model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct").strip(),
        context_length=_get_int("LLM_CONTEXT_LENGTH", 2048),
        max_tokens=_get_int("LLM_MAX_TOKENS", 768),
        temperature=_get_float("LLM_TEMPERATURE", 0.0),
        timeout=_get_float("LLM_TIMEOUT", 60.0),
        max_retries=_get_int("LLM_MAX_RETRIES", 2),
    )
