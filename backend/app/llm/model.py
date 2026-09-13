"""Concrete provider implementations for PromptLens (Step 14).

Provides:
- TransformersLocalProvider: Real CPU-compatible local LLM inference using transformers
- MockLLMProvider: Deterministic test double for testing without weights or GPU
"""
import json
import logging
import threading
from typing import Any

from app.llm.config import LLMConfig
from app.llm.errors import (
    LLMDisabledError,
    ModelLoadingError,
    ModelNotDownloadedError,
    ModelRuntimeError,
)
from app.llm.provider import LLMProvider
from app.llm.schemas import LLMHealthStatus

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 1. Deterministic Mock Provider (for tests and CI)
# -----------------------------------------------------------------------------
class MockLLMProvider(LLMProvider):
    """Deterministic mock provider for automated tests and CI environments."""

    def __init__(
        self,
        mode: str = "valid",
        health_status: LLMHealthStatus = LLMHealthStatus.MODEL_AVAILABLE,
        model_name: str = "promptlens-mock-llm",
    ) -> None:
        self.mode = mode
        self._health_status = health_status
        self.model_name = model_name
        self.call_count = 0

    def health(self) -> LLMHealthStatus:
        return self._health_status

    def model_metadata(self) -> dict[str, Any]:
        return {
            "name": self.model_name,
            "provider": "mock",
            "version": "0.1.0",
            "mode": self.mode,
        }

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        self.call_count += 1

        if self.mode == "error":
            raise ModelRuntimeError("Simulated mock runtime inference failure.")

        if self.mode == "malformed_json":
            return "{ unquoted_key: 'malformed JSON text that cannot parse"

        if self.mode == "schema_invalid":
            # Missing interpreted_goal, score > 100
            return json.dumps({
                "context": "General prompt context",
                "instruction_quality": {"score": 250.0, "reason": "Out of bounds score"},
            })

        if self.mode == "fail_once_then_succeed":
            if self.call_count == 1:
                return "{ incomplete json"
            # Second attempt succeeds:
            return self._valid_payload()

        if self.mode == "markdown_fenced":
            return f"```json\n{self._valid_payload()}\n```"

        # Default "valid" mode
        return self._valid_payload()

    def _valid_payload(self) -> str:
        return json.dumps({
            "interpreted_goal": "Implement or assess prompt engineering directives with high fidelity.",
            "context": "Standard developer or analytical prompt scenario.",
            "ambiguities": ["Input data schema is not explicitly provided."],
            "missing_information": ["Target execution environment constraints."],
            "contradictions": [],
            "instruction_quality": {
                "score": 82.0,
                "reason": "Direct active imperative phrasing with clear scope.",
            },
            "strengths": ["Explicit task goal", "Clear deliverable description"],
            "weaknesses": ["Lack of negative boundary conditions"],
            "recommendations": [
                "Specify explicit parameter boundaries.",
                "Define error handling constraints.",
            ],
        })


# -----------------------------------------------------------------------------
# 2. Local Transformers Provider (Real Open-Source Local Inference)
# -----------------------------------------------------------------------------
class TransformersLocalProvider(LLMProvider):
    """Real local CPU-first inference provider using Hugging Face Transformers.

    Uses lazy loading, process-level caching, and thread safety.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._model: Any = None
        self._tokenizer: Any = None
        self._lock = threading.Lock()
        self._load_error: Exception | None = None
        self._status: LLMHealthStatus = LLMHealthStatus.MODEL_NOT_DOWNLOADED

    def health(self) -> LLMHealthStatus:
        if not self.config.enabled:
            return LLMHealthStatus.LLM_DISABLED
        if self._model is not None:
            return LLMHealthStatus.MODEL_AVAILABLE
        if self._load_error is not None:
            return LLMHealthStatus.MODEL_ERROR

        # Check if model files exist in local Hugging Face cache
        try:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(self.config.model, "config.json")
            if cached is not None and isinstance(cached, str):
                return LLMHealthStatus.MODEL_AVAILABLE
        except Exception:
            pass

        return LLMHealthStatus.MODEL_NOT_DOWNLOADED

    def model_metadata(self) -> dict[str, Any]:
        return {
            "name": self.config.model,
            "provider": "transformers",
            "version": "0.1.0",
            "context_length": self.config.context_length,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "device": "cpu",
        }

    def _ensure_loaded(self) -> None:
        """Lazy-load the model and tokenizer into memory if not already present."""
        if self._model is not None:
            return

        if not self.config.enabled:
            raise LLMDisabledError("Local LLM is disabled by configuration (LLM_ENABLED=false).")

        with self._lock:
            if self._model is not None:
                return

            self._status = LLMHealthStatus.MODEL_LOADING
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                LOGGER.info("Loading local model %s onto CPU...", self.config.model)
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model,
                    trust_remote_code=False,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.model,
                    torch_dtype=torch.float32,
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                )
                self._model.eval()
                self._status = LLMHealthStatus.MODEL_AVAILABLE
                LOGGER.info("Model %s loaded successfully.", self.config.model)
            except Exception as e:
                self._load_error = e
                self._status = LLMHealthStatus.MODEL_ERROR
                LOGGER.error("Failed to load local model %s: %s", self.config.model, e)
                raise ModelLoadingError(f"Failed to load model '{self.config.model}': {e}") from e

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        self._ensure_loaded()

        try:
            import torch

            # Format chat prompt using tokenizer chat template if available
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            if hasattr(self._tokenizer, "apply_chat_template"):
                input_text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                input_text = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

            inputs = self._tokenizer(input_text, return_tensors="pt")
            input_ids = inputs["input_ids"]

            generation_kwargs = {
                "max_new_tokens": min(max_tokens, self.config.max_tokens),
                "pad_token_id": self._tokenizer.eos_token_id,
            }
            if temperature > 0.0:
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = temperature
            else:
                generation_kwargs["do_sample"] = False

            with torch.no_grad():
                output_ids = self._model.generate(input_ids, **generation_kwargs)

            # Extract only the newly generated tokens
            generated_tokens = output_ids[0][input_ids.shape[1] :]
            response_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return response_text.strip()

        except Exception as e:
            raise ModelRuntimeError(f"Inference error during generation: {e}") from e


def create_provider(config: LLMConfig) -> LLMProvider:
    """Factory creating the appropriate LLM provider according to config."""
    if config.provider == "mock":
        return MockLLMProvider()
    if config.provider == "transformers":
        return TransformersLocalProvider(config)
    raise ValueError(f"Unsupported LLM provider: '{config.provider}'")
