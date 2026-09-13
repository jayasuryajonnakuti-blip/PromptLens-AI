"""LLM service orchestrating generation, JSON extraction, validation, and retries (Step 14)."""
import json
import logging
import re
import time
from typing import Any

from app.llm.config import LLMConfig, load_llm_config
from app.llm.errors import (
    ContextOverflowError,
    InvalidGenerationError,
    InvalidJsonError,
    LLMDisabledError,
    ModelLoadingError,
    ModelNotDownloadedError,
    ModelRuntimeError,
    RetryExhaustedError,
    SchemaValidationError,
)
from app.llm.model import create_provider
from app.llm.prompts import (
    PROMPTLENS_LLM_ANALYSIS_V1,
    apply_context_budget,
    build_analysis_input,
    build_correction_prompt,
)
from app.llm.provider import LLMProvider
from app.llm.schemas import (
    LLMAnalysisResponse,
    LLMHealthStatus,
    LLMModelMetadata,
    PromptAnalysisOutput,
)
from app.services.intent_service import classify_prompt
from app.services.preprocessing_service import preprocess_prompt
from app.services.quality_service import predict_quality
from app.services.scoring_service import score_prompt

LOGGER = logging.getLogger(__name__)


class LLMService:
    """Orchestrator for PromptLens local LLM operations."""

    def __init__(self, config: LLMConfig | None = None, provider: LLMProvider | None = None) -> None:
        self.config = config or load_llm_config()
        self.provider = provider or create_provider(self.config)

    def health(self) -> LLMHealthStatus:
        """Check operational state of the configured provider."""
        return self.provider.health()

    def analyze_prompt(self, prompt: str) -> LLMAnalysisResponse:
        """Analyze a user prompt using the local LLM with structured output guarantees.

        Executes:
        1. Input validation & context assembly (reusing Steps 8-13)
        2. Context budget management and safe truncation
        3. Model generation
        4. Defensive JSON extraction
        5. Pydantic validation with targeted error correction retries
        6. Returns typed response with latency metadata
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty or whitespace-only")

        # Verify operational status
        health_state = self.health()
        if health_state == LLMHealthStatus.LLM_DISABLED:
            raise LLMDisabledError(
                "Local LLM is disabled by configuration. Enable it by setting LLM_ENABLED=true."
            )
        if health_state == LLMHealthStatus.MODEL_NOT_DOWNLOADED:
            raise ModelNotDownloadedError(
                f"Model weights for '{self.config.model}' are not present locally. "
                "Download or cache the model before running inference."
            )
        if health_state == LLMHealthStatus.MODEL_ERROR:
            raise ModelRuntimeError("Local LLM is in an error state and cannot accept requests.")

        t0 = time.perf_counter()

        # Step 1: Assemble compact analysis context from Steps 8-13
        context_summary = self._build_context_summary(prompt)

        # Step 2: Context budgeting & safe truncation
        budgeted_prompt, was_truncated = apply_context_budget(
            prompt=prompt,
            max_context_length=self.config.context_length,
            max_tokens=self.config.max_tokens,
        )
        if was_truncated:
            LOGGER.info("Prompt exceeded context length and was safely truncated.")

        input_text = build_analysis_input(budgeted_prompt, context_summary)

        # Step 3: Generation & Retry Loop
        analysis_output = self._generate_and_validate(input_text)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        meta = self.provider.model_metadata()

        return LLMAnalysisResponse(
            analysis=analysis_output,
            model=LLMModelMetadata(name=meta.get("name", self.config.model)),
            provider=meta.get("provider", self.config.provider),
            latency_ms=latency_ms,
            structured_output_valid=True,
        )

    def _generate_and_validate(self, input_text: str) -> PromptAnalysisOutput:
        """Execute generation and validate JSON output with targeted retries."""
        current_prompt = input_text
        max_attempts = 1 + self.config.max_retries
        last_error = ""

        for attempt in range(1, max_attempts + 1):
            try:
                raw_response = self.provider.generate(
                    prompt=current_prompt,
                    system_prompt=PROMPTLENS_LLM_ANALYSIS_V1,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            except Exception as e:
                LOGGER.error("LLM provider generation error on attempt %d: %s", attempt, e)
                raise ModelRuntimeError(f"Model generation failed: {e}") from e

            if not raw_response or not raw_response.strip():
                last_error = "Model generated empty response."
                if attempt < max_attempts:
                    current_prompt = build_correction_prompt(raw_response, last_error)
                    continue
                raise InvalidGenerationError("Model returned empty generation after retries.")

            # Defensive extraction & parsing
            try:
                parsed_json = self._extract_json(raw_response)
            except InvalidJsonError as e:
                last_error = str(e)
                LOGGER.warning("JSON parsing failed on attempt %d: %s", attempt, e)
                if attempt < max_attempts:
                    current_prompt = build_correction_prompt(raw_response, last_error)
                    continue
                raise RetryExhaustedError(
                    f"Failed to extract valid JSON after {self.config.max_retries} retries: {last_error}"
                ) from e

            # Pydantic schema validation
            try:
                validated = PromptAnalysisOutput.model_validate(parsed_json)
                return validated
            except Exception as e:
                last_error = f"Schema validation failed: {e}"
                LOGGER.warning("Schema validation failed on attempt %d: %s", attempt, e)
                if attempt < max_attempts:
                    current_prompt = build_correction_prompt(raw_response, last_error)
                    continue
                raise RetryExhaustedError(
                    f"Output failed schema validation after {self.config.max_retries} retries: {last_error}"
                ) from e

        raise RetryExhaustedError(f"Retry limit exhausted without valid output: {last_error}")

    def _extract_json(self, raw_text: str) -> dict[str, Any]:
        """Extract and parse JSON from raw model output defensively."""
        text = raw_text.strip()

        # Check for markdown code blocks (```json ... ``` or ``` ... ```)
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            candidate = fence_match.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                raise InvalidJsonError(f"Malformed JSON within markdown fences: {e}") from e

        # If text is already a direct JSON object
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                raise InvalidJsonError(f"Malformed JSON object: {e}") from e

        # Find first '{' and matching last '}'
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx > start_idx:
            candidate = text[start_idx : end_idx + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                raise InvalidJsonError(f"Malformed embedded JSON substring: {e}") from e

        raise InvalidJsonError(f"No JSON object structure found in response: '{text[:120]}...'")

    def _build_context_summary(self, prompt: str) -> dict[str, Any]:
        """Build compact analysis summary from Steps 8-13 without duplicating logic."""
        summary: dict[str, Any] = {}

        try:
            prep = preprocess_prompt(prompt)
            summary["word_count"] = prep.text.word_count
            summary["output_formats"] = prep.output_formats
            summary["sections_detected"] = prep.sections
        except Exception:
            pass

        try:
            intent = classify_prompt(prompt)
            summary["detected_intent"] = intent.intent.value
            summary["intent_confidence"] = intent.confidence
        except Exception:
            pass

        try:
            quality = predict_quality(prompt)
            summary["baseline_quality_score"] = quality.score
            summary["quality_category"] = quality.label.value
        except Exception:
            pass

        return summary


_SERVICE_INSTANCE: LLMService | None = None


def get_llm_service() -> LLMService:
    """Return process-level singleton instance of the LLMService."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = LLMService()
    return _SERVICE_INSTANCE


def set_llm_service(service: LLMService | None) -> None:
    """Set or reset the global LLMService instance (useful for test isolation)."""
    global _SERVICE_INSTANCE
    _SERVICE_INSTANCE = service
