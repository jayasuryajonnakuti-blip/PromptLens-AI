"""ValidatorService: process-level singleton wrapping PromptValidator (Step 18).

Follows the same singleton + test-isolation pattern as CriticService (Step 17),
OptimizerService (Step 16), AnalyzerService (Step 15), and LLMService (Step 14):
- get_validator_service() returns the process-level singleton.
- set_validator_service(None) resets the singleton for test tearDown().
"""
from __future__ import annotations

from app.validator.schemas import (
    CriticResultInput,
    OptimizationMetadataInput,
    ValidationResult,
)
from app.validator.validator import PromptValidator


class ValidatorService:
    """Thin service wrapper around PromptValidator providing lifecycle management."""

    def __init__(self, validator: PromptValidator | None = None) -> None:
        self._validator = validator or PromptValidator()

    def validate(
        self,
        original_prompt: str,
        optimized_prompt: str,
        critic_result: CriticResultInput | None = None,
        optimization_metadata: OptimizationMetadataInput | None = None,
    ) -> ValidationResult:
        """Delegate validation to PromptValidator."""
        return self._validator.validate(
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt,
            critic_result=critic_result,
            optimization_metadata=optimization_metadata,
        )


_SERVICE_INSTANCE: ValidatorService | None = None


def get_validator_service() -> ValidatorService:
    """Return the process-level singleton ValidatorService."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = ValidatorService()
    return _SERVICE_INSTANCE


def set_validator_service(service: ValidatorService | None) -> None:
    """Set or reset the global ValidatorService instance (for test isolation)."""
    global _SERVICE_INSTANCE
    _SERVICE_INSTANCE = service
