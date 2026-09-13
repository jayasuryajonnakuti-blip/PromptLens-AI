"""CriticService: process-level singleton wrapping PromptCritic (Step 17).

Follows the same singleton + test-isolation pattern as OptimizerService (Step 16),
AnalyzerService (Step 15), and LLMService (Step 14):
- get_critic_service() returns the process-level singleton.
- set_critic_service(None) resets the singleton for test tearDown().
"""
from __future__ import annotations

from app.critic.critic import PromptCritic
from app.critic.schemas import CriticEvaluation, OptimizationMetadataInput


class CriticService:
    """Thin service wrapper around PromptCritic providing lifecycle management."""

    def __init__(self, critic: PromptCritic | None = None) -> None:
        self._critic = critic or PromptCritic()

    def evaluate(
        self,
        original_prompt: str,
        optimized_prompt: str,
        optimization_metadata: OptimizationMetadataInput | None = None,
    ) -> CriticEvaluation:
        """Delegate evaluation to PromptCritic."""
        return self._critic.evaluate(
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt,
            optimization_metadata=optimization_metadata,
        )


_SERVICE_INSTANCE: CriticService | None = None


def get_critic_service() -> CriticService:
    """Return the process-level singleton CriticService."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = CriticService()
    return _SERVICE_INSTANCE


def set_critic_service(service: CriticService | None) -> None:
    """Set or reset the global CriticService instance (for test isolation)."""
    global _SERVICE_INSTANCE
    _SERVICE_INSTANCE = service
