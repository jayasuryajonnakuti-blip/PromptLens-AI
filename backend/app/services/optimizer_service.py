"""OptimizerService: process-level singleton wrapping PromptOptimizer (Step 16).

Follows the same pattern as AnalyzerService (Step 15) and LLMService (Step 14):
- get_optimizer_service() returns the process-level singleton.
- set_optimizer_service(None) resets the singleton for test tearDown().
"""
from __future__ import annotations

from typing import Any

from app.optimizer.optimizer import PromptOptimizer
from app.optimizer.schemas import OptimizationMode, OptimizerResult


class OptimizerService:
    """Thin service wrapper around PromptOptimizer providing lifecycle management."""

    def __init__(self, optimizer: PromptOptimizer | None = None) -> None:
        self._optimizer = optimizer or PromptOptimizer()

    def optimize(
        self,
        prompt: str,
        mode: OptimizationMode = OptimizationMode.BALANCED,
        analyzer_result: dict[str, Any] | None = None,
    ) -> OptimizerResult:
        """Delegate to the underlying PromptOptimizer."""
        return self._optimizer.optimize(
            prompt=prompt,
            mode=mode,
            analyzer_result=analyzer_result,
        )


_SERVICE_INSTANCE: OptimizerService | None = None


def get_optimizer_service() -> OptimizerService:
    """Return the process-level singleton OptimizerService."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = OptimizerService()
    return _SERVICE_INSTANCE


def set_optimizer_service(service: OptimizerService | None) -> None:
    """Set or reset the global OptimizerService instance (for test isolation)."""
    global _SERVICE_INSTANCE
    _SERVICE_INSTANCE = service
