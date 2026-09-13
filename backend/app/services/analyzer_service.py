"""AnalyzerService: process-level singleton wrapping PromptAnalyzer (Step 15).

Follows the same singleton + test-isolation pattern as LLMService (Step 14):
- get_analyzer_service() returns the process-level singleton.
- set_analyzer_service(None) resets the singleton for test tearDown().
"""
from __future__ import annotations

from app.analyzer.analyzer import PromptAnalyzer
from app.analyzer.schemas import AnalyzerAnalysis


class AnalyzerService:
    """Thin service wrapper around PromptAnalyzer providing lifecycle management."""

    def __init__(self, analyzer: PromptAnalyzer | None = None) -> None:
        self._analyzer = analyzer or PromptAnalyzer()

    def analyze(self, prompt: str) -> AnalyzerAnalysis:
        """Delegate to the underlying PromptAnalyzer."""
        return self._analyzer.analyze(prompt)


_SERVICE_INSTANCE: AnalyzerService | None = None


def get_analyzer_service() -> AnalyzerService:
    """Return the process-level singleton AnalyzerService."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = AnalyzerService()
    return _SERVICE_INSTANCE


def set_analyzer_service(service: AnalyzerService | None) -> None:
    """Set or reset the global AnalyzerService instance (for test isolation)."""
    global _SERVICE_INSTANCE
    _SERVICE_INSTANCE = service
