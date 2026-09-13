"""Configuration for PromptLens AI Analyzer (Step 15).

Keeps configuration minimal and explicit.  All defaults are conservative so
the analyzer degrades gracefully when the LLM layer (Step 14) is unavailable.
"""
from __future__ import annotations

ANALYZER_VERSION = "1.0.0"

# Minimum confidence thresholds below which deterministic evidence is still
# included but labelled with lower confidence.
EVIDENCE_CONFIDENCE_MIN: float = 0.10

# Minimum intent confidence before we consider the label reliable evidence.
INTENT_CONFIDENCE_THRESHOLD: float = 0.50

# How many top scoring/quality signals to include in context sent to the LLM.
MAX_DIMENSION_SUMMARIES: int = 6

# Maximum tokens to spend on the upstream evidence block in the analyzer prompt.
EVIDENCE_BLOCK_CHAR_LIMIT: int = 1600
