"""Configuration for PromptLens AI Optimizer (Step 16)."""
from __future__ import annotations

OPTIMIZER_VERSION = "1.0.0"

# Character limit for the analyzer summary block sent to the LLM.
ANALYSIS_SUMMARY_CHAR_LIMIT: int = 1800

# Maximum placeholder tokens the LLM may insert for missing information.
# Keeps the optimized prompt concise even when a lot of info is missing.
MAX_PLACEHOLDERS: int = 6

# How many Step 15 recommendations to forward to the optimizer LLM.
MAX_RECOMMENDATIONS_IN_CONTEXT: int = 5
