"""Configuration and decision thresholds for PromptLens AI Critic (Step 17).

Provides explicit, documented thresholds for deterministic evaluation
and context budget limits for LLM input assembly.
"""
from __future__ import annotations

CRITIC_VERSION = "1.0.0"

# Maximum length for the context block sent to the LLM
CRITIC_CONTEXT_CHAR_LIMIT: int = 2200

# Scoring regression thresholds (Step 13 score deltas)
# Score drop > 15.0 is an automatic critical regression (FAIL)
SCORE_REGRESSION_FAIL_THRESHOLD: float = 15.0

# Score drop > 5.0 triggers an issue warning (NEEDS_REVIEW or penalized score)
SCORE_REGRESSION_WARN_THRESHOLD: float = 5.0

# Critical dimension score drop threshold
DIMENSION_REGRESSION_FAIL_THRESHOLD: float = 20.0
DIMENSION_REGRESSION_WARN_THRESHOLD: float = 10.0

# Semantic similarity thresholds
# Extremely low similarity (< 0.40) strongly indicates complete topic departure
SEMANTIC_SIMILARITY_SUSPICIOUS: float = 0.40

# High similarity threshold above which prompts are semantically aligned
SEMANTIC_SIMILARITY_ALIGNED: float = 0.70

# Length and verbosity ratios (optimized_words / original_words)
# Length expansion > 4.0x warns about unnecessary verbosity
LENGTH_EXPANSION_WARN_RATIO: float = 4.0

# Length expansion > 7.0x is severe bloating
LENGTH_EXPANSION_FAIL_RATIO: float = 7.0

# Length reduction < 0.25x warns about possible severe truncation
LENGTH_REDUCTION_WARN_RATIO: float = 0.25
