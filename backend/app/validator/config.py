"""Configuration and thresholds for PromptLens AI Validator (Step 18).

The Validator acts as an independent safety and structural gate.
Thresholds here are tuned specifically for final acceptance criteria
(distinct from the Critic's exploratory review metrics).
"""
from __future__ import annotations

VALIDATOR_VERSION = "1.0.0"

# --- Semantic Similarity Thresholds ---
# Below 0.35 indicates radical departure from the original subject matter (FAIL)
MIN_SEMANTIC_SIMILARITY_FAIL: float = 0.35
# Below 0.55 indicates potential semantic drift requiring human review (NEEDS_REVIEW)
MIN_SEMANTIC_SIMILARITY_WARN: float = 0.55

# --- Quality Score Regression Thresholds (Step 13 0–100 scale) ---
# A quality drop > 12.0 points indicates an unacceptable deterioration (FAIL)
MAX_SCORE_REGRESSION_FAIL: float = 12.0
# A quality drop > 4.0 points warrants review before acceptance (NEEDS_REVIEW / WARN)
MAX_SCORE_REGRESSION_WARN: float = 4.0

# --- Expansion and Verbosity Ratios (optimized_words / original_words) ---
# Expansion > 5.5x represents extreme bloat and token inefficiency (FAIL)
MAX_EXPANSION_RATIO_FAIL: float = 5.5
# Expansion > 3.5x indicates notable verbosity that should be reviewed (WARN)
MAX_EXPANSION_RATIO_WARN: float = 3.5
# Excessive reduction (< 30% of original words for prompts > 25 words) risks dropping context
MAX_TRUNCATION_RATIO_WARN: float = 0.30

# --- Safety & Requirement Tolerances ---
# Zero tolerance for lost critical requirements (e.g. format constraints, explicit exclusions)
MAX_LOST_CRITICAL_REQUIREMENTS: int = 0

# Zero tolerance for introduced contradictions
MAX_FATAL_CONTRADICTIONS: int = 0

# --- Context Budget Limit for Validator LLM Input ---
VALIDATOR_CONTEXT_CHAR_LIMIT: int = 2400
