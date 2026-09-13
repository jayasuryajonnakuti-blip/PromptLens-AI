"""Configuration constants and limits for the PromptLens AI Agent Loop (Step 19).

Invariants:
- MAX_OPTIMIZATION_ITERATIONS is strictly enforced in code. The loop is never unbounded.
- "Optimization is iterative, but bounded."
- "The Agent Loop never accepts an unvalidated prompt as successfully optimized."
- "PASS terminates the loop."
- "NEEDS_REVIEW terminates the loop."
- "Semantic similarity is supporting evidence and is not, by itself, proof of correctness."
"""
from __future__ import annotations

# Agent component version
AGENT_VERSION = "1.0.0"

# Strict loop iteration bounds — hard cap enforced by code
MAX_OPTIMIZATION_ITERATIONS = 3
DEFAULT_MAX_ITERATIONS = 3

# Minimum useful improvement threshold (score delta)
MIN_SCORE_IMPROVEMENT = 1.0

# Prompt bloat limit (ratio of optimized length to original length)
MAX_PROMPT_EXPANSION_RATIO = 4.0

# Semantic similarity floor below which semantic collapse is detected
MIN_SEMANTIC_SIMILARITY = 0.40

# Negligible prompt change threshold: edit distance or char delta ratio
MIN_CHAR_DELTA_RATIO = 0.02

# Maximum allowable consecutive score regressions before declaring non-improving
MAX_CONSECUTIVE_REGRESSIONS = 2

# Maximum allowable consecutive identical validation fatal errors
MAX_CONSECUTIVE_IDENTICAL_ERRORS = 2

# Bounded context token/char limit passed to iterative optimizer calls
AGENT_CONTEXT_CHAR_LIMIT = 2500

# Core architectural disclaimers
DISCLAIMER_BOUNDED_LOOP = "Optimization is iterative, but bounded."
DISCLAIMER_UNVALIDATED_NEVER_ACCEPTED = (
    "The Agent Loop never accepts an unvalidated prompt as successfully optimized."
)
DISCLAIMER_PASS_TERMINATES = "PASS terminates the loop."
DISCLAIMER_REVIEW_TERMINATES = "NEEDS_REVIEW terminates the loop."
DISCLAIMER_SEMANTIC_SIMILARITY = (
    "Semantic similarity is supporting evidence and is not, by itself, proof of correctness."
)

AGENT_DISCLAIMERS = [
    DISCLAIMER_BOUNDED_LOOP,
    DISCLAIMER_UNVALIDATED_NEVER_ACCEPTED,
    DISCLAIMER_PASS_TERMINATES,
    DISCLAIMER_REVIEW_TERMINATES,
    DISCLAIMER_SEMANTIC_SIMILARITY,
]
