"""Typed exception hierarchy for PromptLens AI Agent Loop (Step 19)."""
from __future__ import annotations


class AgentException(Exception):
    """Base exception for all Agent Loop operations."""


class AgentConfigurationError(AgentException):
    """Raised when an invalid agent loop configuration is supplied."""


class AgentStateError(AgentException):
    """Raised when the agent state is corrupted, missing, or in an invalid state."""


class MaxIterationsExceededError(AgentException):
    """Raised if execution is attempted beyond the hard iteration cap."""


class AgentServiceUnavailableError(AgentException):
    """Raised when an essential upstream service (Analyzer, Optimizer, Critic, Validator) is unavailable."""
