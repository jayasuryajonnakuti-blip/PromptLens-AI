"""Typed exception hierarchy for PromptLens AI Database & Persistence Layer (Step 20)."""
from __future__ import annotations


class DatabaseError(Exception):
    """Base exception for all database and persistence operations."""


class DatabaseConfigurationError(DatabaseError):
    """Raised when database configuration is invalid or cannot be initialized."""


class DatabaseIntegrityError(DatabaseError):
    """Raised when a database constraint or data integrity violation occurs."""


class AgentRunNotFoundError(DatabaseError):
    """Raised when a requested AgentRun record does not exist."""
