"""Typed repository layer for AgentRun database entities (Step 20).

Invariants:
- Transaction management: commits on success, rolls back on failure.
- Translates raw SQLAlchemy exceptions into typed DatabaseError exceptions.
- Zero business logic or Agent orchestration lives here.
- Strict ordering by created_at DESC for chronological retrieval.
"""
from __future__ import annotations

import logging
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.errors import (
    AgentRunNotFoundError,
    DatabaseError,
    DatabaseIntegrityError,
)
from app.db.models import AgentRun

LOGGER = logging.getLogger(__name__)


class AgentRunRepository:
    """Repository handling CRUD operations and queries for AgentRun entities."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, agent_run: AgentRun) -> AgentRun:
        """Persist a new AgentRun entity.

        Args:
            agent_run: Unpersisted AgentRun instance.

        Returns:
            The persisted and refreshed AgentRun instance.

        Raises:
            DatabaseIntegrityError: If a constraint or schema invariant is violated.
            DatabaseError: If any other database error occurs.
        """
        try:
            self._session.add(agent_run)
            self._session.commit()
            self._session.refresh(agent_run)
            return agent_run
        except IntegrityError as exc:
            self._session.rollback()
            LOGGER.warning("Integrity error persisting AgentRun: %s", exc)
            raise DatabaseIntegrityError(f"Data integrity violation: {exc.orig or exc}") from exc
        except SQLAlchemyError as exc:
            self._session.rollback()
            LOGGER.error("Database error persisting AgentRun: %s", exc, exc_info=True)
            raise DatabaseError(f"Database operation failed: {exc}") from exc

    def get(self, run_id: str) -> AgentRun | None:
        """Fetch an AgentRun by its UUID string identifier.

        Returns None if no matching record is found.
        """
        try:
            stmt = select(AgentRun).where(AgentRun.id == run_id)
            return self._session.scalars(stmt).first()
        except SQLAlchemyError as exc:
            LOGGER.error("Database error retrieving AgentRun '%s': %s", run_id, exc, exc_info=True)
            raise DatabaseError(f"Failed to retrieve AgentRun '{run_id}': {exc}") from exc

    def get_or_raise(self, run_id: str) -> AgentRun:
        """Fetch an AgentRun by ID, raising AgentRunNotFoundError if missing."""
        run = self.get(run_id)
        if run is None:
            raise AgentRunNotFoundError(f"AgentRun with id '{run_id}' not found.")
        return run

    def list(self, skip: int = 0, limit: int = 50) -> list[AgentRun]:
        """List AgentRun records ordered by created_at DESC with pagination."""
        try:
            stmt = (
                select(AgentRun)
                .order_by(AgentRun.created_at.desc())
                .offset(max(skip, 0))
                .limit(max(min(limit, 100), 1))
            )
            return list(self._session.scalars(stmt).all())
        except SQLAlchemyError as exc:
            LOGGER.error("Database error listing AgentRuns: %s", exc, exc_info=True)
            raise DatabaseError(f"Failed to list AgentRuns: {exc}") from exc

    def delete(self, run_id: str) -> bool:
        """Delete an AgentRun record by ID.

        Returns True if a record was deleted, False if not found.
        """
        try:
            run = self.get(run_id)
            if run is None:
                return False

            self._session.delete(run)
            self._session.commit()
            return True
        except SQLAlchemyError as exc:
            self._session.rollback()
            LOGGER.error("Database error deleting AgentRun '%s': %s", run_id, exc, exc_info=True)
            raise DatabaseError(f"Failed to delete AgentRun '{run_id}': {exc}") from exc

    def count(self) -> int:
        """Return the total count of persisted AgentRun records."""
        try:
            stmt = select(func.count()).select_from(AgentRun)
            count_val = self._session.scalar(stmt)
            return int(count_val or 0)
        except SQLAlchemyError as exc:
            LOGGER.error("Database error counting AgentRuns: %s", exc, exc_info=True)
            raise DatabaseError(f"Failed to count AgentRuns: {exc}") from exc
