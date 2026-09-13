"""Unit tests for AgentRunService persistence service (Step 20).

Tests:
- save_agent_run() serializes and persists a complete AgentResponse.
- JSON fields (iterations, score_history, critic_result, validation, metrics, run_metadata) stored correctly.
- get_agent_run() retrieves by run_id.
- list_agent_runs() returns records in order with pagination.
- delete_agent_run() removes a record.
- count_agent_runs() returns the correct count.
- Singleton lifecycle: get_agent_run_service() / set_agent_run_service().
- Test isolation: in-memory SQLite per test class via reset_db_engine().
"""
from __future__ import annotations

import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent.schemas import (
    AgentIteration,
    AgentMetrics,
    AgentResponse,
    AgentStatus,
    AgentTerminationReason,
    CriticResultSummary,
    OptimizerResultSummary,
)
from app.db.base import Base
from app.db.models import AgentRun
from app.db.session import reset_db_engine
from app.services.agent_run_service import (
    AgentRunService,
    get_agent_run_service,
    set_agent_run_service,
)
from app.validator.schemas import (
    PromptMetadata,
    ValidationMetadata,
    ValidationMode,
    ValidationResult,
    ValidatorDecision,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prompt_metadata() -> PromptMetadata:
    return PromptMetadata(
        prompt_length=22,
        word_count=4,
        sentence_count=1,
        detected_intent="question",
        intent_confidence=0.9,
        quality_score=75.0,
    )


def _make_validation_metadata() -> ValidationMetadata:
    return ValidationMetadata(
        validator_version="1.0.0",
        validation_mode=ValidationMode.MOCK,
        model="none",
        latency_ms=15.0,
        critic_consistency_checked=False,
    )


def _make_validation_result(decision: ValidatorDecision = ValidatorDecision.PASS) -> ValidationResult:
    meta = _make_prompt_metadata()
    return ValidationResult(
        decision=decision,
        is_valid=(decision == ValidatorDecision.PASS),
        safety_score=85.0,
        original_metadata=meta,
        optimized_metadata=meta,
        issues=[],
        metadata=_make_validation_metadata(),
    )


def _make_optimizer_result() -> OptimizerResultSummary:
    return OptimizerResultSummary(
        optimized_prompt="Optimized prompt text.",
        summary="Minor clarity improvements applied.",
        changes=[{"category": "clarity", "description": "Reworded opening."}],
        preserved_requirements=[{"requirement": "no jargon", "status": "preserved"}],
        placeholders_inserted=[],
        improvement_score_delta=3.5,
        optimizer_mode="MOCK",
    )


def _make_critic_result(decision: str = "PASS") -> CriticResultSummary:
    return CriticResultSummary(
        decision=decision,
        overall_critique_score=82.0,
        issues=[],
        lost_requirements=[],
        introduced_requirements=[],
        unsupported_assumptions=[],
        strengths=["Clear intent"],
        weaknesses=[],
        recommendations=[],
        semantic_similarity=0.95,
    )


def _make_agent_iteration(n: int = 1) -> AgentIteration:
    return AgentIteration(
        iteration_number=n,
        prompt_before="Original prompt text.",
        prompt_after="Optimized prompt text.",
        optimizer_result=_make_optimizer_result(),
        critic_result=_make_critic_result("PASS"),
        validator_result=_make_validation_result(ValidatorDecision.PASS),
        score_before=70.0,
        score_after=85.0,
        decision="PASS",
        latency_ms=120.0,
    )


def _make_agent_response(
    *,
    include_iterations: bool = True,
    final_score: float | None = 85.0,
    status: AgentStatus = AgentStatus.COMPLETED,
    termination_reason: AgentTerminationReason = AgentTerminationReason.VALIDATED,
    is_validated: bool = True,
) -> AgentResponse:
    iterations = [_make_agent_iteration(1)] if include_iterations else []
    return AgentResponse(
        original_prompt="Original prompt text.",
        final_prompt="Optimized prompt text.",
        status=status,
        termination_reason=termination_reason,
        iteration_count=len(iterations),
        is_validated=is_validated,
        iterations=iterations,
        final_critic_result=_make_critic_result("PASS") if include_iterations else None,
        final_validation=_make_validation_result() if include_iterations else None,
        final_score=final_score,
        metrics=AgentMetrics(
            total_latency_ms=120.0,
            total_iterations=len(iterations),
            initial_score=70.0 if include_iterations else None,
            final_score=final_score,
            score_delta=15.0 if include_iterations else None,
            prompt_length_before=22,
            prompt_length_after=24,
            expansion_ratio=1.09,
            semantic_similarity=0.95 if include_iterations else None,
        ),
    )


def _make_isolated_service():
    """Return (AgentRunService, engine) backed by an isolated in-memory SQLite DB."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return AgentRunService(session_factory=factory), engine


def _teardown_engine(engine) -> None:
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


# ---------------------------------------------------------------------------
# Test: save_agent_run
# ---------------------------------------------------------------------------
class TestAgentRunServiceSave(unittest.TestCase):
    """Tests for AgentRunService.save_agent_run() persistence and serialization."""

    def setUp(self):
        reset_db_engine()
        set_agent_run_service(None)
        self.svc, self.engine = _make_isolated_service()

    def tearDown(self):
        _teardown_engine(self.engine)
        reset_db_engine()
        set_agent_run_service(None)

    def test_save_returns_agent_run_instance(self):
        run = self.svc.save_agent_run(_make_agent_response(), mode="balanced")
        self.assertIsInstance(run, AgentRun)

    def test_save_assigns_string_id(self):
        run = self.svc.save_agent_run(_make_agent_response(), mode="balanced")
        self.assertIsNotNone(run.id)
        self.assertIsInstance(run.id, str)
        self.assertGreater(len(run.id), 0)

    def test_save_preserves_original_prompt(self):
        run = self.svc.save_agent_run(_make_agent_response(), mode="balanced")
        self.assertEqual(run.original_prompt, "Original prompt text.")

    def test_save_preserves_final_prompt(self):
        run = self.svc.save_agent_run(_make_agent_response(), mode="balanced")
        self.assertEqual(run.final_prompt, "Optimized prompt text.")

    def test_save_status_string(self):
        run = self.svc.save_agent_run(_make_agent_response(), mode="balanced")
        self.assertEqual(run.status, "COMPLETED")

    def test_save_termination_reason_string(self):
        run = self.svc.save_agent_run(_make_agent_response(), mode="balanced")
        self.assertEqual(run.termination_reason, "VALIDATED")

    def test_save_is_validated_true(self):
        run = self.svc.save_agent_run(_make_agent_response(is_validated=True), mode="balanced")
        self.assertTrue(run.is_validated)

    def test_save_is_validated_false(self):
        run = self.svc.save_agent_run(
            _make_agent_response(
                is_validated=False,
                status=AgentStatus.NEEDS_REVIEW,
                termination_reason=AgentTerminationReason.VALIDATION_REVIEW,
            ),
            mode="analytical",
        )
        self.assertFalse(run.is_validated)

    def test_save_iteration_count(self):
        run = self.svc.save_agent_run(_make_agent_response(), mode="balanced")
        self.assertEqual(run.iteration_count, 1)

    def test_save_final_score(self):
        run = self.svc.save_agent_run(_make_agent_response(final_score=85.0), mode="balanced")
        self.assertAlmostEqual(run.final_score, 85.0)

    def test_save_final_score_none(self):
        run = self.svc.save_agent_run(
            _make_agent_response(
                final_score=None,
                include_iterations=False,
                is_validated=False,
                status=AgentStatus.FAILED,
                termination_reason=AgentTerminationReason.VALIDATION_FAILED,
            ),
            mode="creative",
        )
        self.assertIsNone(run.final_score)

    def test_save_mode_stored(self):
        run = self.svc.save_agent_run(_make_agent_response(), mode="expert")
        self.assertEqual(run.mode, "expert")

    def test_save_agent_version(self):
        run = self.svc.save_agent_run(_make_agent_response(), mode="balanced")
        self.assertEqual(run.agent_version, "1.0.0")

    def test_save_score_history_contains_before_and_after(self):
        run = self.svc.save_agent_run(_make_agent_response(include_iterations=True), mode="balanced")
        self.assertIsInstance(run.score_history, list)
        self.assertIn(70.0, run.score_history)
        self.assertIn(85.0, run.score_history)

    def test_save_iterations_is_list_of_dicts(self):
        run = self.svc.save_agent_run(_make_agent_response(include_iterations=True), mode="balanced")
        self.assertIsInstance(run.iterations, list)
        self.assertEqual(len(run.iterations), 1)
        self.assertIsInstance(run.iterations[0], dict)

    def test_save_iterations_empty_when_no_iterations(self):
        run = self.svc.save_agent_run(
            _make_agent_response(
                include_iterations=False,
                is_validated=False,
                final_score=None,
                status=AgentStatus.FAILED,
                termination_reason=AgentTerminationReason.VALIDATION_FAILED,
            ),
            mode="balanced",
        )
        self.assertEqual(run.iterations, [])

    def test_save_run_metadata_has_disclaimers(self):
        run = self.svc.save_agent_run(_make_agent_response(), mode="balanced")
        self.assertIn("disclaimers", run.run_metadata)
        self.assertIsInstance(run.run_metadata["disclaimers"], list)

    def test_save_with_custom_run_id(self):
        custom_id = "00000000-0000-0000-0000-000000000001"
        run = self.svc.save_agent_run(_make_agent_response(), mode="balanced", run_id=custom_id)
        self.assertEqual(run.id, custom_id)

    def test_save_timestamps_populated(self):
        run = self.svc.save_agent_run(_make_agent_response(), mode="balanced")
        self.assertIsNotNone(run.created_at)
        self.assertIsNotNone(run.updated_at)


# ---------------------------------------------------------------------------
# Test: get_agent_run
# ---------------------------------------------------------------------------
class TestAgentRunServiceGet(unittest.TestCase):
    """Tests for AgentRunService.get_agent_run()."""

    def setUp(self):
        reset_db_engine()
        set_agent_run_service(None)
        self.svc, self.engine = _make_isolated_service()

    def tearDown(self):
        _teardown_engine(self.engine)
        reset_db_engine()
        set_agent_run_service(None)

    def test_get_returns_persisted_run(self):
        run = self.svc.save_agent_run(_make_agent_response(), mode="balanced")
        retrieved = self.svc.get_agent_run(run.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, run.id)

    def test_get_nonexistent_returns_none(self):
        self.assertIsNone(self.svc.get_agent_run("nonexistent-id-12345"))

    def test_get_preserves_prompt_content(self):
        run = self.svc.save_agent_run(_make_agent_response(), mode="balanced")
        retrieved = self.svc.get_agent_run(run.id)
        self.assertEqual(retrieved.original_prompt, "Original prompt text.")
        self.assertEqual(retrieved.final_prompt, "Optimized prompt text.")


# ---------------------------------------------------------------------------
# Test: list_agent_runs
# ---------------------------------------------------------------------------
class TestAgentRunServiceList(unittest.TestCase):
    """Tests for AgentRunService.list_agent_runs()."""

    def setUp(self):
        reset_db_engine()
        set_agent_run_service(None)
        self.svc, self.engine = _make_isolated_service()

    def tearDown(self):
        _teardown_engine(self.engine)
        reset_db_engine()
        set_agent_run_service(None)

    def test_list_empty_initially(self):
        self.assertEqual(self.svc.list_agent_runs(), [])

    def test_list_returns_all_records(self):
        for _ in range(3):
            self.svc.save_agent_run(_make_agent_response(), mode="balanced")
        self.assertEqual(len(self.svc.list_agent_runs()), 3)

    def test_list_pagination_skip(self):
        for _ in range(3):
            self.svc.save_agent_run(_make_agent_response(), mode="balanced")
        runs = self.svc.list_agent_runs(skip=2, limit=10)
        self.assertEqual(len(runs), 1)

    def test_list_pagination_limit(self):
        for _ in range(3):
            self.svc.save_agent_run(_make_agent_response(), mode="balanced")
        runs = self.svc.list_agent_runs(skip=0, limit=2)
        self.assertEqual(len(runs), 2)


# ---------------------------------------------------------------------------
# Test: delete_agent_run
# ---------------------------------------------------------------------------
class TestAgentRunServiceDelete(unittest.TestCase):
    """Tests for AgentRunService.delete_agent_run()."""

    def setUp(self):
        reset_db_engine()
        set_agent_run_service(None)
        self.svc, self.engine = _make_isolated_service()

    def tearDown(self):
        _teardown_engine(self.engine)
        reset_db_engine()
        set_agent_run_service(None)

    def test_delete_existing_returns_true(self):
        run = self.svc.save_agent_run(_make_agent_response(), mode="balanced")
        self.assertTrue(self.svc.delete_agent_run(run.id))

    def test_delete_removes_record(self):
        run = self.svc.save_agent_run(_make_agent_response(), mode="balanced")
        self.svc.delete_agent_run(run.id)
        self.assertIsNone(self.svc.get_agent_run(run.id))

    def test_delete_nonexistent_returns_false(self):
        self.assertFalse(self.svc.delete_agent_run("nonexistent-id-xyz"))


# ---------------------------------------------------------------------------
# Test: count_agent_runs
# ---------------------------------------------------------------------------
class TestAgentRunServiceCount(unittest.TestCase):
    """Tests for AgentRunService.count_agent_runs()."""

    def setUp(self):
        reset_db_engine()
        set_agent_run_service(None)
        self.svc, self.engine = _make_isolated_service()

    def tearDown(self):
        _teardown_engine(self.engine)
        reset_db_engine()
        set_agent_run_service(None)

    def test_count_empty_is_zero(self):
        self.assertEqual(self.svc.count_agent_runs(), 0)

    def test_count_after_saves(self):
        for _ in range(4):
            self.svc.save_agent_run(_make_agent_response(), mode="balanced")
        self.assertEqual(self.svc.count_agent_runs(), 4)

    def test_count_decrements_after_delete(self):
        run = self.svc.save_agent_run(_make_agent_response(), mode="balanced")
        self.svc.save_agent_run(_make_agent_response(), mode="analytical")
        self.assertEqual(self.svc.count_agent_runs(), 2)
        self.svc.delete_agent_run(run.id)
        self.assertEqual(self.svc.count_agent_runs(), 1)


# ---------------------------------------------------------------------------
# Test: Singleton lifecycle
# ---------------------------------------------------------------------------
class TestAgentRunServiceSingleton(unittest.TestCase):
    """Tests for get_agent_run_service / set_agent_run_service singleton management."""

    def setUp(self):
        reset_db_engine()
        set_agent_run_service(None)

    def tearDown(self):
        reset_db_engine()
        set_agent_run_service(None)

    def test_singleton_returns_same_instance(self):
        svc1 = get_agent_run_service()
        svc2 = get_agent_run_service()
        self.assertIs(svc1, svc2)

    def test_singleton_reset_creates_new_instance(self):
        svc1 = get_agent_run_service()
        set_agent_run_service(None)
        svc2 = get_agent_run_service()
        self.assertIsNot(svc1, svc2)

    def test_set_custom_service(self):
        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=engine)
        factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        custom_svc = AgentRunService(session_factory=factory)
        set_agent_run_service(custom_svc)
        self.assertIs(get_agent_run_service(), custom_svc)
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


if __name__ == "__main__":
    unittest.main()
