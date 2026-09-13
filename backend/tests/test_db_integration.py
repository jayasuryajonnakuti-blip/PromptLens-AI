"""DB integration tests for Step 20 — end-to-end persistence through AgentService.

Tests:
- AgentService.run() calls save_agent_run() and sets run_id + persistence_status on response.
- A completed (VALIDATED) agent run is written to DB and retrievable.
- A NEEDS_REVIEW agent run is persisted with is_validated=False.
- A FAILED agent run is persisted with appropriate status.
- DB persistence failure sets persistence_status='failed' without breaking AgentResponse.
- Multiple runs accumulate correctly in DB.
- Singleton isolation: each test class resets AgentRunService and DB engine.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
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
from app.agent.orchestrator import AgentOrchestrator
from app.db.base import Base
from app.db.session import reset_db_engine
from app.optimizer.schemas import OptimizationMode
from app.services.agent_run_service import AgentRunService, set_agent_run_service
from app.services.agent_service import AgentService, set_agent_service
from app.validator.schemas import (
    PromptMetadata,
    ValidationMetadata,
    ValidationMode,
    ValidationResult,
    ValidatorDecision,
)


# ---------------------------------------------------------------------------
# Helpers (shared with test_agent_run_service.py style, duplicated for isolation)
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


def _make_validation_result(decision: ValidatorDecision = ValidatorDecision.PASS) -> ValidationResult:
    meta = _make_prompt_metadata()
    return ValidationResult(
        decision=decision,
        is_valid=(decision == ValidatorDecision.PASS),
        safety_score=85.0,
        original_metadata=meta,
        optimized_metadata=meta,
        issues=[],
        metadata=ValidationMetadata(
            validator_version="1.0.0",
            validation_mode=ValidationMode.MOCK,
            model="none",
            latency_ms=10.0,
            critic_consistency_checked=False,
        ),
    )


def _make_optimizer_result() -> OptimizerResultSummary:
    return OptimizerResultSummary(
        optimized_prompt="Improved prompt text.",
        summary="Clarity improvements applied.",
        changes=[{"category": "clarity", "description": "Reworded."}],
        preserved_requirements=[],
        placeholders_inserted=[],
        improvement_score_delta=5.0,
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


def _make_agent_iteration() -> AgentIteration:
    return AgentIteration(
        iteration_number=1,
        prompt_before="Original prompt text.",
        prompt_after="Improved prompt text.",
        optimizer_result=_make_optimizer_result(),
        critic_result=_make_critic_result("PASS"),
        validator_result=_make_validation_result(ValidatorDecision.PASS),
        score_before=70.0,
        score_after=85.0,
        decision="PASS",
        latency_ms=100.0,
    )


def _make_completed_response() -> AgentResponse:
    return AgentResponse(
        original_prompt="Original prompt text.",
        final_prompt="Improved prompt text.",
        status=AgentStatus.COMPLETED,
        termination_reason=AgentTerminationReason.VALIDATED,
        iteration_count=1,
        is_validated=True,
        iterations=[_make_agent_iteration()],
        final_critic_result=_make_critic_result("PASS"),
        final_validation=_make_validation_result(ValidatorDecision.PASS),
        final_score=85.0,
        metrics=AgentMetrics(
            total_latency_ms=100.0,
            total_iterations=1,
            initial_score=70.0,
            final_score=85.0,
            score_delta=15.0,
            prompt_length_before=22,
            prompt_length_after=22,
            expansion_ratio=1.0,
            semantic_similarity=0.95,
        ),
    )


def _make_needs_review_response() -> AgentResponse:
    return AgentResponse(
        original_prompt="Original prompt text.",
        final_prompt="Improved prompt text.",
        status=AgentStatus.NEEDS_REVIEW,
        termination_reason=AgentTerminationReason.VALIDATION_REVIEW,
        iteration_count=1,
        is_validated=False,
        iterations=[_make_agent_iteration()],
        final_critic_result=_make_critic_result("NEEDS_REVIEW"),
        final_validation=_make_validation_result(ValidatorDecision.NEEDS_REVIEW),
        final_score=72.0,
        metrics=AgentMetrics(
            total_latency_ms=100.0,
            total_iterations=1,
            initial_score=68.0,
            final_score=72.0,
            score_delta=4.0,
            prompt_length_before=22,
            prompt_length_after=22,
            expansion_ratio=1.0,
            semantic_similarity=0.90,
        ),
    )


def _make_failed_response() -> AgentResponse:
    return AgentResponse(
        original_prompt="Original prompt text.",
        final_prompt="Original prompt text.",
        status=AgentStatus.FAILED,
        termination_reason=AgentTerminationReason.VALIDATION_FAILED,
        iteration_count=0,
        is_validated=False,
        iterations=[],
        final_critic_result=None,
        final_validation=None,
        final_score=None,
        metrics=AgentMetrics(
            total_latency_ms=0.0,
            total_iterations=0,
            prompt_length_before=22,
            prompt_length_after=22,
        ),
    )


def _make_isolated_run_service():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    svc = AgentRunService(session_factory=factory)
    return svc, engine, factory


# ---------------------------------------------------------------------------
# Test: AgentService integration with persistence
# ---------------------------------------------------------------------------
class TestAgentServicePersistenceIntegration(unittest.TestCase):
    """Integration tests verifying AgentService wires into AgentRunService."""

    def setUp(self):
        reset_db_engine()
        set_agent_service(None)
        set_agent_run_service(None)
        self.run_svc, self.engine, self.factory = _make_isolated_run_service()
        set_agent_run_service(self.run_svc)

    def tearDown(self):
        set_agent_service(None)
        set_agent_run_service(None)
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()
        reset_db_engine()

    def _make_agent_service_with_mock(self, response: AgentResponse) -> AgentService:
        mock_orch = MagicMock(spec=AgentOrchestrator)
        mock_orch.run.return_value = response
        return AgentService(orchestrator=mock_orch)

    def test_completed_run_sets_persistence_status_persisted(self):
        response = _make_completed_response()
        svc = self._make_agent_service_with_mock(response)
        from app.agent.schemas import AgentRequest
        req = AgentRequest(prompt="Original prompt text.", mode=OptimizationMode.BALANCED)
        result = svc.run(req)
        self.assertEqual(result.persistence_status, "persisted")

    def test_completed_run_sets_run_id(self):
        response = _make_completed_response()
        svc = self._make_agent_service_with_mock(response)
        from app.agent.schemas import AgentRequest
        req = AgentRequest(prompt="Original prompt text.", mode=OptimizationMode.BALANCED)
        result = svc.run(req)
        self.assertIsNotNone(result.run_id)
        self.assertIsInstance(result.run_id, str)

    def test_completed_run_is_retrievable_from_db(self):
        response = _make_completed_response()
        svc = self._make_agent_service_with_mock(response)
        from app.agent.schemas import AgentRequest
        req = AgentRequest(prompt="Original prompt text.", mode=OptimizationMode.BALANCED)
        result = svc.run(req)
        run_id = result.run_id
        db_run = self.run_svc.get_agent_run(run_id)
        self.assertIsNotNone(db_run)
        self.assertEqual(db_run.status, "COMPLETED")
        self.assertTrue(db_run.is_validated)

    def test_needs_review_run_persisted_with_not_validated(self):
        response = _make_needs_review_response()
        svc = self._make_agent_service_with_mock(response)
        from app.agent.schemas import AgentRequest
        req = AgentRequest(prompt="Original prompt text.", mode=OptimizationMode.ANALYTICAL)
        result = svc.run(req)
        self.assertEqual(result.persistence_status, "persisted")
        db_run = self.run_svc.get_agent_run(result.run_id)
        self.assertIsNotNone(db_run)
        self.assertFalse(db_run.is_validated)
        self.assertEqual(db_run.status, "NEEDS_REVIEW")

    def test_failed_run_persisted(self):
        response = _make_failed_response()
        svc = self._make_agent_service_with_mock(response)
        from app.agent.schemas import AgentRequest
        req = AgentRequest(prompt="Original prompt text.", mode=OptimizationMode.CREATIVE)
        result = svc.run(req)
        self.assertEqual(result.persistence_status, "persisted")
        db_run = self.run_svc.get_agent_run(result.run_id)
        self.assertIsNotNone(db_run)
        self.assertEqual(db_run.status, "FAILED")
        self.assertFalse(db_run.is_validated)
        self.assertIsNone(db_run.final_score)

    def test_multiple_runs_accumulate_in_db(self):
        for _ in range(3):
            response = _make_completed_response()
            svc = self._make_agent_service_with_mock(response)
            from app.agent.schemas import AgentRequest
            req = AgentRequest(prompt="Original prompt text.")
            svc.run(req)
        count = self.run_svc.count_agent_runs()
        self.assertEqual(count, 3)

    def test_mode_stored_correctly_for_analytical(self):
        response = _make_completed_response()
        svc = self._make_agent_service_with_mock(response)
        from app.agent.schemas import AgentRequest
        req = AgentRequest(prompt="Original prompt text.", mode=OptimizationMode.ANALYTICAL)
        result = svc.run(req)
        db_run = self.run_svc.get_agent_run(result.run_id)
        self.assertEqual(db_run.mode, "analytical")

    def test_mode_stored_correctly_for_expert(self):
        response = _make_completed_response()
        svc = self._make_agent_service_with_mock(response)
        from app.agent.schemas import AgentRequest
        req = AgentRequest(prompt="Original prompt text.", mode=OptimizationMode.EXPERT)
        result = svc.run(req)
        db_run = self.run_svc.get_agent_run(result.run_id)
        self.assertEqual(db_run.mode, "expert")


# ---------------------------------------------------------------------------
# Test: Persistence failure handling
# ---------------------------------------------------------------------------
class TestPersistenceFailureHandling(unittest.TestCase):
    """Tests verifying persistence failure does not crash AgentService.run()."""

    def setUp(self):
        reset_db_engine()
        set_agent_service(None)
        set_agent_run_service(None)

    def tearDown(self):
        set_agent_service(None)
        set_agent_run_service(None)
        reset_db_engine()

    def test_db_failure_sets_persistence_status_failed(self):
        """When AgentRunService.save_agent_run() raises, response.persistence_status = 'failed'."""
        # Inject broken AgentRunService
        broken_svc = MagicMock(spec=AgentRunService)
        broken_svc.save_agent_run.side_effect = RuntimeError("DB connection failed")
        set_agent_run_service(broken_svc)

        response = _make_completed_response()
        mock_orch = MagicMock(spec=AgentOrchestrator)
        mock_orch.run.return_value = response
        svc = AgentService(orchestrator=mock_orch)

        from app.agent.schemas import AgentRequest
        req = AgentRequest(prompt="Original prompt text.")
        result = svc.run(req)

        self.assertEqual(result.persistence_status, "failed")

    def test_db_failure_does_not_affect_agent_result(self):
        """When persistence fails, agent result fields remain intact."""
        broken_svc = MagicMock(spec=AgentRunService)
        broken_svc.save_agent_run.side_effect = RuntimeError("DB connection failed")
        set_agent_run_service(broken_svc)

        response = _make_completed_response()
        mock_orch = MagicMock(spec=AgentOrchestrator)
        mock_orch.run.return_value = response
        svc = AgentService(orchestrator=mock_orch)

        from app.agent.schemas import AgentRequest
        req = AgentRequest(prompt="Original prompt text.")
        result = svc.run(req)

        # Core agent result fields unaffected by persistence failure
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(result.termination_reason, AgentTerminationReason.VALIDATED)
        self.assertTrue(result.is_validated)
        self.assertAlmostEqual(result.final_score, 85.0)

    def test_db_failure_does_not_set_run_id(self):
        """When persistence fails, run_id remains None."""
        broken_svc = MagicMock(spec=AgentRunService)
        broken_svc.save_agent_run.side_effect = RuntimeError("DB connection failed")
        set_agent_run_service(broken_svc)

        response = _make_completed_response()
        mock_orch = MagicMock(spec=AgentOrchestrator)
        mock_orch.run.return_value = response
        svc = AgentService(orchestrator=mock_orch)

        from app.agent.schemas import AgentRequest
        req = AgentRequest(prompt="Original prompt text.")
        result = svc.run(req)

        self.assertIsNone(result.run_id)


if __name__ == "__main__":
    unittest.main()
