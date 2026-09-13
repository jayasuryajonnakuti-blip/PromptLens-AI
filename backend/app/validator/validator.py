"""Core PromptValidator orchestrator for PromptLens AI (Step 18).

Coordinates deterministic checks, comparative pipeline evidence, Step 17 Critic
consistency, and optional local LLM semantic reasoning to make the final acceptance
decision before any downstream execution.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from app.llm.errors import LLMDisabledError, ModelNotDownloadedError, RetryExhaustedError
from app.llm.schemas import LLMHealthStatus
from app.validator.checks import run_deterministic_validation
from app.validator.config import VALIDATOR_CONTEXT_CHAR_LIMIT, VALIDATOR_VERSION
from app.validator.context import ValidatorContext, build_validator_context
from app.validator.evidence import (
    build_llm_validation_evidence,
    build_validation_evidence,
)
from app.validator.prompts import (
    PROMPTLENS_VALIDATOR_V1,
    build_validator_correction_prompt,
    build_validator_input,
)
from app.validator.schemas import (
    CriticResultInput,
    IssueSeverity,
    OptimizationMetadataInput,
    ValidationEvidence,
    ValidationIssue,
    ValidationMetadata,
    ValidationMode,
    ValidationResult,
    ValidatorDecision,
)

LOGGER = logging.getLogger(__name__)


class PromptValidator:
    """Independent acceptance gate validating optimized prompts."""

    def validate(
        self,
        original_prompt: str,
        optimized_prompt: str,
        critic_result: CriticResultInput | None = None,
        optimization_metadata: OptimizationMetadataInput | None = None,
    ) -> ValidationResult:
        """Run complete validation pipeline on original vs optimized prompt."""
        if not original_prompt or not original_prompt.strip():
            raise ValueError("original_prompt must not be empty or whitespace-only")
        if not optimized_prompt or not optimized_prompt.strip():
            raise ValueError("optimized_prompt must not be empty or whitespace-only")

        t0 = time.perf_counter()

        # 1. Build validation context
        ctx = build_validator_context(
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt,
            critic_result=critic_result,
        )

        # 2. Run deterministic checks FIRST
        det = run_deterministic_validation(ctx)

        # 3. Assemble baseline evidence
        evidence = build_validation_evidence(ctx, det.issues)

        # 4. Attempt Local LLM validation if available
        llm_output: dict[str, Any] | None = None
        validation_mode = ValidationMode.UNAVAILABLE
        llm_model_name = "none"

        try:
            from app.services.llm_service import get_llm_service

            svc = get_llm_service()
            health = svc.health()

            if health == LLMHealthStatus.MODEL_AVAILABLE:
                context_dict = ctx.to_compact_dict()
                det_issues_dict = [
                    {
                        "code": issue.code,
                        "severity": issue.severity.value,
                        "message": issue.message,
                        "evidence": issue.evidence,
                    }
                    for issue in det.issues
                ]
                user_message = build_validator_input(
                    original_prompt=original_prompt,
                    optimized_prompt=optimized_prompt,
                    context_dict=context_dict,
                    deterministic_issues=det_issues_dict,
                    char_limit=VALIDATOR_CONTEXT_CHAR_LIMIT,
                )
                llm_output, llm_model_name = self._run_llm_validation(svc, user_message)

                meta = svc.provider.model_metadata()
                provider_name = meta.get("provider", "")
                validation_mode = (
                    ValidationMode.MOCK
                    if provider_name == "mock"
                    else ValidationMode.LOCAL_LLM
                )
            else:
                LOGGER.info("LLM health is %s — using UNAVAILABLE validation mode.", health)
        except (LLMDisabledError, ModelNotDownloadedError):
            LOGGER.info("LLM unavailable for validator — using UNAVAILABLE mode.")
        except RetryExhaustedError as exc:
            LOGGER.warning("Validator LLM retries exhausted: %s — UNAVAILABLE mode.", exc)
        except Exception as exc:
            LOGGER.warning("Unexpected LLM error in validator: %s — UNAVAILABLE mode.", exc)

        # 5. Synthesize final decision
        if llm_output is not None:
            result = self._build_from_llm(
                llm=llm_output,
                ctx=ctx,
                det=det,
                evidence=evidence,
            )
        else:
            result = self._build_unavailable(
                ctx=ctx,
                det=det,
                evidence=evidence,
            )

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        result.metadata = ValidationMetadata(
            validator_version=VALIDATOR_VERSION,
            validation_mode=validation_mode,
            model=llm_model_name,
            latency_ms=latency_ms,
            critic_consistency_checked=ctx.critic_provided,
        )

        return result

    # -----------------------------------------------------------------------
    # LLM execution & extraction
    # -----------------------------------------------------------------------
    def _run_llm_validation(self, svc: Any, user_message: str) -> tuple[dict[str, Any], str]:
        config = svc.config
        max_attempts = 1 + config.max_retries
        current_prompt = user_message
        last_error = ""

        for attempt in range(1, max_attempts + 1):
            try:
                raw = svc.provider.generate(
                    prompt=current_prompt,
                    system_prompt=PROMPTLENS_VALIDATOR_V1,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )
            except Exception as exc:
                LOGGER.error("Validator LLM generate() failed attempt %d: %s", attempt, exc)
                raise

            if not raw or not raw.strip():
                last_error = "Model generated empty response."
                if attempt < max_attempts:
                    current_prompt = build_validator_correction_prompt(raw, last_error)
                    continue
                raise RetryExhaustedError("Validator: empty response after retries.")

            try:
                parsed = self._extract_json(raw)
            except ValueError as exc:
                last_error = str(exc)
                LOGGER.warning("Validator JSON extraction failed attempt %d: %s", attempt, exc)
                if attempt < max_attempts:
                    current_prompt = build_validator_correction_prompt(raw, last_error)
                    continue
                raise RetryExhaustedError(
                    f"Validator: JSON extraction exhausted: {last_error}"
                ) from exc

            try:
                self._validate_llm_dict(parsed)
                meta = svc.provider.model_metadata()
                return parsed, meta.get("name", "unknown")
            except ValueError as exc:
                last_error = str(exc)
                LOGGER.warning("Validator schema validation failed attempt %d: %s", attempt, exc)
                if attempt < max_attempts:
                    current_prompt = build_validator_correction_prompt(raw, last_error)
                    continue
                raise RetryExhaustedError(
                    f"Validator: schema validation exhausted: {last_error}"
                ) from exc

        raise RetryExhaustedError(f"Validator retries exhausted: {last_error}")

    def _extract_json(self, raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if fence:
            candidate = fence.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in markdown fences: {exc}") from exc

        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON object: {exc}") from exc

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed embedded JSON: {exc}") from exc

        raise ValueError(f"No JSON found in response: '{text[:100]}...'")

    def _validate_llm_dict(self, data: dict[str, Any]) -> None:
        if "decision" not in data:
            raise ValueError("Missing required key: 'decision'")
        decision = str(data["decision"]).upper()
        if decision not in {"PASS", "FAIL", "NEEDS_REVIEW"}:
            raise ValueError(f"Invalid decision: '{decision}'")

        score = data.get("safety_score")
        if score is None or not (0.0 <= float(score) <= 100.0):
            raise ValueError(f"safety_score must be in [0, 100], got {score!r}")

    # -----------------------------------------------------------------------
    # Decision Arbitration & Synthesis
    # -----------------------------------------------------------------------
    def _build_from_llm(
        self,
        llm: dict[str, Any],
        ctx: ValidatorContext,
        det: Any,
        evidence: list[ValidationEvidence],
    ) -> ValidationResult:
        llm_decision_str = str(llm.get("decision", "NEEDS_REVIEW")).upper()
        try:
            llm_decision = ValidatorDecision(llm_decision_str)
        except ValueError:
            llm_decision = ValidatorDecision.NEEDS_REVIEW

        # Decision Arbitration:
        # Deterministic fatal checks MUST override LLM PASS
        if det.decision == ValidatorDecision.FAIL:
            final_decision = ValidatorDecision.FAIL
        elif llm_decision == ValidatorDecision.FAIL:
            final_decision = ValidatorDecision.FAIL
        elif det.decision == ValidatorDecision.NEEDS_REVIEW or llm_decision == ValidatorDecision.NEEDS_REVIEW:
            final_decision = ValidatorDecision.NEEDS_REVIEW
        else:
            final_decision = ValidatorDecision.PASS

        # Combine issues
        all_issues = list(det.issues)
        for issue_data in llm.get("issues", []):
            if isinstance(issue_data, dict) and issue_data.get("message"):
                sev_str = str(issue_data.get("severity", "WARNING")).upper()
                sev = IssueSeverity.WARNING
                if sev_str in {"ERROR", "WARNING", "INFO"}:
                    sev = IssueSeverity(sev_str)
                all_issues.append(
                    ValidationIssue(
                        code=str(issue_data.get("code", "LLM_ISSUE")),
                        severity=sev,
                        message=str(issue_data.get("message", "")),
                        field_or_scope=str(issue_data.get("field_or_scope", "semantic")),
                        evidence=str(issue_data.get("evidence", "")),
                    )
                )

        # Combine passed and failed checks
        passed = list(dict.fromkeys(det.passed_checks + [str(c) for c in llm.get("passed_checks", []) if c]))
        failed = list(dict.fromkeys(det.failed_checks + [str(c) for c in llm.get("failed_checks", []) if c]))

        # Append LLM evidence
        all_evidence = evidence + build_llm_validation_evidence(llm)

        # Safety score synthesis
        llm_safety = float(llm.get("safety_score", det.safety_score))
        combined_safety = min(det.safety_score, llm_safety)
        if final_decision == ValidatorDecision.FAIL:
            combined_safety = min(combined_safety, 45.0)

        return ValidationResult(
            decision=final_decision,
            is_valid=(final_decision == ValidatorDecision.PASS),
            safety_score=round(combined_safety, 1),
            original_metadata=ctx.to_original_metadata(),
            optimized_metadata=ctx.to_optimized_metadata(),
            issues=all_issues,
            evidence=all_evidence,
            passed_checks=passed,
            failed_checks=failed,
            metadata=ValidationMetadata(
                validator_version=VALIDATOR_VERSION,
                validation_mode=ValidationMode.UNAVAILABLE,
                model="none",
                latency_ms=0.0,
                critic_consistency_checked=ctx.critic_provided,
            ),
        )

    def _build_unavailable(
        self,
        ctx: ValidatorContext,
        det: Any,
        evidence: list[ValidationEvidence],
    ) -> ValidationResult:
        """Construct a complete, sound validation result purely from deterministic checks."""
        return ValidationResult(
            decision=det.decision,
            is_valid=(det.decision == ValidatorDecision.PASS),
            safety_score=det.safety_score,
            original_metadata=ctx.to_original_metadata(),
            optimized_metadata=ctx.to_optimized_metadata(),
            issues=det.issues,
            evidence=evidence,
            passed_checks=det.passed_checks,
            failed_checks=det.failed_checks,
            metadata=ValidationMetadata(
                validator_version=VALIDATOR_VERSION,
                validation_mode=ValidationMode.UNAVAILABLE,
                model="none",
                latency_ms=0.0,
                critic_consistency_checked=ctx.critic_provided,
            ),
        )
