"""Core PromptCritic orchestrator for PromptLens AI (Step 17).

Evaluates whether an optimized prompt represents genuine improvement over the original.
Combines:
1. Deterministic structural, format, constraint, and regression checks
2. Upstream pipeline signals (intent, embeddings similarity, Step 13 12 dimensions)
3. Local LLM deep reasoning (when available)
4. Graceful UNAVAILABLE mode fallback when LLM is offline
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from app.critic.checks import run_deterministic_checks
from app.critic.config import CRITIC_CONTEXT_CHAR_LIMIT, CRITIC_VERSION
from app.critic.context import CriticContext, build_critic_context
from app.critic.evidence import build_deterministic_evidence, build_llm_evidence
from app.critic.prompts import (
    PROMPTLENS_CRITIC_V1,
    build_critic_correction_prompt,
    build_critic_input,
)
from app.critic.schemas import (
    CriticAnalysisMode,
    CriticDecision,
    CriticEvaluation,
    CriticIssue,
    CriticMetadata,
    IntentPreservationResult,
    IssueSeverity,
    MetricChange,
    OptimizationMetadataInput,
    QualityStatus,
    RequirementPreservationResult,
)
from app.llm.errors import LLMDisabledError, ModelNotDownloadedError, RetryExhaustedError
from app.llm.schemas import LLMHealthStatus

LOGGER = logging.getLogger(__name__)


class PromptCritic:
    """Evaluates optimization quality between original and optimized prompts."""

    def evaluate(
        self,
        original_prompt: str,
        optimized_prompt: str,
        optimization_metadata: OptimizationMetadataInput | None = None,
    ) -> CriticEvaluation:
        """Run comprehensive critique evaluating the optimized prompt."""
        if not original_prompt or not original_prompt.strip():
            raise ValueError("original_prompt must not be empty or whitespace-only")
        if not optimized_prompt or not optimized_prompt.strip():
            raise ValueError("optimized_prompt must not be empty or whitespace-only")

        t0 = time.perf_counter()

        # 1. Build comparative pipeline context
        ctx = build_critic_context(original_prompt, optimized_prompt)

        # 2. Run deterministic checks
        det = run_deterministic_checks(ctx, original_prompt, optimized_prompt)

        # 3. Build deterministic evidence items
        evidence = build_deterministic_evidence(ctx, original_prompt, optimized_prompt, det.issues)

        # 4. Attempt LLM Critique
        llm_output: dict[str, Any] | None = None
        analysis_mode = CriticAnalysisMode.UNAVAILABLE
        llm_model_name = "none"

        try:
            from app.services.llm_service import get_llm_service

            svc = get_llm_service()
            health = svc.health()

            if health == LLMHealthStatus.MODEL_AVAILABLE:
                context_dict = ctx.to_compact_dict()
                det_issues_dict = [
                    {
                        "type": issue.type,
                        "severity": issue.severity.value,
                        "description": issue.description,
                        "evidence": issue.evidence,
                    }
                    for issue in det.issues
                ]
                user_message = build_critic_input(
                    original_prompt=original_prompt,
                    optimized_prompt=optimized_prompt,
                    context_dict=context_dict,
                    deterministic_issues=det_issues_dict,
                    char_limit=CRITIC_CONTEXT_CHAR_LIMIT,
                )
                llm_output, llm_model_name = self._run_llm_critique(svc, user_message)

                meta = svc.provider.model_metadata()
                provider_name = meta.get("provider", "")
                if provider_name == "mock":
                    analysis_mode = CriticAnalysisMode.MOCK
                else:
                    analysis_mode = CriticAnalysisMode.LOCAL_LLM
            else:
                LOGGER.info("LLM health is %s — using UNAVAILABLE mode for critic.", health)
        except (LLMDisabledError, ModelNotDownloadedError):
            LOGGER.info("LLM unavailable for critic — using UNAVAILABLE mode.")
        except RetryExhaustedError as exc:
            LOGGER.warning("Critic LLM retries exhausted: %s — UNAVAILABLE mode.", exc)
        except Exception as exc:
            LOGGER.warning("Unexpected LLM error in critic: %s — UNAVAILABLE mode.", exc)

        # 5. Synthesize evaluation
        if llm_output is not None:
            evaluation = self._build_from_llm(
                llm=llm_output,
                ctx=ctx,
                det=det,
                evidence=evidence,
            )
        else:
            evaluation = self._build_unavailable(
                ctx=ctx,
                det=det,
                evidence=evidence,
            )

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        evaluation.metadata = CriticMetadata(
            critic_version=CRITIC_VERSION,
            analysis_mode=analysis_mode,
            model=llm_model_name,
            semantic_similarity=round(ctx.semantic_similarity, 4),
            latency_ms=latency_ms,
        )

        return evaluation

    # -----------------------------------------------------------------------
    # LLM execution & extraction
    # -----------------------------------------------------------------------
    def _run_llm_critique(self, svc: Any, user_message: str) -> tuple[dict[str, Any], str]:
        config = svc.config
        max_attempts = 1 + config.max_retries
        current_prompt = user_message

        last_error = ""

        for attempt in range(1, max_attempts + 1):
            try:
                raw = svc.provider.generate(
                    prompt=current_prompt,
                    system_prompt=PROMPTLENS_CRITIC_V1,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )
            except Exception as exc:
                LOGGER.error("Critic LLM generate() failed attempt %d: %s", attempt, exc)
                raise

            if not raw or not raw.strip():
                last_error = "Model generated empty response."
                if attempt < max_attempts:
                    current_prompt = build_critic_correction_prompt(raw, last_error)
                    continue
                raise RetryExhaustedError("Critic: empty generation after retries.")

            try:
                parsed = self._extract_json(raw)
            except ValueError as exc:
                last_error = str(exc)
                LOGGER.warning("Critic JSON extraction failed attempt %d: %s", attempt, exc)
                if attempt < max_attempts:
                    current_prompt = build_critic_correction_prompt(raw, last_error)
                    continue
                raise RetryExhaustedError(
                    f"Critic: JSON extraction exhausted: {last_error}"
                ) from exc

            try:
                self._validate_llm_dict(parsed)
                meta = svc.provider.model_metadata()
                return parsed, meta.get("name", "unknown")
            except ValueError as exc:
                last_error = str(exc)
                LOGGER.warning("Critic schema validation failed attempt %d: %s", attempt, exc)
                if attempt < max_attempts:
                    current_prompt = build_critic_correction_prompt(raw, last_error)
                    continue
                raise RetryExhaustedError(
                    f"Critic: schema validation exhausted: {last_error}"
                ) from exc

        raise RetryExhaustedError(f"Critic retries exhausted: {last_error}")

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
        required = ["decision", "overall_critique_score", "intent_preservation", "requirement_preservation"]
        for key in required:
            if key not in data:
                raise ValueError(f"Missing required key: '{key}'")

        decision = str(data.get("decision", "")).upper()
        if decision not in {"PASS", "FAIL", "NEEDS_REVIEW"}:
            raise ValueError(f"Invalid decision: '{decision}'")

        score = data.get("overall_critique_score")
        if score is None or not (0.0 <= float(score) <= 100.0):
            raise ValueError(f"overall_critique_score must be in [0, 100], got {score!r}")

        for field in ("intent_preservation", "requirement_preservation"):
            sub = data.get(field, {})
            if not isinstance(sub, dict):
                raise ValueError(f"'{field}' must be a dictionary")
            sub_score = sub.get("score")
            if sub_score is None or not (0.0 <= float(sub_score) <= 100.0):
                raise ValueError(f"{field}.score must be in [0, 100], got {sub_score!r}")

    # -----------------------------------------------------------------------
    # Synthesizers
    # -----------------------------------------------------------------------
    def _build_from_llm(
        self,
        llm: dict[str, Any],
        ctx: CriticContext,
        det: Any,
        evidence: list,
    ) -> CriticEvaluation:
        # LLM suggested decision
        llm_decision_str = str(llm.get("decision", "NEEDS_REVIEW")).upper()
        try:
            llm_decision = CriticDecision(llm_decision_str)
        except ValueError:
            llm_decision = CriticDecision.NEEDS_REVIEW

        # Controlled decision arbitration:
        # Deterministic FAIL overrides LLM PASS (e.g. fatal format lost, severe regression)
        if det.suggested_decision == CriticDecision.FAIL:
            final_decision = CriticDecision.FAIL
        elif llm_decision == CriticDecision.FAIL:
            final_decision = CriticDecision.FAIL
        elif (
            det.suggested_decision == CriticDecision.NEEDS_REVIEW
            or llm_decision == CriticDecision.NEEDS_REVIEW
        ):
            final_decision = CriticDecision.NEEDS_REVIEW
        else:
            final_decision = CriticDecision.PASS

        # Combine deterministic & LLM issues
        all_issues = list(det.issues)
        for issue_dict in llm.get("issues", []):
            if isinstance(issue_dict, dict) and issue_dict.get("description"):
                severity_str = str(issue_dict.get("severity", "WARNING")).upper()
                sev = IssueSeverity.WARNING
                if severity_str in {"ERROR", "WARNING", "INFO"}:
                    sev = IssueSeverity(severity_str)
                all_issues.append(
                    CriticIssue(
                        type=str(issue_dict.get("type", "llm_issue")),
                        severity=sev,
                        description=str(issue_dict.get("description", "")),
                        evidence=str(issue_dict.get("evidence", "")),
                    )
                )

        # Append LLM inference evidence
        all_evidence = evidence + build_llm_evidence(llm)

        # Merge requirements
        lost_reqs = list(dict.fromkeys(det.lost_requirements + [str(r) for r in llm.get("lost_requirements", []) if r]))
        intro_reqs = list(dict.fromkeys(det.introduced_requirements + [str(r) for r in llm.get("introduced_requirements", []) if r]))
        unsupported = list(dict.fromkeys(det.unsupported_assumptions + [str(a) for a in llm.get("unsupported_assumptions", []) if a]))
        preserved = list(dict.fromkeys([str(p) for p in llm.get("preserved_requirements", []) if p]))

        # Intent preservation
        ip_data = llm.get("intent_preservation", {})
        ip_score = float(ip_data.get("score", det.intent_preservation_score))
        ip_status_str = str(ip_data.get("status", "GOOD")).upper()
        ip_status = QualityStatus.GOOD
        if ip_status_str in QualityStatus.__members__:
            ip_status = QualityStatus(ip_status_str)
        intent_res = IntentPreservationResult(
            score=ip_score,
            status=ip_status,
            reason=str(ip_data.get("reason", "Evaluated by local model")),
        )

        # Requirement preservation
        rp_data = llm.get("requirement_preservation", {})
        rp_score = float(rp_data.get("score", det.requirement_preservation_score))
        rp_status_str = str(rp_data.get("status", "GOOD")).upper()
        rp_status = QualityStatus.GOOD
        if rp_status_str in QualityStatus.__members__:
            rp_status = QualityStatus(rp_status_str)
        req_res = RequirementPreservationResult(
            score=rp_score,
            status=rp_status,
            reason=str(rp_data.get("reason", "Evaluated by local model")),
        )

        # Metric changes
        clarity_change = MetricChange(
            original_score=ctx.clarity_original,
            optimized_score=ctx.clarity_optimized,
            delta=ctx.clarity_delta,
            assessment=str(llm.get("clarity_assessment", f"Clarity changed by {ctx.clarity_delta:+.1f}")),
        )
        specificity_change = MetricChange(
            original_score=ctx.specificity_original,
            optimized_score=ctx.specificity_optimized,
            delta=ctx.specificity_delta,
            assessment=str(llm.get("specificity_assessment", f"Specificity changed by {ctx.specificity_delta:+.1f}")),
        )
        ambiguity_change = MetricChange(
            original_score=ctx.ambiguity_original,
            optimized_score=ctx.ambiguity_optimized,
            delta=ctx.ambiguity_delta,
            assessment=str(llm.get("ambiguity_assessment", f"Ambiguity changed by {ctx.ambiguity_delta:+.1f}")),
        )
        completeness_change = MetricChange(
            original_score=ctx.completeness_original,
            optimized_score=ctx.completeness_optimized,
            delta=ctx.completeness_delta,
            assessment=str(llm.get("completeness_assessment", f"Completeness changed by {ctx.completeness_delta:+.1f}")),
        )

        # Overall critique score: balance LLM score with deterministic penalties
        overall_score = float(llm.get("overall_critique_score", det.base_critique_score))
        if final_decision == CriticDecision.FAIL:
            overall_score = min(overall_score, 45.0)

        return CriticEvaluation(
            decision=final_decision,
            overall_critique_score=round(overall_score, 1),
            intent_preservation=intent_res,
            requirement_preservation=req_res,
            clarity_change=clarity_change,
            specificity_change=specificity_change,
            ambiguity_change=ambiguity_change,
            completeness_change=completeness_change,
            issues=all_issues,
            preserved_requirements=preserved,
            lost_requirements=lost_reqs,
            introduced_requirements=intro_reqs,
            unsupported_assumptions=unsupported,
            strengths=[str(s) for s in llm.get("strengths", []) if s],
            weaknesses=[str(w) for w in llm.get("weaknesses", []) if w],
            recommendations=[str(r) for r in llm.get("recommendations", []) if r],
            evidence=all_evidence,
            metadata=CriticMetadata(
                critic_version=CRITIC_VERSION,
                analysis_mode=CriticAnalysisMode.UNAVAILABLE,
                model="unknown",
                semantic_similarity=round(ctx.semantic_similarity, 4),
            ),
        )

    def _build_unavailable(
        self,
        ctx: CriticContext,
        det: Any,
        evidence: list,
    ) -> CriticEvaluation:
        """Construct a rigorous evaluation based entirely on deterministic pipeline evidence."""
        # Intent preservation status
        ip_score = det.intent_preservation_score
        if ip_score >= 85:
            ip_status = QualityStatus.EXCELLENT
        elif ip_score >= 70:
            ip_status = QualityStatus.STRONG
        elif ip_score >= 55:
            ip_status = QualityStatus.GOOD
        elif ip_score >= 40:
            ip_status = QualityStatus.FAIR
        else:
            ip_status = QualityStatus.POOR

        intent_reason = (
            f"Evaluated via Step 10 classifier ({ctx.intent_original} vs {ctx.intent_optimized}) "
            f"and semantic embeddings (similarity: {ctx.semantic_similarity:.3f})."
        )

        # Requirement preservation status
        rp_score = det.requirement_preservation_score
        if rp_score >= 85:
            rp_status = QualityStatus.EXCELLENT
        elif rp_score >= 70:
            rp_status = QualityStatus.STRONG
        elif rp_score >= 55:
            rp_status = QualityStatus.GOOD
        elif rp_score >= 40:
            rp_status = QualityStatus.FAIR
        else:
            rp_status = QualityStatus.POOR

        req_reason = (
            f"Deterministic checks evaluated format, role, and constraint preservation. "
            f"{len(det.lost_requirements)} lost, {len(det.introduced_requirements)} introduced."
        )

        # Assessments
        def _assess_delta(name: str, delta: float) -> str:
            if delta >= 10:
                return f"{name} significantly improved (+{delta:.1f} pts)."
            if delta > 0:
                return f"{name} slightly improved (+{delta:.1f} pts)."
            if delta == 0:
                return f"{name} is unchanged."
            if delta > -10:
                return f"{name} slightly declined ({delta:.1f} pts)."
            return f"{name} significantly declined ({delta:.1f} pts)."

        strengths: list[str] = []
        weaknesses: list[str] = []
        recs: list[str] = []

        if ctx.overall_score_delta > 0:
            strengths.append(f"Overall prompt quality score improved by {ctx.overall_score_delta:+.1f} points.")
        if ctx.clarity_delta > 0:
            strengths.append(f"Clarity dimension improved by {ctx.clarity_delta:+.1f} points.")
        if ctx.specificity_delta > 0:
            strengths.append(f"Specificity dimension improved by {ctx.specificity_delta:+.1f} points.")

        if ctx.overall_score_delta < 0:
            weaknesses.append(f"Overall prompt quality score regressed by {ctx.overall_score_delta:.1f} points.")
        if det.lost_requirements:
            weaknesses.append(f"Lost requirements: {', '.join(det.lost_requirements)}")
            recs.append("Restore original constraints and formatting requirements.")
        if det.introduced_requirements:
            weaknesses.append(f"Introduced unrequested requirements: {', '.join(det.introduced_requirements)}")
            recs.append("Remove assumptions or replace with generic placeholders (e.g. [FRAMEWORK]).")

        return CriticEvaluation(
            decision=det.suggested_decision,
            overall_critique_score=det.base_critique_score,
            intent_preservation=IntentPreservationResult(
                score=ip_score,
                status=ip_status,
                reason=intent_reason,
            ),
            requirement_preservation=RequirementPreservationResult(
                score=rp_score,
                status=rp_status,
                reason=req_reason,
            ),
            clarity_change=MetricChange(
                original_score=ctx.clarity_original,
                optimized_score=ctx.clarity_optimized,
                delta=ctx.clarity_delta,
                assessment=_assess_delta("Clarity", ctx.clarity_delta),
            ),
            specificity_change=MetricChange(
                original_score=ctx.specificity_original,
                optimized_score=ctx.specificity_optimized,
                delta=ctx.specificity_delta,
                assessment=_assess_delta("Specificity", ctx.specificity_delta),
            ),
            ambiguity_change=MetricChange(
                original_score=ctx.ambiguity_original,
                optimized_score=ctx.ambiguity_optimized,
                delta=ctx.ambiguity_delta,
                assessment=_assess_delta("Ambiguity", ctx.ambiguity_delta),
            ),
            completeness_change=MetricChange(
                original_score=ctx.completeness_original,
                optimized_score=ctx.completeness_optimized,
                delta=ctx.completeness_delta,
                assessment=_assess_delta("Completeness", ctx.completeness_delta),
            ),
            issues=det.issues,
            preserved_requirements=["Original objective preserved"] if det.intent_preservation_score >= 60 else [],
            lost_requirements=det.lost_requirements,
            introduced_requirements=det.introduced_requirements,
            unsupported_assumptions=det.unsupported_assumptions,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recs,
            evidence=evidence,
            metadata=CriticMetadata(
                critic_version=CRITIC_VERSION,
                analysis_mode=CriticAnalysisMode.UNAVAILABLE,
                model="none",
                semantic_similarity=round(ctx.semantic_similarity, 4),
            ),
        )
