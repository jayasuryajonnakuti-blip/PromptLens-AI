"""Core PromptOptimizer class for PromptLens (Step 16).

Orchestrates:
1. Build compact analysis summary from Step 15 AnalyzerAnalysis (or lightweight
   fallback if no analyzer result was provided)
2. Select mode-specific system prompt
3. Invoke Step 14 LLMService with retry/correction loop
4. Validate structured JSON output
5. Fall back to deterministic UNAVAILABLE result if LLM is disabled/failed
6. Return typed OptimizerResult

Key invariants:
- optimized_prompt is NEVER empty — UNAVAILABLE path returns the original prompt
  with an explanatory prefix so the caller always gets a usable value.
- optimizer_mode is ALWAYS accurate (LOCAL_LLM | MOCK | UNAVAILABLE).
- The optimizer NEVER rewrites the prompt when it cannot verify the analysis.
- original intent, domain, scope, and stated constraints are always preserved.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from app.optimizer.config import (
    ANALYSIS_SUMMARY_CHAR_LIMIT,
    MAX_PLACEHOLDERS,
    MAX_RECOMMENDATIONS_IN_CONTEXT,
    OPTIMIZER_VERSION,
)
from app.optimizer.prompts import (
    build_optimizer_correction_prompt,
    build_optimizer_input,
    get_system_prompt,
)
from app.optimizer.schemas import (
    ChangeRecord,
    OptimizationMode,
    OptimizerMetadata,
    OptimizerMode,
    OptimizerResult,
    PreservedRequirement,
)
from app.llm.errors import LLMDisabledError, ModelNotDownloadedError, RetryExhaustedError
from app.llm.schemas import LLMHealthStatus

LOGGER = logging.getLogger(__name__)


class PromptOptimizer:
    """Generates an improved prompt using the Step 14 LLM provider."""

    def optimize(
        self,
        prompt: str,
        mode: OptimizationMode = OptimizationMode.BALANCED,
        analyzer_result: dict[str, Any] | None = None,
    ) -> OptimizerResult:
        """Optimize the given prompt using mode-specific LLM guidance.

        Args:
            prompt: Original prompt text.
            mode: Optimization style (balanced | analytical | creative | expert).
            analyzer_result: Optional serialised AnalyzerAnalysis from Step 15.
                             When provided, gives richer context to the LLM.

        Returns:
            OptimizerResult with optimized_prompt, changes, preserved requirements,
            placeholders, and metadata.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty or whitespace-only")

        t0 = time.perf_counter()

        # 1. Build compact analysis summary for LLM context
        analysis_summary = self._build_analysis_summary(prompt, analyzer_result)

        # 2. Attempt LLM optimization
        llm_output: dict[str, Any] | None = None
        optimizer_mode = OptimizerMode.UNAVAILABLE
        llm_model_name = "none"

        try:
            from app.services.llm_service import get_llm_service

            svc = get_llm_service()
            health = svc.health()

            if health == LLMHealthStatus.MODEL_AVAILABLE:
                llm_output, llm_model_name = self._run_llm_optimization(
                    svc=svc,
                    prompt=prompt,
                    mode=mode.value,
                    analysis_summary=analysis_summary,
                )
                # Determine mode from provider type
                meta = svc.provider.model_metadata()
                provider_name = meta.get("provider", "")
                optimizer_mode = (
                    OptimizerMode.MOCK
                    if provider_name == "mock"
                    else OptimizerMode.LOCAL_LLM
                )
            else:
                LOGGER.info(
                    "LLM health state is %s — falling back to UNAVAILABLE mode.", health
                )

        except (LLMDisabledError, ModelNotDownloadedError):
            LOGGER.info("LLM unavailable for optimizer — using UNAVAILABLE mode.")
        except RetryExhaustedError as exc:
            LOGGER.warning("Optimizer LLM retries exhausted: %s — UNAVAILABLE mode.", exc)
        except Exception as exc:
            LOGGER.warning("Unexpected LLM error in optimizer: %s — UNAVAILABLE mode.", exc)

        # 3. Build result
        if llm_output is not None:
            result = self._build_from_llm_output(llm_output, mode)
        else:
            result = self._build_unavailable_result(prompt, mode, analysis_summary)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # 4. Attach accurate metadata
        result.metadata = OptimizerMetadata(
            optimizer_version=OPTIMIZER_VERSION,
            llm_model=llm_model_name,
            optimizer_mode=optimizer_mode,
            optimization_mode=mode,
            latency_ms=latency_ms,
        )
        return result

    # -----------------------------------------------------------------------
    # Analysis summary builder
    # -----------------------------------------------------------------------
    def _build_analysis_summary(
        self,
        prompt: str,
        analyzer_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build a compact dict suitable for including in the optimizer LLM context.

        Priority: use provided analyzer_result; fall back to lightweight upstream
        context assembly so the optimizer always has some signal.
        """
        summary: dict[str, Any] = {}

        if analyzer_result:
            # Extract the most useful fields from a serialised AnalyzerAnalysis
            summary["interpreted_goal"] = analyzer_result.get("interpreted_goal", "")
            intent = analyzer_result.get("intent") or {}
            summary["intent_label"] = intent.get("label", "")
            summary["intent_confidence"] = intent.get("confidence", 0.0)
            summary["context_summary"] = analyzer_result.get("context_summary", "")

            ambiguities = analyzer_result.get("ambiguities") or []
            if ambiguities:
                summary["ambiguities"] = [
                    a.get("issue", "") for a in ambiguities if isinstance(a, dict)
                ][:5]

            missing = analyzer_result.get("missing_information") or []
            if missing:
                summary["missing_information"] = [
                    m.get("item", "") for m in missing if isinstance(m, dict)
                ][:5]

            contradictions = analyzer_result.get("contradictions") or []
            if contradictions:
                summary["contradictions"] = [
                    c.get("issue", "") for c in contradictions if isinstance(c, dict)
                ][:3]

            weaknesses = analyzer_result.get("weaknesses") or []
            if weaknesses:
                summary["weaknesses"] = [
                    w.get("weakness", "") for w in weaknesses if isinstance(w, dict)
                ][:5]

            recommendations = analyzer_result.get("recommendations") or []
            if recommendations:
                summary["recommendations"] = [
                    r.get("recommendation", "") for r in recommendations
                    if isinstance(r, dict)
                ][:MAX_RECOMMENDATIONS_IN_CONTEXT]

            iq = analyzer_result.get("instruction_quality") or {}
            if iq:
                summary["instruction_quality_score"] = iq.get("score", 0.0)
                summary["instruction_quality_reason"] = iq.get("reason", "")

        else:
            # Lightweight fallback — run only quick deterministic services
            try:
                from app.services.preprocessing_service import preprocess_prompt

                prep = preprocess_prompt(prompt)
                summary["word_count"] = getattr(prep.text, "word_count", 0)
                summary["output_formats"] = list(prep.output_formats or [])
            except Exception:
                pass

            try:
                from app.services.intent_service import classify_prompt

                intent_resp = classify_prompt(prompt)
                summary["intent_label"] = intent_resp.intent.value
                summary["intent_confidence"] = float(intent_resp.confidence)
            except Exception:
                pass

            try:
                from app.services.quality_service import predict_quality

                quality = predict_quality(prompt)
                summary["baseline_quality_score"] = float(quality.score)
                summary["quality_category"] = quality.label.value
            except Exception:
                pass

        return summary

    # -----------------------------------------------------------------------
    # LLM invocation
    # -----------------------------------------------------------------------
    def _run_llm_optimization(
        self,
        svc: Any,
        prompt: str,
        mode: str,
        analysis_summary: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        """Invoke LLMService with optimizer prompt and extract valid JSON."""
        config = svc.config
        max_attempts = 1 + config.max_retries

        system_prompt = get_system_prompt(mode)
        user_message = build_optimizer_input(
            original_prompt=prompt,
            analysis_summary=analysis_summary,
            mode=mode,
            max_placeholders=MAX_PLACEHOLDERS,
            char_limit=ANALYSIS_SUMMARY_CHAR_LIMIT,
        )

        last_error = ""
        raw_response = ""

        for attempt in range(1, max_attempts + 1):
            try:
                raw_response = svc.provider.generate(
                    prompt=user_message,
                    system_prompt=system_prompt,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )
            except Exception as exc:
                LOGGER.error("Optimizer LLM generate() failed attempt %d: %s", attempt, exc)
                raise

            if not raw_response or not raw_response.strip():
                last_error = "Model returned empty response."
                if attempt < max_attempts:
                    user_message = build_optimizer_correction_prompt(raw_response, last_error)
                    continue
                raise RetryExhaustedError("Optimizer: empty response after retries.")

            try:
                parsed = self._extract_json(raw_response)
            except ValueError as exc:
                last_error = str(exc)
                LOGGER.warning("Optimizer JSON extraction failed attempt %d: %s", attempt, exc)
                if attempt < max_attempts:
                    user_message = build_optimizer_correction_prompt(raw_response, last_error)
                    continue
                raise RetryExhaustedError(
                    f"Optimizer: JSON extraction exhausted retries: {last_error}"
                ) from exc

            try:
                self._validate_llm_dict(parsed)
                meta = svc.provider.model_metadata()
                return parsed, meta.get("name", "unknown")
            except ValueError as exc:
                last_error = str(exc)
                LOGGER.warning(
                    "Optimizer schema validation failed attempt %d: %s", attempt, exc
                )
                if attempt < max_attempts:
                    user_message = build_optimizer_correction_prompt(raw_response, last_error)
                    continue
                raise RetryExhaustedError(
                    f"Optimizer: schema validation exhausted retries: {last_error}"
                ) from exc

        raise RetryExhaustedError(f"Optimizer retry limit reached: {last_error}")

    def _extract_json(self, raw_text: str) -> dict[str, Any]:
        """Extract and parse JSON from raw model output defensively."""
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

        raise ValueError(f"No JSON object found in response: '{text[:100]}...'")

    def _validate_llm_dict(self, data: dict[str, Any]) -> None:
        """Validate required fields and value types in the LLM output dict."""
        if not isinstance(data.get("optimized_prompt"), str) or not data["optimized_prompt"].strip():
            raise ValueError("'optimized_prompt' must be a non-empty string")
        if "summary" not in data:
            raise ValueError("Missing required field: 'summary'")
        # improvement_score_delta must be numeric
        delta = data.get("improvement_score_delta", 0.0)
        try:
            float(delta)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"'improvement_score_delta' must be a float, got {delta!r}"
            ) from exc
        # changes must be a list of dicts with category and description
        for ch in data.get("changes", []):
            if isinstance(ch, dict):
                if "category" not in ch:
                    raise ValueError("Each change must have a 'category' field")
                if "description" not in ch:
                    raise ValueError("Each change must have a 'description' field")
        # preserved_requirements must be a list of dicts with requirement and reason
        for pr in data.get("preserved_requirements", []):
            if isinstance(pr, dict):
                if "requirement" not in pr:
                    raise ValueError("Each preserved_requirement must have a 'requirement' field")
                if "reason" not in pr:
                    raise ValueError("Each preserved_requirement must have a 'reason' field")

    # -----------------------------------------------------------------------
    # Build OptimizerResult from LLM output
    # -----------------------------------------------------------------------
    def _build_from_llm_output(
        self,
        llm: dict[str, Any],
        mode: OptimizationMode,
    ) -> OptimizerResult:
        """Assemble OptimizerResult from validated LLM dict."""

        def _safe_list(obj: Any) -> list:
            return obj if isinstance(obj, list) else []

        changes = [
            ChangeRecord(
                category=str(c.get("category", "clarity") if isinstance(c, dict) else "clarity"),
                description=str(c.get("description", "") if isinstance(c, dict) else c),
            )
            for c in _safe_list(llm.get("changes"))
            if c
        ]

        preserved = [
            PreservedRequirement(
                requirement=str(p.get("requirement", "") if isinstance(p, dict) else p),
                reason=str(p.get("reason", "") if isinstance(p, dict) else "Preserved as stated."),
            )
            for p in _safe_list(llm.get("preserved_requirements"))
            if p
        ]

        placeholders = [
            str(ph) for ph in _safe_list(llm.get("placeholders_inserted")) if ph
        ]

        delta = 0.0
        try:
            delta = float(llm.get("improvement_score_delta", 0.0))
        except (TypeError, ValueError):
            pass

        return OptimizerResult(
            optimized_prompt=str(llm["optimized_prompt"]),
            summary=str(llm.get("summary", "")),
            changes=changes,
            preserved_requirements=preserved,
            placeholders_inserted=placeholders,
            improvement_score_delta=delta,
            metadata=OptimizerMetadata(
                optimizer_version=OPTIMIZER_VERSION,
                llm_model="unknown",        # overwritten by caller
                optimizer_mode=OptimizerMode.UNAVAILABLE,  # overwritten by caller
                optimization_mode=mode,
                latency_ms=0.0,             # overwritten by caller
            ),
        )

    # -----------------------------------------------------------------------
    # Deterministic UNAVAILABLE fallback
    # -----------------------------------------------------------------------
    def _build_unavailable_result(
        self,
        prompt: str,
        mode: OptimizationMode,
        analysis_summary: dict[str, Any],
    ) -> OptimizerResult:
        """Return the original prompt unchanged when LLM is unavailable.

        The optimized_prompt is the ORIGINAL prompt — we never fabricate an
        optimized version without LLM assistance.  The caller is explicitly
        told via optimizer_mode=UNAVAILABLE.
        """
        changes: list[ChangeRecord] = []
        preserved: list[PreservedRequirement] = []

        # Record any clearly extractable requirements from analysis summary
        for rec in (analysis_summary.get("recommendations") or [])[:3]:
            if rec:
                changes.append(
                    ChangeRecord(
                        category="completeness",
                        description=(
                            f"[UNAVAILABLE] Recommendation could not be applied without LLM: {rec}"
                        ),
                    )
                )

        # Preserve goal if known
        goal = analysis_summary.get("interpreted_goal", "")
        if goal:
            preserved.append(
                PreservedRequirement(
                    requirement=goal,
                    reason="Preserved from analyzer result; no LLM available to optimize.",
                )
            )

        return OptimizerResult(
            optimized_prompt=prompt,  # original returned unchanged
            summary=(
                f"LLM optimization is unavailable (mode: {mode.value}). "
                "The original prompt is returned unchanged. "
                "Enable the local LLM (LLM_ENABLED=true) to generate an optimized version."
            ),
            changes=changes,
            preserved_requirements=preserved,
            placeholders_inserted=[],
            improvement_score_delta=0.0,
            metadata=OptimizerMetadata(
                optimizer_version=OPTIMIZER_VERSION,
                llm_model="none",
                optimizer_mode=OptimizerMode.UNAVAILABLE,
                optimization_mode=mode,
                latency_ms=0.0,  # overwritten by caller
            ),
        )
