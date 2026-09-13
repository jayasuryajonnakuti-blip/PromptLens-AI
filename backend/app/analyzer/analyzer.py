"""Core PromptAnalyzer class for PromptLens (Step 15).

Orchestrates:
1. Upstream context assembly (Steps 8-13) via AnalyzerContext
2. Deterministic evidence generation
3. LLM-assisted analysis via Step 14 LLMService (optional)
4. Fallback UNAVAILABLE analysis when LLM is disabled/unavailable
5. Schema validation and evidence attachment
6. Return typed AnalyzerAnalysis

Key invariants:
- The Step 13 overall quality score is NEVER replaced.  The LLM assessment
  goes into instruction_quality.score which is a distinct, labelled field.
- analysis_mode is ALWAYS set accurately (LOCAL_LLM | MOCK | UNAVAILABLE).
  The analyzer NEVER silently uses mock output in production.
- Evidence ordering: preprocessing > NLP > intent > quality_ml > scoring > LLM.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from app.analyzer.config import ANALYZER_VERSION, EVIDENCE_BLOCK_CHAR_LIMIT
from app.analyzer.context import AnalyzerContext, build_analyzer_context
from app.analyzer.evidence import build_deterministic_evidence, build_llm_evidence_items
from app.analyzer.prompts import (
    PROMPTLENS_ANALYZER_V1,
    build_analyzer_correction_prompt,
    build_analyzer_input,
)
from app.analyzer.schemas import (
    AnalysisMode,
    AnalyzerAnalysis,
    AnalyzerMetadata,
    AmbiguityItem,
    ContradictionItem,
    InstructionQualityResult,
    IntentResult,
    MissingInfoItem,
    RecommendationItem,
    RecommendationPriority,
    StrengthItem,
    WeaknessItem,
)
from app.llm.errors import LLMDisabledError, ModelNotDownloadedError, RetryExhaustedError
from app.llm.schemas import LLMHealthStatus

LOGGER = logging.getLogger(__name__)

_VALID_PRIORITIES = {p.value for p in RecommendationPriority}


class PromptAnalyzer:
    """Evidence-grounded prompt analyzer combining deterministic and LLM signals."""

    def analyze(self, prompt: str) -> AnalyzerAnalysis:
        """Run complete analysis for the given prompt.

        Steps:
        1. Build upstream context (Steps 8-13)
        2. Build deterministic evidence items
        3. Attempt LLM-assisted analysis if available
        4. Fall back to UNAVAILABLE path if LLM is off/missing
        5. Attach all evidence; return AnalyzerAnalysis
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty or whitespace-only")

        t0 = time.perf_counter()

        # 1. Assemble upstream context
        ctx = build_analyzer_context(prompt)

        # 2. Deterministic evidence
        deterministic_evidence = build_deterministic_evidence(ctx)

        # 3. Attempt LLM analysis
        llm_output: dict[str, Any] | None = None
        analysis_mode = AnalysisMode.UNAVAILABLE
        llm_model_name = "none"

        try:
            from app.services.llm_service import get_llm_service

            svc = get_llm_service()
            health = svc.health()

            if health == LLMHealthStatus.MODEL_AVAILABLE:
                context_dict = ctx.to_compact_dict()
                user_message = build_analyzer_input(
                    prompt=prompt,
                    context_dict=context_dict,
                    char_limit=EVIDENCE_BLOCK_CHAR_LIMIT,
                )
                llm_output, llm_model_name = self._run_llm_analysis(svc, user_message)

                # Determine mode from provider type
                meta = svc.provider.model_metadata()
                provider_name = meta.get("provider", "")
                if provider_name == "mock":
                    analysis_mode = AnalysisMode.MOCK
                else:
                    analysis_mode = AnalysisMode.LOCAL_LLM

            else:
                LOGGER.info(
                    "LLM health state is %s — falling back to UNAVAILABLE mode.", health
                )
                analysis_mode = AnalysisMode.UNAVAILABLE

        except (LLMDisabledError, ModelNotDownloadedError):
            LOGGER.info("LLM unavailable for analyzer — using UNAVAILABLE mode.")
            analysis_mode = AnalysisMode.UNAVAILABLE
        except RetryExhaustedError as exc:
            LOGGER.warning("Analyzer LLM retries exhausted: %s — using UNAVAILABLE mode.", exc)
            analysis_mode = AnalysisMode.UNAVAILABLE
        except Exception as exc:
            LOGGER.warning("Unexpected LLM error in analyzer: %s — using UNAVAILABLE mode.", exc)
            analysis_mode = AnalysisMode.UNAVAILABLE

        # 4. Build analysis from LLM output or deterministic fallback
        if llm_output is not None:
            analysis = self._build_from_llm_output(llm_output, ctx, deterministic_evidence)
        else:
            analysis = self._build_unavailable_analysis(ctx, deterministic_evidence)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # 5. Attach metadata (always overwrite mode/version from the inner build)
        analysis.metadata = AnalyzerMetadata(
            analyzer_version=ANALYZER_VERSION,
            llm_model=llm_model_name,
            analysis_mode=analysis_mode,
            latency_ms=latency_ms,
        )
        return analysis

    # -----------------------------------------------------------------------
    # LLM invocation
    # -----------------------------------------------------------------------
    def _run_llm_analysis(
        self, svc: Any, user_message: str
    ) -> tuple[dict[str, Any], str]:
        """Invoke LLMService with the analyzer prompt and extract valid JSON.

        Returns (parsed_dict, model_name).
        """
        config = svc.config
        max_attempts = 1 + config.max_retries

        raw_response = ""
        last_error = ""

        for attempt in range(1, max_attempts + 1):
            try:
                raw_response = svc.provider.generate(
                    prompt=user_message,
                    system_prompt=PROMPTLENS_ANALYZER_V1,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )
            except Exception as exc:
                LOGGER.error("Analyzer LLM generate() failed attempt %d: %s", attempt, exc)
                raise

            if not raw_response or not raw_response.strip():
                last_error = "Model returned empty response."
                if attempt < max_attempts:
                    user_message = build_analyzer_correction_prompt(raw_response, last_error)
                    continue
                raise RetryExhaustedError("Analyzer: empty response after retries.")

            try:
                parsed = self._extract_json(raw_response)
            except ValueError as exc:
                last_error = str(exc)
                LOGGER.warning("Analyzer JSON extraction failed attempt %d: %s", attempt, exc)
                if attempt < max_attempts:
                    user_message = build_analyzer_correction_prompt(raw_response, last_error)
                    continue
                raise RetryExhaustedError(
                    f"Analyzer: JSON extraction exhausted retries: {last_error}"
                ) from exc

            try:
                self._validate_llm_dict(parsed)
                meta = svc.provider.model_metadata()
                return parsed, meta.get("name", "unknown")
            except ValueError as exc:
                last_error = str(exc)
                LOGGER.warning(
                    "Analyzer schema validation failed attempt %d: %s", attempt, exc
                )
                if attempt < max_attempts:
                    user_message = build_analyzer_correction_prompt(raw_response, last_error)
                    continue
                raise RetryExhaustedError(
                    f"Analyzer: schema validation exhausted retries: {last_error}"
                ) from exc

        raise RetryExhaustedError(f"Analyzer retry limit reached: {last_error}")

    def _extract_json(self, raw_text: str) -> dict[str, Any]:
        """Extract and parse JSON from raw model output defensively."""
        text = raw_text.strip()

        # Markdown fences
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
        """Validate required fields and value ranges in the LLM output dict."""
        required = [
            "interpreted_goal",
            "context_summary",
            "intent_label",
            "intent_confidence",
            "instruction_quality",
        ]
        for key in required:
            if key not in data:
                raise ValueError(f"Missing required field: '{key}'")

        iq = data.get("instruction_quality", {})
        if not isinstance(iq, dict):
            raise ValueError("instruction_quality must be a dict")
        score = iq.get("score")
        if score is None or not (0.0 <= float(score) <= 100.0):
            raise ValueError(
                f"instruction_quality.score must be between 0.0 and 100.0, got {score!r}"
            )

        ic = data.get("intent_confidence", 0.0)
        if not (0.0 <= float(ic) <= 1.0):
            raise ValueError(
                f"intent_confidence must be between 0.0 and 1.0, got {ic!r}"
            )

        # Validate recommendation priorities
        for rec in data.get("recommendations", []):
            if isinstance(rec, dict):
                p = rec.get("priority", "MEDIUM")
                if str(p).upper() not in _VALID_PRIORITIES:
                    raise ValueError(
                        f"Invalid recommendation priority: {p!r}. "
                        f"Must be one of {_VALID_PRIORITIES}."
                    )

    # -----------------------------------------------------------------------
    # Build AnalyzerAnalysis from LLM output
    # -----------------------------------------------------------------------
    def _build_from_llm_output(
        self,
        llm: dict[str, Any],
        ctx: AnalyzerContext,
        deterministic_evidence: list,
    ) -> AnalyzerAnalysis:
        """Assemble AnalyzerAnalysis from validated LLM dict + upstream evidence."""

        def _safe_list(obj: Any) -> list:
            return obj if isinstance(obj, list) else []

        ambiguities = [
            AmbiguityItem(
                issue=str(a.get("issue", "")),
                evidence=str(a.get("evidence", "")),
                confidence=float(a.get("confidence", 0.60)),
            )
            for a in _safe_list(llm.get("ambiguities"))
            if isinstance(a, dict) and a.get("issue")
        ]

        missing_info = [
            MissingInfoItem(
                item=str(m.get("item", "")),
                why_it_matters=str(m.get("why_it_matters", "")),
                confidence=float(m.get("confidence", 0.60)),
            )
            for m in _safe_list(llm.get("missing_information"))
            if isinstance(m, dict) and m.get("item")
        ]

        contradictions = [
            ContradictionItem(
                issue=str(c.get("issue", "")),
                evidence=str(c.get("evidence", "")),
                confidence=float(c.get("confidence", 0.60)),
            )
            for c in _safe_list(llm.get("contradictions"))
            if isinstance(c, dict) and c.get("issue")
        ]

        strengths = [
            StrengthItem(
                strength=str(s.get("strength", "") if isinstance(s, dict) else s),
                evidence=str(s.get("evidence", "") if isinstance(s, dict) else ""),
            )
            for s in _safe_list(llm.get("strengths"))
        ]

        weaknesses = [
            WeaknessItem(
                weakness=str(w.get("weakness", "") if isinstance(w, dict) else w),
                evidence=str(w.get("evidence", "") if isinstance(w, dict) else ""),
            )
            for w in _safe_list(llm.get("weaknesses"))
        ]

        recommendations = [
            RecommendationItem(
                recommendation=str(r.get("recommendation", "") if isinstance(r, dict) else r),
                reason=str(r.get("reason", "") if isinstance(r, dict) else ""),
                priority=RecommendationPriority(
                    str(r.get("priority", "MEDIUM")).upper()
                    if isinstance(r, dict) and str(r.get("priority", "")).upper() in _VALID_PRIORITIES
                    else "MEDIUM"
                ),
            )
            for r in _safe_list(llm.get("recommendations"))
        ]

        iq = llm.get("instruction_quality", {})

        # Build LLM-sourced evidence items and attach after deterministic evidence
        llm_evidence = build_llm_evidence_items(
            ambiguities=[a.model_dump() for a in ambiguities],
            missing_info=[m.model_dump() for m in missing_info],
            contradictions=[c.model_dump() for c in contradictions],
        )
        all_evidence = deterministic_evidence + llm_evidence

        # Resolve intent: prefer LLM label if intent classifier was unavailable
        intent_label = llm.get("intent_label", ctx.intent_label or "other")
        intent_conf = float(llm.get("intent_confidence", ctx.intent_confidence))

        return AnalyzerAnalysis(
            interpreted_goal=str(llm.get("interpreted_goal", "")),
            intent=IntentResult(label=intent_label, confidence=intent_conf),
            context_summary=str(llm.get("context_summary", "")),
            ambiguities=ambiguities,
            missing_information=missing_info,
            contradictions=contradictions,
            instruction_quality=InstructionQualityResult(
                score=float(iq.get("score", 50.0)),
                reason=str(iq.get("reason", "")),
            ),
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            evidence=all_evidence,
            metadata=AnalyzerMetadata(
                analyzer_version=ANALYZER_VERSION,
                llm_model="unknown",  # overwritten by caller
                analysis_mode=AnalysisMode.UNAVAILABLE,  # overwritten by caller
            ),
        )

    # -----------------------------------------------------------------------
    # Deterministic UNAVAILABLE fallback
    # -----------------------------------------------------------------------
    def _build_unavailable_analysis(
        self,
        ctx: AnalyzerContext,
        deterministic_evidence: list,
    ) -> AnalyzerAnalysis:
        """Build a best-effort analysis from deterministic signals only.

        The LLM is unavailable; the analysis is explicitly labelled as such.
        No mock data is used — only evidence actually present in the signals.
        """
        # Derive interpreted goal heuristic from intent + word_count
        if ctx.intent_available:
            goal_hint = f"Classified as '{ctx.intent_label}' intent"
        else:
            goal_hint = "Primary intent could not be determined without LLM analysis"

        if ctx.word_count > 0:
            interpreted_goal = (
                f"{goal_hint}. "
                f"Prompt is {ctx.word_count} words in length. "
                "Full goal interpretation requires LLM analysis (currently unavailable)."
            )
        else:
            interpreted_goal = (
                "Goal interpretation unavailable — LLM analysis is disabled or model not loaded."
            )

        # Derive simple strengths/weaknesses from deterministic signals
        strengths: list[StrengthItem] = []
        weaknesses: list[WeaknessItem] = []
        recommendations: list[RecommendationItem] = []

        if ctx.has_role_definition:
            strengths.append(
                StrengthItem(
                    strength="Role definition is present.",
                    evidence="Preprocessing detected role-definition language.",
                )
            )
        if ctx.has_constraints:
            strengths.append(
                StrengthItem(
                    strength="Explicit constraints or restrictions provided.",
                    evidence="Preprocessing detected constraint/restriction language.",
                )
            )
        if ctx.has_examples:
            strengths.append(
                StrengthItem(
                    strength="Examples or sample content included.",
                    evidence="Preprocessing detected example markers.",
                )
            )

        if ctx.word_count < 20:
            weaknesses.append(
                WeaknessItem(
                    weakness="Prompt may be too brief for reliable analysis.",
                    evidence=f"Preprocessing reports only {ctx.word_count} words.",
                )
            )
            recommendations.append(
                RecommendationItem(
                    recommendation="Expand the prompt with more context and constraints.",
                    reason="Very short prompts are harder to interpret and typically produce lower-quality outputs.",
                    priority=RecommendationPriority.HIGH,
                )
            )

        # Derive weak dimensions from Step 13
        for dim in ctx.top_dimensions[:3]:
            dim_score = dim.get("score", 100.0)
            dim_name = dim.get("dimension", "unknown")
            if dim_score < 50:
                weaknesses.append(
                    WeaknessItem(
                        weakness=f"Dimension '{dim_name}' is weak (score: {dim_score:.0f}/100).",
                        evidence=f"Step 13 Scoring & Fusion: status={dim.get('status', '')}.",
                    )
                )
                recommendations.append(
                    RecommendationItem(
                        recommendation=f"Improve the '{dim_name}' dimension.",
                        reason=dim.get("reason", "Low score from scoring engine."),
                        priority=RecommendationPriority.MEDIUM,
                    )
                )

        # Step 13 recommendations as LOW priority items (already deterministic)
        for rec_text in ctx.step13_recommendations[:3]:
            recommendations.append(
                RecommendationItem(
                    recommendation=rec_text,
                    reason="Derived from Step 13 Scoring & Fusion engine.",
                    priority=RecommendationPriority.LOW,
                )
            )

        # Instruction quality from Quality ML if available, else neutral
        if ctx.quality_available:
            iq_score = ctx.quality_score
            iq_reason = (
                f"Estimated from Quality ML model (category: {ctx.quality_category}). "
                "LLM-based instruction quality assessment unavailable."
            )
        elif ctx.step13_available:
            iq_score = ctx.step13_overall_score
            iq_reason = (
                f"Derived from Step 13 overall score ({ctx.step13_category}). "
                "LLM-based instruction quality assessment unavailable."
            )
        else:
            iq_score = 0.0
            iq_reason = "Instruction quality assessment unavailable — all analysis services offline."

        intent_label = ctx.intent_label if ctx.intent_available else "unknown"
        intent_conf = ctx.intent_confidence if ctx.intent_available else 0.0

        return AnalyzerAnalysis(
            interpreted_goal=interpreted_goal,
            intent=IntentResult(label=intent_label, confidence=intent_conf),
            context_summary=(
                "LLM analysis unavailable. Context summary derived from deterministic signals only."
            ),
            ambiguities=[],
            missing_information=[],
            contradictions=[],
            instruction_quality=InstructionQualityResult(score=iq_score, reason=iq_reason),
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            evidence=deterministic_evidence,
            metadata=AnalyzerMetadata(
                analyzer_version=ANALYZER_VERSION,
                llm_model="none",
                analysis_mode=AnalysisMode.UNAVAILABLE,
            ),
        )
