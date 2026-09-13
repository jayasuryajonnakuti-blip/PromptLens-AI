"""Bounded Context Builder for PromptLens AI Agent Loop (Step 19).

Invariants:
- Generates compact, actionable feedback for subsequent optimization iterations.
- Injects structured Critic and Validator findings without dumping raw transcripts.
- Strictly bounds context size to prevent prompt bloat or LLM context overflow.
"""
from __future__ import annotations

from typing import Any

from app.agent.config import AGENT_CONTEXT_CHAR_LIMIT
from app.agent.state import AgentState
from app.optimizer.schemas import OptimizationMode
from app.validator.schemas import IssueSeverity


def build_iteration_context(
    state: AgentState,
    mode: OptimizationMode = OptimizationMode.BALANCED,
) -> dict[str, Any]:
    """Build a compact, enriched analyzer_result dict for the Optimizer.

    For iteration 1:
        Returns base analyzer summary.
    For iteration 2 or 3:
        Injects structured feedback from previous Critic and Validator evaluations
        detailing lost requirements, lost constraints, contradictions, and score regressions.
    """
    # Start with base analyzer dictionary or minimal structure
    base: dict[str, Any] = dict(state.analyzer_dict or {})

    # Ensure baseline keys exist
    interpreted_goal = base.get("interpreted_goal", "")
    intent = base.get("intent", {})
    context_summary = base.get("context_summary", "")
    ambiguities = list(base.get("ambiguities") or [])
    missing_info = list(base.get("missing_information") or [])
    contradictions = list(base.get("contradictions") or [])
    weaknesses = list(base.get("weaknesses") or [])
    recommendations = list(base.get("recommendations") or [])

    # Iteration 1: return clean base context
    if not state.iterations:
        return {
            "interpreted_goal": interpreted_goal,
            "intent": intent,
            "context_summary": context_summary,
            "ambiguities": ambiguities,
            "missing_information": missing_info,
            "contradictions": contradictions,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
        }

    # Iteration 2+: Inject previous feedback
    last_iteration = state.iterations[-1]
    last_validator = state.validation_history[-1]
    last_critic = state.critic_history[-1]

    enriched_weaknesses: list[dict[str, str]] = []
    enriched_recommendations: list[dict[str, str]] = []
    enriched_missing: list[dict[str, str]] = list(missing_info)
    enriched_contradictions: list[dict[str, str]] = list(contradictions)

    # 1. Validator issues (prioritize ERROR and WARNING)
    for issue in last_validator.issues:
        prefix = "CRITICAL FAILURE" if issue.severity == IssueSeverity.ERROR else "WARNING"
        enriched_weaknesses.append({
            "weakness": f"[{prefix} {issue.code}] {issue.message}",
            "evidence": issue.evidence[:150] if issue.evidence else "",
        })
        enriched_recommendations.append({
            "recommendation": f"FIX {issue.code}: {issue.message}",
            "priority": "HIGH" if issue.severity == IssueSeverity.ERROR else "MEDIUM",
        })

    # 2. Critic lost requirements
    for req in last_critic.lost_requirements:
        enriched_missing.append({
            "item": f"Lost requirement from original prompt: {req}",
            "importance": "CRITICAL",
        })
        enriched_recommendations.append({
            "recommendation": f"RESTORE REQUIREMENT: Must preserve '{req}' verbatim or equivalent.",
            "priority": "HIGH",
        })

    # 3. Critic unsupported assumptions / contradictions
    for assumption in last_critic.unsupported_assumptions:
        enriched_contradictions.append({
            "issue": f"Unsupported assumption introduced: {assumption}",
        })
        enriched_recommendations.append({
            "recommendation": f"REMOVE UNSUPPORTED ADDITION: Do not assume '{assumption}'.",
            "priority": "HIGH",
        })

    # 4. Critic specific recommendations
    for rec in last_critic.recommendations[:3]:
        enriched_recommendations.append({
            "recommendation": f"Critic Advice: {rec}",
            "priority": "MEDIUM",
        })

    # Combine with original findings
    combined_weaknesses = enriched_weaknesses + weaknesses
    combined_recommendations = enriched_recommendations + recommendations

    # Build actionable context summary
    iter_num = state.iteration_count + 1
    score_before_str = (
        f"{state.score_history[-1]:.1f}" if state.score_history else "N/A"
    )
    feedback_summary = (
        f"Iteration {iter_num} Optimization Guidance: "
        f"Attempt {last_iteration.iteration_number} was rejected by Validator ({last_iteration.decision}). "
        f"Previous score: {score_before_str}. "
        f"Identified {len(last_validator.issues)} validation issues and {len(last_critic.lost_requirements)} lost requirements. "
        f"You MUST preserve all negative constraints, stated requirements, and format rules. Do NOT expand scope."
    )

    result_dict: dict[str, Any] = {
        "interpreted_goal": interpreted_goal,
        "intent": intent,
        "context_summary": feedback_summary[:AGENT_CONTEXT_CHAR_LIMIT],
        "ambiguities": ambiguities[:5],
        "missing_information": enriched_missing[:6],
        "contradictions": enriched_contradictions[:5],
        "weaknesses": combined_weaknesses[:8],
        "recommendations": combined_recommendations[:8],
    }

    return result_dict
