"""Deterministic evaluation rules and checks for PromptLens AI Critic (Step 17).

Provides independent, deterministic verification of:
1. Intent regression (Step 10 classifier & textual cues)
2. Format regression (Step 8 preprocessing formats)
3. Negative constraint preservation (exclusions, boundaries)
4. Score & dimension regression (Step 13 scoring deltas)
5. Verbosity & bloat analysis (word and character ratios)
6. Contradiction introduction
7. Unsupported technical stack assumptions
"""
from __future__ import annotations

import re
from typing import NamedTuple

from app.critic.config import (
    LENGTH_EXPANSION_FAIL_RATIO,
    LENGTH_EXPANSION_WARN_RATIO,
    LENGTH_REDUCTION_WARN_RATIO,
    SCORE_REGRESSION_FAIL_THRESHOLD,
    SCORE_REGRESSION_WARN_THRESHOLD,
    SEMANTIC_SIMILARITY_SUSPICIOUS,
)
from app.critic.context import CriticContext
from app.critic.schemas import CriticDecision, CriticIssue, IssueSeverity


class DeterministicCheckResults(NamedTuple):
    issues: list[CriticIssue]
    suggested_decision: CriticDecision
    intent_preservation_score: float
    requirement_preservation_score: float
    base_critique_score: float
    lost_requirements: list[str]
    introduced_requirements: list[str]
    unsupported_assumptions: list[str]


def check_intent_regression(ctx: CriticContext, orig_prompt: str, opt_prompt: str) -> list[CriticIssue]:
    issues: list[CriticIssue] = []

    # High-confidence intent divergence
    if (
        not ctx.intent_match
        and ctx.intent_confidence_original >= 0.40
        and ctx.intent_confidence_optimized >= 0.40
    ):
        issues.append(
            CriticIssue(
                type="intent_regression",
                severity=IssueSeverity.ERROR,
                description=(
                    f"Intent classification changed from '{ctx.intent_original}' "
                    f"to '{ctx.intent_optimized}'. The optimization may have altered the primary task."
                ),
                evidence=(
                    f"Original intent: {ctx.intent_original} ({ctx.intent_confidence_original:.2f}), "
                    f"Optimized intent: {ctx.intent_optimized} ({ctx.intent_confidence_optimized:.2f})"
                ),
            )
        )

    # Extreme semantic departure
    if ctx.similarity_available and ctx.semantic_similarity < SEMANTIC_SIMILARITY_SUSPICIOUS:
        issues.append(
            CriticIssue(
                type="semantic_divergence",
                severity=IssueSeverity.ERROR,
                description=(
                    "Extremely low semantic similarity between original and optimized prompts. "
                    "The optimization may have drifted from the original subject matter."
                ),
                evidence=f"Cosine semantic similarity: {ctx.semantic_similarity:.3f}",
            )
        )

    return issues


def check_format_regression(ctx: CriticContext) -> tuple[list[CriticIssue], list[str]]:
    issues: list[CriticIssue] = []
    lost_reqs: list[str] = []

    orig_formats = set(ctx.formats_original)
    opt_formats = set(ctx.formats_optimized)

    for fmt in orig_formats:
        if fmt not in opt_formats:
            lost_reqs.append(f"Output format '{fmt}'")
            issues.append(
                CriticIssue(
                    type="lost_output_format",
                    severity=IssueSeverity.ERROR,
                    description=f"Original prompt explicitly requested '{fmt}' format, which was omitted in the optimized prompt.",
                    evidence=f"Original formats: {list(orig_formats)}, Optimized formats: {list(opt_formats)}",
                )
            )

    return issues, lost_reqs


def check_negative_constraints(
    orig_prompt: str, opt_prompt: str
) -> tuple[list[CriticIssue], list[str]]:
    issues: list[CriticIssue] = []
    lost_reqs: list[str] = []

    neg_patterns = [
        r"\b(?:do not|don't|must not|avoid|never|prohibit)\s+([a-zA-Z0-9_\- ]{3,35})\b",
        r"\b(?:without)\s+([a-zA-Z0-9_\- ]{3,30})\b",
    ]

    orig_lower = orig_prompt.lower()
    opt_lower = opt_prompt.lower()

    for pattern in neg_patterns:
        for match in re.finditer(pattern, orig_lower):
            constraint_clause = match.group(0).strip()
            key_term = match.group(1).strip()
            # If the negative keyword or constraint object is missing in optimized prompt
            if key_term and key_term not in opt_lower:
                lost_reqs.append(f"Constraint: '{constraint_clause}'")
                issues.append(
                    CriticIssue(
                        type="lost_constraint",
                        severity=IssueSeverity.WARNING,
                        description=(
                            f"Original prompt contained constraint '{constraint_clause}' "
                            f"which appears weakened or absent in the optimized prompt."
                        ),
                        evidence=f"Original clause: \"{constraint_clause}\"",
                    )
                )

    return issues, lost_reqs


def check_score_regression(ctx: CriticContext) -> list[CriticIssue]:
    issues: list[CriticIssue] = []

    if ctx.scoring_available:
        if ctx.overall_score_delta <= -SCORE_REGRESSION_FAIL_THRESHOLD:
            issues.append(
                CriticIssue(
                    type="score_regression_critical",
                    severity=IssueSeverity.ERROR,
                    description=(
                        f"Significant quality regression: overall score decreased by "
                        f"{-ctx.overall_score_delta:.1f} points (from {ctx.overall_score_original:.1f} to {ctx.overall_score_optimized:.1f})."
                    ),
                    evidence=f"Original: {ctx.overall_score_original:.1f}, Optimized: {ctx.overall_score_optimized:.1f}",
                )
            )
        elif ctx.overall_score_delta <= -SCORE_REGRESSION_WARN_THRESHOLD:
            issues.append(
                CriticIssue(
                    type="score_regression_moderate",
                    severity=IssueSeverity.WARNING,
                    description=(
                        f"Moderate quality regression: overall score decreased by "
                        f"{-ctx.overall_score_delta:.1f} points."
                    ),
                    evidence=f"Original: {ctx.overall_score_original:.1f}, Optimized: {ctx.overall_score_optimized:.1f}",
                )
            )

    return issues


def check_verbosity(ctx: CriticContext) -> list[CriticIssue]:
    issues: list[CriticIssue] = []

    if ctx.word_count_ratio >= LENGTH_EXPANSION_FAIL_RATIO:
        issues.append(
            CriticIssue(
                type="excessive_verbosity",
                severity=IssueSeverity.ERROR,
                description=(
                    f"Excessive prompt expansion: word count expanded by {ctx.word_count_ratio:.1f}x "
                    f"({ctx.word_count_original} -> {ctx.word_count_optimized} words)."
                ),
                evidence=f"{ctx.word_count_original} words -> {ctx.word_count_optimized} words",
            )
        )
    elif ctx.word_count_ratio >= LENGTH_EXPANSION_WARN_RATIO:
        issues.append(
            CriticIssue(
                type="unnecessary_verbosity",
                severity=IssueSeverity.WARNING,
                description=(
                    f"Noticeable prompt expansion: word count expanded by {ctx.word_count_ratio:.1f}x. "
                    "Check for redundant boilerplate."
                ),
                evidence=f"{ctx.word_count_original} words -> {ctx.word_count_optimized} words",
            )
        )
    elif ctx.word_count_ratio <= LENGTH_REDUCTION_WARN_RATIO and ctx.word_count_original > 25:
        issues.append(
            CriticIssue(
                type="excessive_truncation",
                severity=IssueSeverity.WARNING,
                description=(
                    f"Prompt was drastically shortened by {(1 - ctx.word_count_ratio)*100:.0f}%. "
                    "Important requirements or context may have been dropped."
                ),
                evidence=f"{ctx.word_count_original} words -> {ctx.word_count_optimized} words",
            )
        )

    return issues


def check_contradictions(orig_prompt: str, opt_prompt: str) -> list[CriticIssue]:
    issues: list[CriticIssue] = []
    opt_lower = opt_prompt.lower()

    # Detect JSON only vs prose explanation
    has_json_only = any(term in opt_lower for term in ("json only", "return json only", "pure json", "valid json only"))
    has_explanation = any(term in opt_lower for term in ("explain your answer", "detailed explanation", "step by step explanation", "describe why"))

    if has_json_only and has_explanation:
        issues.append(
            CriticIssue(
                type="introduced_contradiction",
                severity=IssueSeverity.ERROR,
                description="Optimized prompt requires 'JSON only' but simultaneously requests detailed explanations.",
                evidence="Found conflicting requirements for JSON only output and narrative explanation.",
            )
        )

    return issues


def check_unsupported_technical_assumptions(
    orig_prompt: str, opt_prompt: str
) -> tuple[list[CriticIssue], list[str], list[str]]:
    """Detect specific libraries/frameworks/databases introduced without prompt justification."""
    issues: list[CriticIssue] = []
    introduced_reqs: list[str] = []
    unsupported_assumptions: list[str] = []

    tech_keywords = [
        "postgresql", "mysql", "mongodb", "sqlite", "redis",
        "django", "flask", "fastapi", "spring boot", "express.js",
        "react", "vue", "angular", "next.js", "tailwind",
        "aws", "azure", "gcp", "docker", "kubernetes"
    ]

    orig_lower = orig_prompt.lower()
    opt_lower = opt_prompt.lower()

    for tech in tech_keywords:
        pattern = rf"\b{re.escape(tech)}\b"
        # If in optimized but not in original and not wrapped in a placeholder like [DATABASE]
        if re.search(pattern, opt_lower) and not re.search(pattern, orig_lower):
            # Check if it was framed as a concrete requirement rather than an example
            # (e.g. 'Use PostgreSQL' vs 'such as PostgreSQL')
            req_pattern = rf"\b(?:use|with|using|in|require|install)\s+{re.escape(tech)}\b"
            if re.search(req_pattern, opt_lower):
                introduced_reqs.append(f"Specific technology: {tech}")
                unsupported_assumptions.append(f"Assumed tech stack: {tech}")
                issues.append(
                    CriticIssue(
                        type="unsupported_introduced_requirement",
                        severity=IssueSeverity.WARNING,
                        description=f"Optimized prompt introduced specific requirement for '{tech}' not requested in original prompt.",
                        evidence=f"'{tech}' specified as requirement without original prompt mentioning it.",
                    )
                )

    return issues, introduced_reqs, unsupported_assumptions


def run_deterministic_checks(
    ctx: CriticContext, orig_prompt: str, opt_prompt: str
) -> DeterministicCheckResults:
    """Run all deterministic checks and produce an overall score & decision."""
    issues: list[CriticIssue] = []
    lost_reqs: list[str] = []
    introduced_reqs: list[str] = []
    unsupported_assumptions: list[str] = []

    # 1. Intent check
    issues.extend(check_intent_regression(ctx, orig_prompt, opt_prompt))

    # 2. Output format check
    fmt_issues, fmt_lost = check_format_regression(ctx)
    issues.extend(fmt_issues)
    lost_reqs.extend(fmt_lost)

    # 3. Negative constraints check
    neg_issues, neg_lost = check_negative_constraints(orig_prompt, opt_prompt)
    issues.extend(neg_issues)
    lost_reqs.extend(neg_lost)

    # 4. Score regression check
    issues.extend(check_score_regression(ctx))

    # 5. Verbosity check
    issues.extend(check_verbosity(ctx))

    # 6. Contradictions check
    issues.extend(check_contradictions(orig_prompt, opt_prompt))

    # 7. Unsupported assumptions check
    tech_issues, tech_intro, tech_assump = check_unsupported_technical_assumptions(orig_prompt, opt_prompt)
    issues.extend(tech_issues)
    introduced_reqs.extend(tech_intro)
    unsupported_assumptions.extend(tech_assump)

    # Calculate baseline scores
    error_count = sum(1 for i in issues if i.severity == IssueSeverity.ERROR)
    warn_count = sum(1 for i in issues if i.severity == IssueSeverity.WARNING)

    # Intent preservation score
    intent_score = 90.0
    if not ctx.intent_match and ctx.intent_confidence_original >= 0.40:
        intent_score = 30.0
    elif ctx.similarity_available and ctx.semantic_similarity < SEMANTIC_SIMILARITY_SUSPICIOUS:
        intent_score = 40.0
    elif ctx.similarity_available:
        intent_score = max(50.0, min(100.0, ctx.semantic_similarity * 100))

    # Requirement preservation score
    req_score = 90.0 - (len(lost_reqs) * 25.0) - (len(introduced_reqs) * 10.0)
    req_score = max(10.0, min(100.0, req_score))

    # Base critique score
    base_critique = (
        (intent_score * 0.35)
        + (req_score * 0.35)
        + (max(0.0, min(100.0, ctx.overall_score_optimized)) * 0.30)
    )
    base_critique -= (error_count * 25.0)
    base_critique -= (warn_count * 5.0)
    base_critique = max(0.0, min(100.0, base_critique))

    # Determine suggested decision
    if error_count > 0 or intent_score < 50.0 or req_score < 40.0 or base_critique < 50.0:
        suggested_decision = CriticDecision.FAIL
    elif warn_count >= 2 or (50.0 <= base_critique < 70.0):
        suggested_decision = CriticDecision.NEEDS_REVIEW
    else:
        suggested_decision = CriticDecision.PASS

    return DeterministicCheckResults(
        issues=issues,
        suggested_decision=suggested_decision,
        intent_preservation_score=round(intent_score, 1),
        requirement_preservation_score=round(req_score, 1),
        base_critique_score=round(base_critique, 1),
        lost_requirements=lost_reqs,
        introduced_requirements=introduced_reqs,
        unsupported_assumptions=unsupported_assumptions,
    )
