"""Deterministic validation rules and checks for PromptLens AI Validator (Step 18).

Implements independent deterministic checks covering:
1. Schema and field integrity
2. Intent preservation
3. Requirement preservation
4. Negative constraint preservation
5. Output format preservation
6. Contradiction detection
7. Unsupported tech additions
8. Score regression & integrity
9. Length and bloat integrity
10. Critic consistency check
"""
from __future__ import annotations

import re
from typing import NamedTuple

from app.validator.config import (
    MAX_EXPANSION_RATIO_FAIL,
    MAX_EXPANSION_RATIO_WARN,
    MAX_SCORE_REGRESSION_FAIL,
    MAX_SCORE_REGRESSION_WARN,
    MAX_TRUNCATION_RATIO_WARN,
    MIN_SEMANTIC_SIMILARITY_FAIL,
    MIN_SEMANTIC_SIMILARITY_WARN,
)
from app.validator.context import ValidatorContext
from app.validator.schemas import IssueSeverity, ValidationIssue, ValidatorDecision


class DeterministicValidationSummary(NamedTuple):
    issues: list[ValidationIssue]
    passed_checks: list[str]
    failed_checks: list[str]
    decision: ValidatorDecision
    safety_score: float


# ---------------------------------------------------------------------------
# Check A: Schema / Field Validation
# ---------------------------------------------------------------------------
def check_field_integrity(ctx: ValidatorContext) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if not ctx.original_prompt.strip():
        issues.append(
            ValidationIssue(
                code="EMPTY_ORIGINAL_PROMPT",
                severity=IssueSeverity.ERROR,
                message="Original prompt cannot be empty or whitespace.",
                field_or_scope="original_prompt",
                evidence=repr(ctx.original_prompt),
            )
        )
    if not ctx.optimized_prompt.strip():
        issues.append(
            ValidationIssue(
                code="EMPTY_OPTIMIZED_PROMPT",
                severity=IssueSeverity.ERROR,
                message="Optimized prompt cannot be empty or whitespace.",
                field_or_scope="optimized_prompt",
                evidence=repr(ctx.optimized_prompt),
            )
        )

    for name, val in [("original_score", ctx.original_score), ("optimized_score", ctx.optimized_score)]:
        if not (0.0 <= val <= 100.0):
            issues.append(
                ValidationIssue(
                    code="INVALID_SCORE_RANGE",
                    severity=IssueSeverity.ERROR,
                    message=f"Score '{name}' value {val} is outside valid [0, 100] range.",
                    field_or_scope=name,
                    evidence=f"{name}={val}",
                )
            )

    if not (0.0 <= ctx.semantic_similarity <= 1.0):
        issues.append(
            ValidationIssue(
                code="INVALID_SIMILARITY_RANGE",
                severity=IssueSeverity.ERROR,
                message=f"Semantic similarity {ctx.semantic_similarity} is outside [0, 1] range.",
                field_or_scope="semantic_similarity",
                evidence=f"similarity={ctx.semantic_similarity}",
            )
        )

    return issues


# ---------------------------------------------------------------------------
# Check B: Intent Preservation
# ---------------------------------------------------------------------------
def check_intent_preservation(ctx: ValidatorContext) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if (
        not ctx.intent_match
        and ctx.original_intent_confidence >= 0.45
        and ctx.optimized_intent_confidence >= 0.45
    ):
        issues.append(
            ValidationIssue(
                code="INTENT_REGRESSION",
                severity=IssueSeverity.ERROR,
                message=(
                    f"Intent divergence detected: original prompt is classified as '{ctx.original_intent}', "
                    f"whereas optimized prompt shifted to '{ctx.optimized_intent}'."
                ),
                field_or_scope="intent",
                evidence=(
                    f"Original: {ctx.original_intent} (conf: {ctx.original_intent_confidence:.2f}), "
                    f"Optimized: {ctx.optimized_intent} (conf: {ctx.optimized_intent_confidence:.2f})"
                ),
            )
        )
    elif not ctx.intent_match and (ctx.original_intent_confidence >= 0.35 or ctx.optimized_intent_confidence >= 0.35):
        issues.append(
            ValidationIssue(
                code="INTENT_SHIFT_WARNING",
                severity=IssueSeverity.WARNING,
                message=(
                    f"Possible intent shift: original intent '{ctx.original_intent}' "
                    f"vs optimized intent '{ctx.optimized_intent}'."
                ),
                field_or_scope="intent",
                evidence=f"Original: {ctx.original_intent}, Optimized: {ctx.optimized_intent}",
            )
        )

    # Semantic similarity check
    if ctx.similarity_available:
        if ctx.semantic_similarity < MIN_SEMANTIC_SIMILARITY_FAIL:
            issues.append(
                ValidationIssue(
                    code="FATAL_SEMANTIC_DIVERGENCE",
                    severity=IssueSeverity.ERROR,
                    message="Extreme semantic divergence: optimized prompt departs entirely from original topic.",
                    field_or_scope="semantic_similarity",
                    evidence=f"Similarity: {ctx.semantic_similarity:.4f} < {MIN_SEMANTIC_SIMILARITY_FAIL}",
                )
            )
        elif ctx.semantic_similarity < MIN_SEMANTIC_SIMILARITY_WARN:
            issues.append(
                ValidationIssue(
                    code="SUSPICIOUS_SEMANTIC_DIVERGENCE",
                    severity=IssueSeverity.WARNING,
                    message="Moderate semantic divergence: optimized prompt may have drifted from original context.",
                    field_or_scope="semantic_similarity",
                    evidence=f"Similarity: {ctx.semantic_similarity:.4f} < {MIN_SEMANTIC_SIMILARITY_WARN}",
                )
            )

    return issues


# ---------------------------------------------------------------------------
# Check C: Requirement Preservation
# ---------------------------------------------------------------------------
def check_requirement_preservation(orig_prompt: str, opt_prompt: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    orig_lower = orig_prompt.lower()
    opt_lower = opt_prompt.lower()

    # Detect explicit "must / require / include" directives
    req_pattern = r"\b(?:must|shall|require|required to|include)\s+([a-zA-Z0-9_\- ]{3,25})\b"
    for match in re.finditer(req_pattern, orig_lower):
        clause = match.group(0).strip()
        core_obj = match.group(1).strip()
        if core_obj and core_obj not in opt_lower:
            issues.append(
                ValidationIssue(
                    code="LOST_EXPLICIT_REQUIREMENT",
                    severity=IssueSeverity.ERROR,
                    message=f"Original explicit requirement '{clause}' was omitted from the optimized prompt.",
                    field_or_scope="requirements",
                    evidence=f"Clause: \"{clause}\"",
                )
            )

    return issues


# ---------------------------------------------------------------------------
# Check D: Negative Constraint Preservation
# ---------------------------------------------------------------------------
def check_negative_constraints(orig_prompt: str, opt_prompt: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    orig_lower = orig_prompt.lower()
    opt_lower = opt_prompt.lower()

    neg_patterns = [
        r"\b(?:do not|don't|must not|avoid|never|without|prohibit)\s+([a-zA-Z0-9_\- ]{3,35})\b",
    ]

    for pattern in neg_patterns:
        for match in re.finditer(pattern, orig_lower):
            clause = match.group(0).strip()
            target = match.group(1).strip()
            # If the exclusion target is missing in optimized prompt
            if target and target not in opt_lower:
                issues.append(
                    ValidationIssue(
                        code="LOST_NEGATIVE_CONSTRAINT",
                        severity=IssueSeverity.ERROR,
                        message=f"Original negative constraint '{clause}' was dropped in the optimized prompt.",
                        field_or_scope="negative_constraints",
                        evidence=f"Original clause: \"{clause}\"",
                    )
                )

    return issues


# ---------------------------------------------------------------------------
# Check E: Output Format Preservation
# ---------------------------------------------------------------------------
def check_output_format_preservation(ctx: ValidatorContext) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    orig_formats = set(ctx.original_formats)
    opt_formats = set(ctx.optimized_formats)

    opt_lower = ctx.optimized_prompt.lower()
    code_indicators = {"code", "function", "script", "class", "program", "method", "def ", "return "}

    for fmt in orig_formats:
        # If the format was 'code' (from preprocessing), check if the optimized prompt is clearly code-oriented
        if fmt == "code" and any(ind in opt_lower for ind in code_indicators):
            continue
        if fmt not in opt_formats:
            issues.append(
                ValidationIssue(
                    code="LOST_OUTPUT_FORMAT",
                    severity=IssueSeverity.ERROR,
                    message=f"Original requested output format '{fmt}' was removed in the optimized prompt.",
                    field_or_scope="output_format",
                    evidence=f"Original formats: {list(orig_formats)}, Optimized formats: {list(opt_formats)}",
                )
            )

    return issues


# ---------------------------------------------------------------------------
# Check F: Contradiction Detection
# ---------------------------------------------------------------------------
def check_contradictions(opt_prompt: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    opt_lower = opt_prompt.lower()

    # Rule 1: JSON only vs conversational prose/explanation
    has_json_only = any(
        term in opt_lower
        for term in ("json only", "return json only", "pure json", "valid json only")
    )
    has_prose_exp = any(
        term in opt_lower
        for term in (
            "detailed explanation",
            "conversational prose",
            "explain why",
            "step by step explanation",
            "conversational explanation",
            "thought process",
            "walkthrough",
            "in paragraph form",
        )
    ) or (
        "explanation" in opt_lower
        and any(w in opt_lower for w in ("detailed", "conversational", "thought process", "step by step"))
    )
    if has_json_only and has_prose_exp:
        issues.append(
            ValidationIssue(
                code="CONTRADICTORY_FORMAT_DIRECTIVE",
                severity=IssueSeverity.ERROR,
                message="Optimized prompt requires 'JSON only' but simultaneously requests detailed narrative explanation.",
                field_or_scope="contradictions",
                evidence="Found conflicting directives for strict JSON only and explanatory prose.",
            )
        )

    # Rule 2: Code only vs narrative prose
    has_code_only = any(term in opt_lower for term in ("code only", "return only code", "no explanation"))
    has_walkthrough = any(term in opt_lower for term in ("walk through", "detailed walkthrough", "explain your thinking"))
    if has_code_only and has_walkthrough:
        issues.append(
            ValidationIssue(
                code="CONTRADICTORY_CODE_DIRECTIVE",
                severity=IssueSeverity.ERROR,
                message="Optimized prompt demands code only / no explanation but also asks for a detailed walkthrough.",
                field_or_scope="contradictions",
                evidence="Found conflicting directives for code only and explanation.",
            )
        )

    return issues


# ---------------------------------------------------------------------------
# Check G: Unsupported Addition Detection
# ---------------------------------------------------------------------------
def check_unsupported_additions(orig_prompt: str, opt_prompt: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    orig_lower = orig_prompt.lower()
    opt_lower = opt_prompt.lower()

    suspicious_tech = [
        "postgresql", "mysql", "mongodb", "sqlite", "redis",
        "django", "spring boot", "angular",
        "kubernetes", "docker", "aws", "azure", "gcp"
    ]

    for tech in suspicious_tech:
        pattern = rf"\b{re.escape(tech)}\b"
        if re.search(pattern, opt_lower) and not re.search(pattern, orig_lower):
            # Check if imposed as mandatory requirement
            mandate_pattern = rf"\b(?:must use|required to use|deploy on|implement using)\s+{re.escape(tech)}\b"
            if re.search(mandate_pattern, opt_lower):
                issues.append(
                    ValidationIssue(
                        code="UNSUPPORTED_MANDATORY_TECH",
                        severity=IssueSeverity.WARNING,
                        message=f"Optimized prompt introduced mandatory requirement for '{tech}' not requested in original prompt.",
                        field_or_scope="unsupported_additions",
                        evidence=f"Technology '{tech}' imposed without user specification.",
                    )
                )

    return issues


# ---------------------------------------------------------------------------
# Check H: Score Regression
# ---------------------------------------------------------------------------
def check_score_regression(ctx: ValidatorContext) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if ctx.scoring_available:
        if ctx.score_delta <= -MAX_SCORE_REGRESSION_FAIL:
            issues.append(
                ValidationIssue(
                    code="SEVERE_SCORE_REGRESSION",
                    severity=IssueSeverity.ERROR,
                    message=(
                        f"Unacceptable quality regression: Step 13 overall score dropped by "
                        f"{-ctx.score_delta:.1f} points (from {ctx.original_score:.1f} to {ctx.optimized_score:.1f})."
                    ),
                    field_or_scope="scoring",
                    evidence=f"Score delta: {ctx.score_delta:.1f}",
                )
            )
        elif ctx.score_delta <= -MAX_SCORE_REGRESSION_WARN:
            issues.append(
                ValidationIssue(
                    code="MODERATE_SCORE_REGRESSION",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"Noticeable quality score decline: score dropped by "
                        f"{-ctx.score_delta:.1f} points."
                    ),
                    field_or_scope="scoring",
                    evidence=f"Score delta: {ctx.score_delta:.1f}",
                )
            )

    return issues


# ---------------------------------------------------------------------------
# Check I: Length / Bloat Integrity
# ---------------------------------------------------------------------------
def check_length_bloat(ctx: ValidatorContext) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if ctx.expansion_ratio >= MAX_EXPANSION_RATIO_FAIL:
        issues.append(
            ValidationIssue(
                code="EXCESSIVE_LENGTH_BLOAT",
                severity=IssueSeverity.ERROR,
                message=(
                    f"Excessive prompt expansion: prompt expanded by {ctx.expansion_ratio:.1f}x "
                    f"({ctx.original_word_count} -> {ctx.optimized_word_count} words). "
                    "Fails conciseness and token efficiency standards."
                ),
                field_or_scope="length",
                evidence=f"Expansion ratio: {ctx.expansion_ratio:.2f}",
            )
        )
    elif ctx.expansion_ratio >= MAX_EXPANSION_RATIO_WARN:
        issues.append(
            ValidationIssue(
                code="MODERATE_LENGTH_BLOAT",
                severity=IssueSeverity.WARNING,
                message=f"High prompt expansion: prompt expanded by {ctx.expansion_ratio:.1f}x.",
                field_or_scope="length",
                evidence=f"Expansion ratio: {ctx.expansion_ratio:.2f}",
            )
        )
    elif ctx.expansion_ratio <= MAX_TRUNCATION_RATIO_WARN and ctx.original_word_count > 25:
        issues.append(
            ValidationIssue(
                code="EXCESSIVE_TRUNCATION",
                severity=IssueSeverity.WARNING,
                message=(
                    f"Prompt was drastically truncated by {(1 - ctx.expansion_ratio)*100:.0f}%. "
                    "Original context or criteria may have been lost."
                ),
                field_or_scope="length",
                evidence=f"Expansion ratio: {ctx.expansion_ratio:.2f}",
            )
        )

    return issues


# ---------------------------------------------------------------------------
# Check J: Critic Consistency
# ---------------------------------------------------------------------------
def check_critic_consistency(ctx: ValidatorContext) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if ctx.critic_provided:
        # If Critic gave a FAIL decision
        if ctx.critic_decision == "FAIL":
            issues.append(
                ValidationIssue(
                    code="CRITIC_FAIL_UNRESOLVED",
                    severity=IssueSeverity.ERROR,
                    message="Step 17 Critic rejected this optimization. Optimization cannot be validated as safe.",
                    field_or_scope="critic_consistency",
                    evidence=f"Critic decision was FAIL with score {ctx.critic_score}",
                )
            )
        elif ctx.critic_decision == "NEEDS_REVIEW":
            issues.append(
                ValidationIssue(
                    code="CRITIC_FLAGGED_REVIEW",
                    severity=IssueSeverity.WARNING,
                    message="Step 17 Critic flagged this prompt for review. Validator requires clean resolution.",
                    field_or_scope="critic_consistency",
                    evidence=f"Critic decision was NEEDS_REVIEW with score {ctx.critic_score}",
                )
            )

        if ctx.critic_lost_requirements:
            issues.append(
                ValidationIssue(
                    code="CRITIC_REPORTED_LOST_REQUIREMENT",
                    severity=IssueSeverity.ERROR,
                    message=f"Critic identified lost requirements: {', '.join(ctx.critic_lost_requirements)}",
                    field_or_scope="critic_consistency",
                    evidence=f"Lost: {ctx.critic_lost_requirements}",
                )
            )

    return issues


# ---------------------------------------------------------------------------
# Master Runner
# ---------------------------------------------------------------------------
def run_deterministic_validation(ctx: ValidatorContext) -> DeterministicValidationSummary:
    """Run all deterministic validation checks and compute initial safety decision."""
    issues: list[ValidationIssue] = []
    passed_checks: list[str] = []
    failed_checks: list[str] = []

    def _apply_check(name: str, issue_list: list[ValidationIssue]) -> None:
        if issue_list:
            failed_checks.append(name)
            issues.extend(issue_list)
        else:
            passed_checks.append(name)

    _apply_check("schema_field_integrity", check_field_integrity(ctx))
    _apply_check("intent_preservation", check_intent_preservation(ctx))
    _apply_check("requirement_preservation", check_requirement_preservation(ctx.original_prompt, ctx.optimized_prompt))
    _apply_check("negative_constraints", check_negative_constraints(ctx.original_prompt, ctx.optimized_prompt))
    _apply_check("output_format_preservation", check_output_format_preservation(ctx))
    _apply_check("contradiction_detection", check_contradictions(ctx.optimized_prompt))
    _apply_check("unsupported_additions", check_unsupported_additions(ctx.original_prompt, ctx.optimized_prompt))
    _apply_check("score_regression", check_score_regression(ctx))
    _apply_check("length_bloat_integrity", check_length_bloat(ctx))
    _apply_check("critic_consistency", check_critic_consistency(ctx))

    error_count = sum(1 for i in issues if i.severity == IssueSeverity.ERROR)
    warn_count = sum(1 for i in issues if i.severity == IssueSeverity.WARNING)

    # Compute deterministic safety score
    base_safety = 100.0 - (error_count * 35.0) - (warn_count * 10.0)
    if ctx.scoring_available and ctx.score_delta < 0:
        base_safety += ctx.score_delta  # subtract regression
    safety_score = max(0.0, min(100.0, base_safety))

    # Decision logic:
    # Any ERROR -> FAIL
    # Any Critic FAIL -> FAIL
    # Multiple warnings or borderline safety -> NEEDS_REVIEW
    # Clean check -> PASS
    if error_count > 0 or safety_score < 50.0:
        decision = ValidatorDecision.FAIL
    elif warn_count > 0 or safety_score < 75.0:
        decision = ValidatorDecision.NEEDS_REVIEW
    else:
        decision = ValidatorDecision.PASS

    return DeterministicValidationSummary(
        issues=issues,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        decision=decision,
        safety_score=round(safety_score, 1),
    )
