"""Deterministic evaluation for the 12 PromptLens quality dimensions (Step 13).

Dimensions:
1. clarity
2. specificity
3. context
4. goal_definition
5. constraints
6. output_format
7. role_persona
8. audience
9. ambiguity (inverted: high score = low ambiguity)
10. completeness
11. actionability
12. consistency

Each dimension returns a dict containing:
- score: float in [0.0, 100.0]
- status: QualityLabel (POOR, FAIR, GOOD, STRONG, EXCELLENT)
- reason: str
- recommendation: str
"""
from dataclasses import dataclass
from typing import Any

from app.ml.quality.categories import QualityLabel, score_to_label
from app.schemas.nlp import NlpAnalysisResult
from app.schemas.preprocessing import PromptPreprocessingResult

DIMENSION_NAMES: tuple[str, ...] = (
    "clarity",
    "specificity",
    "context",
    "goal_definition",
    "constraints",
    "output_format",
    "role_persona",
    "audience",
    "ambiguity",
    "completeness",
    "actionability",
    "consistency",
)


@dataclass(frozen=True)
class DimensionEvaluation:
    name: str
    score: float
    status: QualityLabel
    reason: str
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "status": self.status.value,
            "reason": self.reason,
            "recommendation": self.recommendation,
        }


def evaluate_dimensions(
    prep: PromptPreprocessingResult,
    nlp: NlpAnalysisResult,
    ml_quality_score: float | None = None,
) -> dict[str, DimensionEvaluation]:
    """Compute all 12 dimensions deterministically using Steps 8-12 signals."""
    if prep.flags.is_empty:
        return {
            name: DimensionEvaluation(
                name=name,
                score=0.0,
                status=QualityLabel.POOR,
                reason="Prompt is empty or contains only whitespace.",
                recommendation="Provide a clear, detailed prompt describing your desired task.",
            )
            for name in DIMENSION_NAMES
        }

    is_near_empty = prep.flags.is_near_empty
    word_count = prep.text.word_count

    dims: dict[str, DimensionEvaluation] = {}

    # -------------------------------------------------------------------------
    # 1. CLARITY
    # -------------------------------------------------------------------------
    clarity_score = 65.0
    if is_near_empty:
        clarity_score = 25.0
    else:
        avg_len = nlp.sentence_statistics.average_sentence_length
        if 8.0 <= avg_len <= 25.0:
            clarity_score += 15.0
        elif avg_len > 40.0:
            clarity_score -= 15.0
        if len(prep.repeated_phrases) > 2:
            clarity_score -= 15.0
        if prep.text.vocabulary_diversity >= 0.70:
            clarity_score += 10.0

    clarity_score = _clamp(clarity_score)
    dims["clarity"] = DimensionEvaluation(
        name="clarity",
        score=clarity_score,
        status=score_to_label(clarity_score),
        reason=(
            "Sentence structure is clear, concise, and easy to parse."
            if clarity_score >= 60.0
            else "Prompt structure contains convoluted or excessively repetitive phrasing."
        ),
        recommendation=(
            "Maintain concise sentence phrasing and avoid unnecessary repetition."
            if clarity_score >= 60.0
            else "Break complex compound sentences into shorter, clearer statements."
        ),
    )

    # -------------------------------------------------------------------------
    # 2. SPECIFICITY
    # -------------------------------------------------------------------------
    spec_score = 45.0
    if is_near_empty:
        spec_score = 15.0
    else:
        if nlp.technical_terms.technical_term_count_estimated >= 1:
            spec_score += min(20.0, nlp.technical_terms.technical_term_count_estimated * 10.0)
        if prep.placeholders:
            spec_score += 15.0
        if nlp.number_token_count > 0:
            spec_score += 10.0
        if prep.structure.has_code_block:
            spec_score += 15.0
        if word_count >= 20:
            spec_score += 10.0

    spec_score = _clamp(spec_score)
    dims["specificity"] = DimensionEvaluation(
        name="specificity",
        score=spec_score,
        status=score_to_label(spec_score),
        reason=(
            "Includes specific parameters, technical terminology, and concrete requirements."
            if spec_score >= 60.0
            else "Lacks concrete details, specific parameters, or domain terminology."
        ),
        recommendation=(
            "Add specific input data examples, parameter values, or exact metrics."
            if spec_score < 75.0
            else "Specificity is well-established; maintain explicit parameter bounds."
        ),
    )

    # -------------------------------------------------------------------------
    # 3. CONTEXT
    # -------------------------------------------------------------------------
    ctx_score = 50.0
    if is_near_empty:
        ctx_score = 10.0
    else:
        if "context" in prep.sections:
            ctx_score += 25.0
        if len(nlp.named_entities) >= 1:
            ctx_score += 10.0
        if prep.text.paragraph_count >= 2:
            ctx_score += 15.0
        if word_count >= 30:
            ctx_score += 10.0
        elif word_count < 8:
            ctx_score -= 20.0

    ctx_score = _clamp(ctx_score)
    dims["context"] = DimensionEvaluation(
        name="context",
        score=ctx_score,
        status=score_to_label(ctx_score),
        reason=(
            "Provides sufficient situational background and context for the task."
            if ctx_score >= 60.0
            else "Minimal or no background context provided."
        ),
        recommendation=(
            "Include relevant background context, system assumptions, or environmental details."
            if ctx_score < 70.0
            else "Context is adequate for informed task execution."
        ),
    )

    # -------------------------------------------------------------------------
    # 4. GOAL_DEFINITION
    # -------------------------------------------------------------------------
    goal_score = 50.0
    if is_near_empty:
        goal_score = 20.0
    else:
        if "task_goal" in prep.sections:
            goal_score += 25.0
        if prep.structure.instruction_count_estimated >= 1:
            goal_score += 15.0
        if prep.structure.question_count > 0:
            goal_score += 10.0
        if nlp.pos_counts.verbs >= 1:
            goal_score += 10.0
        else:
            goal_score -= 25.0

    goal_score = _clamp(goal_score)
    dims["goal_definition"] = DimensionEvaluation(
        name="goal_definition",
        score=goal_score,
        status=score_to_label(goal_score),
        reason=(
            "Task objective is explicitly stated with clear actionable verbs."
            if goal_score >= 60.0
            else "Goal is vague or lacks a clear action verb."
        ),
        recommendation=(
            "State the primary objective explicitly using an active verb (e.g., 'Analyze', 'Generate', 'Compare')."
            if goal_score < 70.0
            else "Goal definition is clear and targeted."
        ),
    )

    # -------------------------------------------------------------------------
    # 5. CONSTRAINTS
    # -------------------------------------------------------------------------
    constr_score = 45.0
    if is_near_empty:
        constr_score = 15.0
    else:
        if "constraints" in prep.sections:
            constr_score += 35.0
        # Check for constraint keywords in lower prompt
        lowered = prep.text.character_count > 0 and prep.text.word_count > 0
        if any(w in prep.sections for w in ["constraints"]):
            constr_score += 10.0
        if prep.structure.bullet_count >= 2:
            constr_score += 10.0

    constr_score = _clamp(constr_score)
    dims["constraints"] = DimensionEvaluation(
        name="constraints",
        score=constr_score,
        status=score_to_label(constr_score),
        reason=(
            "Specifies explicit limitations, negative constraints, or boundary conditions."
            if constr_score >= 60.0
            else "Does not define operational constraints, negative requirements, or boundaries."
        ),
        recommendation=(
            "Define explicit constraints (e.g. length limits, forbidden methods, or required libraries)."
            if constr_score < 70.0
            else "Constraint boundaries are adequately established."
        ),
    )

    # -------------------------------------------------------------------------
    # 6. OUTPUT_FORMAT
    # -------------------------------------------------------------------------
    fmt_score = 40.0
    if is_near_empty:
        fmt_score = 10.0
    elif prep.output_formats:
        fmt_score = 75.0 + min(20.0, (len(prep.output_formats) - 1) * 10.0)
    elif "output_format" in prep.sections:
        fmt_score = 70.0
    elif prep.structure.has_code_block:
        fmt_score = 65.0
    elif prep.structure.bullet_count > 0:
        fmt_score = 60.0

    fmt_score = _clamp(fmt_score)
    dims["output_format"] = DimensionEvaluation(
        name="output_format",
        score=fmt_score,
        status=score_to_label(fmt_score),
        reason=(
            f"Explicit output format requested: {', '.join(prep.output_formats)}."
            if prep.output_formats
            else "Expected output structure or schema is not specified."
        ),
        recommendation=(
            "Specify the desired output format (e.g., JSON, markdown table, bullet list, or CSV)."
            if fmt_score < 65.0
            else "Output format requirement is clearly defined."
        ),
    )

    # -------------------------------------------------------------------------
    # 7. ROLE_PERSONA
    # -------------------------------------------------------------------------
    role_score = 50.0
    if is_near_empty:
        role_score = 20.0
    elif "role_persona" in prep.sections:
        role_score = 85.0

    role_score = _clamp(role_score)
    dims["role_persona"] = DimensionEvaluation(
        name="role_persona",
        score=role_score,
        status=score_to_label(role_score),
        reason=(
            "Specifies an explicit role, persona, or domain expertise level."
            if role_score >= 75.0
            else "No explicit role or persona defined (acceptable for direct tasks)."
        ),
        recommendation=(
            "Adopt an explicit expert role (e.g., 'Act as a principal cloud architect') to tune domain perspective."
            if role_score < 70.0
            else "Role framing is well-defined."
        ),
    )

    # -------------------------------------------------------------------------
    # 8. AUDIENCE
    # -------------------------------------------------------------------------
    aud_score = 50.0
    if is_near_empty:
        aud_score = 20.0
    elif "audience" in prep.sections:
        aud_score = 85.0

    aud_score = _clamp(aud_score)
    dims["audience"] = DimensionEvaluation(
        name="audience",
        score=aud_score,
        status=score_to_label(aud_score),
        reason=(
            "Identifies the target reader, proficiency level, or stakeholder group."
            if aud_score >= 75.0
            else "Target audience or reader proficiency level is implicit or unspecified."
        ),
        recommendation=(
            "Clarify who the response is for (e.g., 'for beginner Python students' or 'for executive stakeholders')."
            if aud_score < 70.0
            else "Audience targeting is well-specified."
        ),
    )

    # -------------------------------------------------------------------------
    # 9. AMBIGUITY (Inverted: high score = low ambiguity)
    # -------------------------------------------------------------------------
    amb_score = 70.0
    if is_near_empty:
        amb_score = 20.0
    else:
        if nlp.pos_counts.verbs == 0:
            amb_score -= 30.0
        if len(prep.repeated_phrases) > 2:
            amb_score -= 15.0
        if word_count < 6:
            amb_score -= 25.0
        if prep.structure.instruction_count_estimated >= 1 and prep.output_formats:
            amb_score += 15.0

    amb_score = _clamp(amb_score)
    dims["ambiguity"] = DimensionEvaluation(
        name="ambiguity",
        score=amb_score,
        status=score_to_label(amb_score),
        reason=(
            "Instructions are direct and leave minimal room for subjective misinterpretation."
            if amb_score >= 60.0
            else "High ambiguity detected; task lacks clear parameters or action verbs."
        ),
        recommendation=(
            "Disambiguate the task by providing explicit inputs, scope boundaries, and measurable deliverables."
            if amb_score < 70.0
            else "Ambiguity is low; instructions are sufficiently well-specified."
        ),
    )

    # -------------------------------------------------------------------------
    # 10. COMPLETENESS
    # -------------------------------------------------------------------------
    comp_score = 45.0
    if is_near_empty:
        comp_score = 15.0
    else:
        # Score based on multi-section structural presence
        sec_count = len(prep.sections)
        comp_score += min(35.0, sec_count * 10.0)
        if prep.output_formats:
            comp_score += 10.0
        if word_count >= 25:
            comp_score += 10.0

    comp_score = _clamp(comp_score)
    dims["completeness"] = DimensionEvaluation(
        name="completeness",
        score=comp_score,
        status=score_to_label(comp_score),
        reason=(
            "Includes multiple structural components (goal, context, requirements, and format)."
            if comp_score >= 60.0
            else "Incomplete task specification; missing supporting parameters or constraints."
        ),
        recommendation=(
            "Provide all required information: goal, background context, format, and evaluation criteria."
            if comp_score < 70.0
            else "Task specification is structurally comprehensive."
        ),
    )

    # -------------------------------------------------------------------------
    # 11. ACTIONABILITY
    # -------------------------------------------------------------------------
    act_score = 55.0
    if is_near_empty:
        act_score = 20.0
    else:
        if prep.structure.instruction_count_estimated >= 2:
            act_score += 20.0
        elif prep.structure.instruction_count_estimated == 1:
            act_score += 10.0
        else:
            act_score -= 15.0

        if nlp.verb_statistics.verb_density >= 0.12:
            act_score += 10.0
        if prep.output_formats:
            act_score += 10.0

    act_score = _clamp(act_score)
    dims["actionability"] = DimensionEvaluation(
        name="actionability",
        score=act_score,
        status=score_to_label(act_score),
        reason=(
            "Instructions are executable with directly actionable verbs and clear deliverables."
            if act_score >= 60.0
            else "Instructions are passive, declarative, or lack direct action verbs."
        ),
        recommendation=(
            "Frame instructions as executable tasks with clear verbs and concrete deliverables."
            if act_score < 70.0
            else "Actionability is strong."
        ),
    )

    # -------------------------------------------------------------------------
    # 12. CONSISTENCY
    # -------------------------------------------------------------------------
    cons_score = 75.0
    if is_near_empty:
        cons_score = 40.0
    else:
        if len(prep.repeated_phrases) > 3:
            cons_score -= 20.0
        elif len(prep.repeated_phrases) > 1:
            cons_score -= 10.0

        # Check for excessive contradictory format indicators
        if "JSON" in prep.output_formats and "plain_text" in prep.output_formats:
            cons_score -= 15.0

    cons_score = _clamp(cons_score)
    dims["consistency"] = DimensionEvaluation(
        name="consistency",
        score=cons_score,
        status=score_to_label(cons_score),
        reason=(
            "Prompt instructions are logically consistent without obvious conflicting requirements."
            if cons_score >= 65.0
            else "Potential conflicting requirements or excessive repetition detected."
        ),
        recommendation=(
            "Eliminate redundant repetition and reconcile conflicting output or structural demands."
            if cons_score < 70.0
            else "Consistency is sound."
        ),
    )

    return dims


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, round(float(value), 2)))
