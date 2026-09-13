"""Scoring service for PromptLens (Step 13).

Orchestrates running and fusing all upstream signals:
- Step 8: Preprocessing
- Step 9: spaCy NLP
- Step 10: Intent Classification (as supporting context)
- Step 11: Embeddings (as supporting metadata)
- Step 12: Quality ML
"""
from app.schemas.scoring import (
    EmbeddingSummary,
    IntentSummary,
    OverallScoreResult,
    QualityDimensionResult,
    ScoreResponse,
    ScoringFindingResult,
    ScoringMetadata,
    SignalResult,
)
from app.scoring.fusion import fuse_signals
from app.services.embedding_service import get_embedding_service
from app.services.intent_service import classify_prompt
from app.services.nlp_service import analyze_prompt_nlp
from app.services.preprocessing_service import preprocess_prompt
from app.services.quality_service import predict_quality


def score_prompt(prompt: str) -> ScoreResponse:
    """Execute complete scoring and fusion for a prompt."""
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty or whitespace-only")

    # 1. Step 8 Preprocessing
    prep_result = preprocess_prompt(prompt)

    # 2. Step 9 NLP analysis
    nlp_result = analyze_prompt_nlp(prompt)

    # 3. Step 12 Quality ML prediction
    quality_result = predict_quality(prompt)

    # 4. Optional Step 10 Intent classification (supporting context)
    intent_summary: IntentSummary | None = None
    try:
        intent_resp = classify_prompt(prompt)
        intent_summary = IntentSummary(
            intent=intent_resp.intent.value,
            confidence=intent_resp.confidence,
            model=intent_resp.model.name,
        )
    except Exception:
        intent_summary = None

    # 5. Optional Step 11 Embeddings metadata (supporting context)
    embedding_summary: EmbeddingSummary | None = None
    try:
        emb_service = get_embedding_service()
        embedding_summary = EmbeddingSummary(
            dimension=emb_service.metadata.dimension,
            normalized=emb_service.metadata.normalized,
            model=emb_service.metadata.name,
            available=True,
        )
    except Exception:
        embedding_summary = None

    # 6. Central Fusion
    fused = fuse_signals(
        prompt=prompt,
        prep=prep_result,
        nlp=nlp_result,
        quality_pred=quality_result,
    )

    # 7. Convert to response schemas
    dim_results = {
        name: QualityDimensionResult(
            score=eval_data.score,
            status=eval_data.status,
            reason=eval_data.reason,
            recommendation=eval_data.recommendation,
        )
        for name, eval_data in fused.dimensions.items()
    }

    signal_results = {
        name: SignalResult(
            source=sig.source,
            raw_score=sig.raw_score,
            configured_weight=sig.configured_weight,
            effective_weight=sig.effective_weight,
            available=sig.available,
            explanation=sig.explanation,
        )
        for name, sig in fused.signals.items()
    }

    finding_results = [
        ScoringFindingResult(
            type=f.type,
            severity=f.severity.value,
            dimension=f.dimension,
            message=f.message,
        )
        for f in fused.findings
    ]

    available_signal_count = sum(1 for s in fused.signals.values() if s.available)

    return ScoreResponse(
        overall_score=OverallScoreResult(
            score=fused.overall_score.score,
            category=fused.overall_score.category,
            status=fused.overall_score.status,
            explanation=fused.overall_score.explanation,
        ),
        dimensions=dim_results,
        signals=signal_results,
        intent=intent_summary,
        embeddings=embedding_summary,
        findings=finding_results,
        recommendations=fused.recommendations,
        metadata=ScoringMetadata(
            scoring_version=fused.scoring_version,
            llm_available=False,
            signals_evaluated=available_signal_count,
        ),
    )
