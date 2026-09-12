from app.engines.nlp import analyze_prompt as run_nlp_analysis
from app.schemas.nlp import NlpAnalysisResult


def analyze_prompt_nlp(prompt: str) -> NlpAnalysisResult:
    """Run the configured spaCy model without exposing engine details to callers."""
    return run_nlp_analysis(prompt)