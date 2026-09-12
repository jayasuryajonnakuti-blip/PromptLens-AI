from app.engines.preprocessing import preprocess_prompt as run_preprocessing
from app.schemas.preprocessing import PromptPreprocessingResult


def preprocess_prompt(prompt: str) -> PromptPreprocessingResult:
    return run_preprocessing(prompt)
