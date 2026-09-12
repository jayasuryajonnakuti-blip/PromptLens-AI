from fastapi import FastAPI

from app.api.embeddings import router as embeddings_router
from app.api.intent import router as intent_router
from app.api.nlp import router as nlp_router
from app.api.preprocessing import router as preprocessing_router

app = FastAPI(title="PromptLens AI backend")
app.include_router(preprocessing_router)
app.include_router(nlp_router)
app.include_router(intent_router)
app.include_router(embeddings_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "PromptLens AI backend"}
