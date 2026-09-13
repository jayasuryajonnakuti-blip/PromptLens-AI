from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI

from app.db.session import init_db
from app.api.embeddings import router as embeddings_router
from app.api.intent import router as intent_router
from app.api.nlp import router as nlp_router
from app.api.preprocessing import router as preprocessing_router
from app.api.quality import router as quality_router
from app.api.scoring import router as scoring_router
from app.api.llm import router as llm_router
from app.api.analyzer import router as analyzer_router
from app.api.optimizer import router as optimizer_router
from app.api.critic import router as critic_router
from app.api.validator import router as validator_router
from app.api.agent import router as agent_router
from app.api.runs import router as runs_router
from fastapi.middleware.cors import CORSMiddleware

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan managing database initialization."""
    try:
        init_db()
        LOGGER.info("PromptLens database initialized at startup.")
    except Exception as exc:
        LOGGER.error("Failed to initialize database during startup: %s", exc, exc_info=True)
        raise
    yield


app = FastAPI(title="PromptLens AI backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(preprocessing_router)
app.include_router(nlp_router)
app.include_router(intent_router)
app.include_router(embeddings_router)
app.include_router(quality_router)
app.include_router(scoring_router)
app.include_router(llm_router)
app.include_router(analyzer_router)
app.include_router(optimizer_router)
app.include_router(critic_router)
app.include_router(validator_router)
app.include_router(agent_router)
app.include_router(runs_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "PromptLens AI backend"}
