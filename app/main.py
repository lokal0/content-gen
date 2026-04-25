import asyncio
import logging
from contextlib import asynccontextmanager

import nltk
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.auth import verify_bearer_token
from app.core.database import engine, async_session
from app.models.tables import Base

logger = logging.getLogger(__name__)


async def _daily_finetune_check():
    """Background task: check if we have enough samples to fine-tune Pioneer model."""
    while True:
        await asyncio.sleep(86400)  # 24 hours
        try:
            from app.services.pioneer_finetuning import maybe_finetune, get_training_sample_count, MIN_SAMPLES_TO_TRAIN
            async with async_session() as db:
                count = await get_training_sample_count(db)
                logger.info("[cron] Training samples: %d / %d required", count, MIN_SAMPLES_TO_TRAIN)
                if count >= MIN_SAMPLES_TO_TRAIN:
                    job_id = await maybe_finetune(db)
                    if job_id:
                        logger.info("[cron] Fine-tuning started: %s", job_id)
                    else:
                        logger.info("[cron] Fine-tuning skipped or failed")
        except Exception as e:
            logger.error("[cron] Fine-tuning check failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    finetune_task = asyncio.create_task(_daily_finetune_check())
    logger.info("[cron] Pioneer fine-tuning check scheduled (every 24h)")

    yield

    finetune_task.cancel()
    await engine.dispose()


app = FastAPI(title="Content Gen - Competitor Analysis Engine", lifespan=lifespan, dependencies=[Depends(verify_bearer_token)])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"ok": True}


app.include_router(router, prefix="/api/v1")
