from contextlib import asynccontextmanager

import nltk
from fastapi import FastAPI

from app.api.routes import router
from app.core.database import engine
from app.models.tables import Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(title="Content Gen - Competitor Analysis Engine", lifespan=lifespan)
app.include_router(router, prefix="/api/v1")
