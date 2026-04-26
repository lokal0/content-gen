import asyncio
import logging
from dataclasses import dataclass

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

PIONEER_BASE_URL = "https://api.pioneer.ai"
MODEL_ID = "fastino/gliner2-base-v1"
INTENT_CATEGORIES = ["informational", "transactional", "navigational", "commercial"]
BATCH_SIZE = 50
MAX_CONCURRENT = 10

INTENT_WEIGHTS = {
    "transactional": 1.5,
    "commercial": 1.3,
    "informational": 1.0,
    "navigational": 0.5,
}


@dataclass
class ClassifiedKeyword:
    keyword: str
    intent: str
    intent_weight: float


async def _resolve_model_id() -> str:
    """Use fine-tuned model if available, otherwise fall back to base."""
    if not settings.pioneer_api_key:
        return MODEL_ID

    try:
        async with httpx.AsyncClient(
            headers={"X-API-Key": settings.pioneer_api_key},
            timeout=10.0,
        ) as client:
            r = await client.get(f"{PIONEER_BASE_URL}/felix/trained-models")
            r.raise_for_status()
            models = r.json()

            if isinstance(models, dict):
                models = models.get("models", [])

            for model in models:
                name = model.get("name", "") or model.get("model_name", "")
                if "seo-intent" in name:
                    model_id = model.get("id") or model.get("model_id")
                    logger.info("Using fine-tuned model: %s", model_id)
                    return model_id
    except Exception as e:
        logger.warning("Could not check for fine-tuned models: %s", e)

    logger.info("No fine-tuned model found, using base: %s", MODEL_ID)
    return MODEL_ID


async def _classify_one(
    client: httpx.AsyncClient,
    keyword: str,
    model_id: str,
    semaphore: asyncio.Semaphore,
) -> ClassifiedKeyword:
    async with semaphore:
        try:
            r = await client.post(
                f"{PIONEER_BASE_URL}/inference",
                json={
                    "model_id": model_id,
                    "task": "classify_text",
                    "text": keyword,
                    "schema": {"categories": INTENT_CATEGORIES},
                },
            )
            r.raise_for_status()
            intent = r.json()["result"]["category"]
        except Exception as e:
            logger.warning("Pioneer classification failed for %r: %s", keyword, e)
            intent = "informational"

        return ClassifiedKeyword(
            keyword=keyword,
            intent=intent,
            intent_weight=INTENT_WEIGHTS.get(intent, 1.0),
        )


async def classify_keywords(keywords: list[str], job_id: "uuid.UUID | None" = None) -> dict[str, ClassifiedKeyword]:
    import uuid
    if not keywords:
        return {}

    if not settings.pioneer_api_key:
        logger.warning("PIONEER_API_KEY not set, skipping classification")
        return {}

    model_id = await _resolve_model_id()
    logger.info("Classifying %d keywords with Pioneer (%s)", len(keywords), model_id)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async with httpx.AsyncClient(
        headers={
            "X-API-Key": settings.pioneer_api_key,
            "Content-Type": "application/json",
        },
        timeout=30.0,
    ) as client:
        tasks = [_classify_one(client, kw, model_id, semaphore) for kw in keywords]
        results = await asyncio.gather(*tasks)

    classified = {r.keyword: r for r in results}
    intent_counts = {}
    for r in results:
        intent_counts[r.intent] = intent_counts.get(r.intent, 0) + 1
    logger.info("Classification complete: %s", intent_counts)

    if job_id:
        from app.services.event_bus import emit
        for r in results[:10]:
            await emit(job_id, "intent_classified", {
                "keyword": r.keyword,
                "intent": r.intent,
                "weight": r.intent_weight,
            })

    return classified
