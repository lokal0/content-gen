import io
import json
import logging
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.tables import IntentTrainingSample
from app.services.keyword_classifier import MODEL_ID as BASE_MODEL_ID

logger = logging.getLogger(__name__)

PIONEER_BASE_URL = "https://api.pioneer.ai"
MIN_SAMPLES_TO_TRAIN = 50
DATASET_NAME = "seo-keyword-intent"


def _headers() -> dict[str, str]:
    return {
        "X-API-Key": settings.pioneer_api_key,
        "Content-Type": "application/json",
    }


async def collect_training_samples(
    keyword_metrics: dict[str, dict],
    db: AsyncSession,
) -> int:
    """Collect keywords with DataForSEO-provided intent as training ground truth."""
    count = 0
    for kw, metrics in keyword_metrics.items():
        intent = metrics.get("intent")
        if not intent or intent == "unknown":
            continue

        existing = await db.execute(
            select(IntentTrainingSample).where(IntentTrainingSample.keyword == kw).limit(1)
        )
        if existing.scalar_one_or_none():
            continue

        sample = IntentTrainingSample(
            keyword=kw,
            intent=intent,
            source="dataforseo",
        )
        db.add(sample)
        count += 1

    if count > 0:
        await db.flush()
        logger.info("Collected %d new intent training samples", count)

    return count


async def get_training_sample_count(db: AsyncSession) -> int:
    result = await db.execute(select(func.count(IntentTrainingSample.id)))
    return result.scalar_one()


async def export_training_data(db: AsyncSession) -> list[dict]:
    result = await db.execute(
        select(IntentTrainingSample).order_by(IntentTrainingSample.created_at)
    )
    samples = result.scalars().all()
    return [
        {"text": s.keyword, "label": s.intent}
        for s in samples
    ]


async def upload_dataset(training_data: list[dict]) -> str | None:
    """Upload training data to Pioneer and return dataset ID."""
    if not settings.pioneer_api_key:
        return None

    async with httpx.AsyncClient(headers=_headers(), timeout=60.0) as client:
        # Get presigned upload URL
        r = await client.post(
            f"{PIONEER_BASE_URL}/felix/datasets/upload/url",
            json={
                "dataset_name": DATASET_NAME,
                "dataset_type": "classification",
                "format": "jsonl",
            },
        )
        r.raise_for_status()
        upload_info = r.json()
        presigned_url = upload_info.get("upload_url") or upload_info.get("url")
        dataset_id = upload_info.get("dataset_id") or upload_info.get("id")

        if not presigned_url:
            logger.error("No upload URL returned: %s", upload_info)
            return None

        # Build JSONL content
        jsonl = "\n".join(json.dumps(row) for row in training_data)

        # Upload file
        async with httpx.AsyncClient(timeout=60.0) as upload_client:
            r = await upload_client.put(
                presigned_url,
                content=jsonl.encode(),
                headers={"Content-Type": "application/octet-stream"},
            )
            r.raise_for_status()

        # Trigger processing
        r = await client.post(
            f"{PIONEER_BASE_URL}/felix/datasets/upload/process",
            json={"dataset_id": dataset_id},
        )
        r.raise_for_status()

        logger.info("Uploaded dataset %s with %d samples", dataset_id, len(training_data))
        return dataset_id


async def start_finetuning(dataset_name: str = DATASET_NAME) -> str | None:
    """Start a fine-tuning job on Pioneer. Returns job ID."""
    if not settings.pioneer_api_key:
        return None

    async with httpx.AsyncClient(headers=_headers(), timeout=60.0) as client:
        r = await client.post(
            f"{PIONEER_BASE_URL}/felix/training-jobs",
            json={
                "model_name": "seo-intent-classifier",
                "base_model": BASE_MODEL_ID,
                "datasets": [{"name": dataset_name}],
                "training_type": "lora",
                "nr_epochs": 10,
                "learning_rate": 5e-5,
                "batch_size": 8,
            },
        )
        r.raise_for_status()
        job = r.json()
        job_id = job.get("id") or job.get("job_id")
        logger.info("Started fine-tuning job: %s", job_id)
        return job_id


async def check_finetuning_status(job_id: str) -> dict:
    """Check status of a fine-tuning job."""
    async with httpx.AsyncClient(headers=_headers(), timeout=30.0) as client:
        r = await client.get(f"{PIONEER_BASE_URL}/felix/training-jobs/{job_id}")
        r.raise_for_status()
        return r.json()


async def get_latest_trained_model() -> str | None:
    """Get the most recently trained intent classifier model ID."""
    if not settings.pioneer_api_key:
        return None

    async with httpx.AsyncClient(headers=_headers(), timeout=30.0) as client:
        r = await client.get(f"{PIONEER_BASE_URL}/felix/trained-models")
        r.raise_for_status()
        models = r.json()

        if isinstance(models, dict):
            models = models.get("models", [])

        for model in models:
            name = model.get("name", "") or model.get("model_name", "")
            if "seo-intent" in name:
                return model.get("id") or model.get("model_id")

    return None


async def maybe_finetune(db: AsyncSession) -> str | None:
    """Check if we have enough samples and kick off fine-tuning if so.
    Returns the job ID if training was started, None otherwise."""
    sample_count = await get_training_sample_count(db)
    logger.info("Intent training samples in DB: %d (need %d)", sample_count, MIN_SAMPLES_TO_TRAIN)

    if sample_count < MIN_SAMPLES_TO_TRAIN:
        return None

    training_data = await export_training_data(db)
    dataset_id = await upload_dataset(training_data)
    if not dataset_id:
        return None

    job_id = await start_finetuning()
    return job_id
