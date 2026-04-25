import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    AgentToolCallOut,
    AnalyzeRequest,
    AnalyzeResponse,
    BusinessProfileOut,
    CompetitorOut,
    ContentAgentOut,
    KeywordOut,
    TopicClusterOut,
)
from app.core.database import async_session, get_db
from app.models.tables import Competitor, CrawledPage, Keyword, Submission
from app.services.content_agent import run_content_agent
from app.services.pioneer_finetuning import collect_training_samples
from app.services.pipeline import run_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/intent-model/status")
async def intent_model_status(db: AsyncSession = Depends(get_db)):
    from app.services.pioneer_finetuning import get_training_sample_count, get_latest_trained_model, MIN_SAMPLES_TO_TRAIN
    sample_count = await get_training_sample_count(db)
    trained_model = await get_latest_trained_model()
    return {
        "training_samples": sample_count,
        "min_samples_required": MIN_SAMPLES_TO_TRAIN,
        "ready_to_train": sample_count >= MIN_SAMPLES_TO_TRAIN,
        "active_model": trained_model or "fastino/gliner2-base-v1 (base)",
    }


@router.post("/intent-model/train")
async def trigger_intent_training(db: AsyncSession = Depends(get_db)):
    from app.services.pioneer_finetuning import maybe_finetune, get_training_sample_count, MIN_SAMPLES_TO_TRAIN
    sample_count = await get_training_sample_count(db)
    if sample_count < MIN_SAMPLES_TO_TRAIN:
        return {
            "status": "insufficient_data",
            "training_samples": sample_count,
            "min_required": MIN_SAMPLES_TO_TRAIN,
        }
    job_id = await maybe_finetune(db)
    return {"status": "training_started" if job_id else "failed", "job_id": job_id}


@router.get("/intent-model/training-job/{job_id}")
async def check_training_job(job_id: str):
    from app.services.pioneer_finetuning import check_finetuning_status
    return await check_finetuning_status(job_id)


async def _run_pipeline_task(
    job_id: uuid.UUID,
    competitor_urls: list[str],
    business_url: str | None,
    business_name: str | None,
    business_category: str | None,
    business_location: str | None,
) -> None:
    try:
        result = await run_pipeline(
            competitor_urls=competitor_urls,
            business_url=business_url,
            business_name=business_name,
            business_category=business_category,
            business_location=business_location,
        )

        agent_result = await run_content_agent(result)

        # Build response JSON
        competitors_out = [
            {
                "url": p.url,
                "domain": p.domain,
                "pages_crawled": len(p.crawl_result.pages) if p.crawl_result else 0,
                "organic_traffic": p.organic_traffic,
                "organic_keywords": p.organic_keywords,
                "ranked_keywords_count": len(p.ranked_keywords),
                "extracted_keywords": [
                    {"keyword": kw["keyword"], "score": kw["score"], "method": kw["method"]}
                    for kw in p.extracted_keywords[:30]
                ],
                "top_pages": p.top_pages[:10],
            }
            for p in result.competitors
        ]

        clusters_out = [
            {
                "id": c.id,
                "label": c.label,
                "keywords": c.keywords,
                "total_search_volume": c.total_search_volume,
                "avg_keyword_difficulty": c.avg_keyword_difficulty,
                "avg_cpc": c.avg_cpc,
                "competitor_coverage": c.competitor_coverage,
                "opportunity_score": c.opportunity_score,
                "keyword_metrics": c.keyword_metrics,
            }
            for c in result.topic_clusters[:20]
        ]

        content_out = {
            "full_response": agent_result.full_response,
            "thinking_blocks": agent_result.thinking_blocks,
            "tool_calls": [
                {"name": tc["name"], "input": tc["input"], "output_preview": tc["output_preview"]}
                for tc in agent_result.tool_calls
            ],
            "total_input_tokens": agent_result.total_input_tokens,
            "total_output_tokens": agent_result.total_output_tokens,
        }

        business_out = {
            "url": result.business.url,
            "domain": result.business.domain,
            "name": result.business.name,
            "organic_traffic": result.business.organic_traffic,
            "organic_keywords": result.business.organic_keywords,
            "ranked_keywords_count": len(result.business.ranked_keywords),
        }

        result_json = {
            "business": business_out,
            "total_keywords_found": result.total_keywords_found,
            "total_clusters": result.total_clusters,
            "competitors": competitors_out,
            "topic_clusters": clusters_out,
            "content": content_out,
        }

        # Persist to DB
        async with async_session() as db:
            submission = await db.get(Submission, job_id)
            if not submission:
                logger.error("Submission %s not found", job_id)
                return

            for profile in result.competitors:
                comp = Competitor(submission_id=submission.id, url=profile.url)
                db.add(comp)
                await db.flush()

                if profile.crawl_result:
                    for page in profile.crawl_result.pages:
                        db.add(CrawledPage(
                            competitor_id=comp.id,
                            url=page.url,
                            title=page.title,
                            full_text=page.full_text,
                            headings=page.headings,
                            page_metadata=page.metadata,
                            schema_org=page.schema_org,
                            raw_content=page.raw_content,
                        ))

                for kw_data in profile.extracted_keywords:
                    db.add(Keyword(
                        competitor_id=comp.id,
                        keyword=kw_data["keyword"],
                        score=kw_data["score"],
                        method=kw_data["method"],
                    ))

            await collect_training_samples(result.all_keyword_metrics, db)

            submission.status = "completed"
            submission.result_json = result_json
            await db.commit()

        logger.info("Job %s completed", job_id)

    except Exception as e:
        logger.exception("Job %s failed: %s", job_id, e)
        async with async_session() as db:
            submission = await db.get(Submission, job_id)
            if submission:
                submission.status = "failed"
                submission.error = str(e)
                await db.commit()


@router.post("/analyze")
async def start_analysis(request: AnalyzeRequest):
    business_url = str(request.business_url) if request.business_url else None
    competitor_urls = [str(u) for u in request.competitor_urls]

    # Create submission record
    async with async_session() as db:
        submission = Submission(status="processing")
        db.add(submission)
        await db.commit()
        job_id = submission.id
        created_at = submission.created_at

    # Fire off the pipeline in the background
    asyncio.create_task(_run_pipeline_task(
        job_id=job_id,
        competitor_urls=competitor_urls,
        business_url=business_url,
        business_name=request.business_name,
        business_category=request.business_category,
        business_location=request.business_location,
    ))

    return {
        "job_id": str(job_id),
        "status": "processing",
        "created_at": created_at.isoformat(),
    }


@router.get("/analyze/{job_id}")
async def get_analysis(job_id: str):
    async with async_session() as db:
        submission = await db.get(Submission, uuid.UUID(job_id))
        if not submission:
            return {"error": "Job not found"}, 404

        response = {
            "job_id": str(submission.id),
            "status": submission.status,
            "created_at": submission.created_at.isoformat(),
        }

        if submission.status == "failed":
            response["error"] = submission.error

        if submission.status == "completed" and submission.result_json:
            response.update(submission.result_json)

        return response
