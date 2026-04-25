from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    AgentToolCallOut,
    AnalyzeRequest,
    AnalyzeResponse,
    CompetitorOut,
    ContentAgentOut,
    KeywordOut,
    TopicClusterOut,
)
from app.core.database import get_db
from app.models.tables import Competitor, CrawledPage, Keyword, Submission
from app.services.content_agent import run_content_agent
from app.services.pioneer_finetuning import collect_training_samples
from app.services.pipeline import run_pipeline

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


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_competitors(request: AnalyzeRequest, db: AsyncSession = Depends(get_db)):
    submission = Submission(status="processing")
    db.add(submission)
    await db.flush()

    urls = [str(u) for u in request.urls]

    result = await run_pipeline(urls)

    # Persist to database
    for profile in result.competitors:
        comp = Competitor(submission_id=submission.id, url=profile.url)
        db.add(comp)
        await db.flush()

        if profile.crawl_result:
            for page in profile.crawl_result.pages:
                crawled = CrawledPage(
                    competitor_id=comp.id,
                    url=page.url,
                    title=page.title,
                    full_text=page.full_text,
                    headings=page.headings,
                    page_metadata=page.metadata,
                    schema_org=page.schema_org,
                    raw_content=page.raw_content,
                )
                db.add(crawled)

        for kw_data in profile.extracted_keywords:
            keyword_record = Keyword(
                competitor_id=comp.id,
                keyword=kw_data["keyword"],
                score=kw_data["score"],
                method=kw_data["method"],
            )
            db.add(keyword_record)

    # Collect training samples from DataForSEO intent labels for Pioneer fine-tuning
    await collect_training_samples(result.all_keyword_metrics, db)

    # Phase 4-5: Agent produces content briefs and writes articles
    agent_result = await run_content_agent(result)

    submission.status = "completed"
    await db.commit()

    # Build response
    competitors_out = [
        CompetitorOut(
            url=p.url,
            domain=p.domain,
            pages_crawled=len(p.crawl_result.pages) if p.crawl_result else 0,
            organic_traffic=p.organic_traffic,
            organic_keywords=p.organic_keywords,
            ranked_keywords_count=len(p.ranked_keywords),
            extracted_keywords=[
                KeywordOut(keyword=kw["keyword"], score=kw["score"], method=kw["method"])
                for kw in p.extracted_keywords[:30]
            ],
            top_pages=p.top_pages[:10],
        )
        for p in result.competitors
    ]

    clusters_out = [
        TopicClusterOut(
            id=c.id,
            label=c.label,
            keywords=c.keywords,
            total_search_volume=c.total_search_volume,
            avg_keyword_difficulty=c.avg_keyword_difficulty,
            avg_cpc=c.avg_cpc,
            competitor_coverage=c.competitor_coverage,
            opportunity_score=c.opportunity_score,
            keyword_metrics=c.keyword_metrics,
        )
        for c in result.topic_clusters[:20]
    ]

    content_out = ContentAgentOut(
        full_response=agent_result.full_response,
        thinking_blocks=agent_result.thinking_blocks,
        tool_calls=[
            AgentToolCallOut(name=tc["name"], input=tc["input"], output_preview=tc["output_preview"])
            for tc in agent_result.tool_calls
        ],
        total_input_tokens=agent_result.total_input_tokens,
        total_output_tokens=agent_result.total_output_tokens,
    )

    return AnalyzeResponse(
        submission_id=submission.id,
        status=submission.status,
        created_at=submission.created_at,
        total_keywords_found=result.total_keywords_found,
        total_clusters=result.total_clusters,
        competitors=competitors_out,
        topic_clusters=clusters_out,
        content=content_out,
    )
