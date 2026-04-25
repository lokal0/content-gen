from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CompetitorOut,
    CrawledPageOut,
    KeywordOut,
)
from app.core.database import get_db
from app.models.tables import Competitor, CrawledPage, Keyword, Submission
from app.services.crawler import crawl_all_competitors
from app.services.keyword_extractor import extract_all_keywords

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_competitors(request: AnalyzeRequest, db: AsyncSession = Depends(get_db)):
    submission = Submission(status="processing")
    db.add(submission)
    await db.flush()

    competitor_records = {}
    for url in request.urls:
        comp = Competitor(submission_id=submission.id, url=str(url))
        db.add(comp)
        competitor_records[str(url)] = comp
    await db.flush()

    crawl_results = await crawl_all_competitors([str(u) for u in request.urls])

    for cr in crawl_results:
        comp = competitor_records[cr.competitor_url]
        for page in cr.pages:
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
    await db.flush()

    all_keywords = extract_all_keywords(crawl_results)

    for url, keywords in all_keywords.items():
        comp = competitor_records[url]
        for kw in keywords:
            keyword_record = Keyword(
                competitor_id=comp.id,
                keyword=kw.keyword,
                score=kw.score,
                method=kw.method,
            )
            db.add(keyword_record)

    submission.status = "completed"
    await db.commit()

    competitors_out = []
    for cr in crawl_results:
        url = cr.competitor_url
        kws = all_keywords.get(url, [])
        competitors_out.append(CompetitorOut(
            url=url,
            pages_crawled=len(cr.pages),
            keywords=[KeywordOut(keyword=k.keyword, score=k.score, method=k.method) for k in kws],
            pages=[
                CrawledPageOut(
                    url=p.url,
                    title=p.title,
                    headings=p.headings,
                    metadata=p.metadata,
                    schema_org=p.schema_org,
                )
                for p in cr.pages
            ],
        ))

    return AnalyzeResponse(
        submission_id=submission.id,
        status=submission.status,
        created_at=submission.created_at,
        competitors=competitors_out,
    )
