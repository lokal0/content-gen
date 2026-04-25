import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from app.services import seo_client
from app.services.crawler import CrawlResult, crawl_all_competitors
from app.services.keyword_classifier import ClassifiedKeyword, classify_keywords
from app.services.keyword_extractor import extract_all_keywords
from app.services.topic_clustering import TopicCluster, build_topic_clusters

logger = logging.getLogger(__name__)


@dataclass
class CompetitorProfile:
    url: str
    domain: str = ""
    organic_traffic: int | None = None
    organic_keywords: int | None = None
    ranked_keywords: set[str] = field(default_factory=set)
    top_pages: list[dict] = field(default_factory=list)
    crawl_result: CrawlResult | None = None
    extracted_keywords: list[dict] = field(default_factory=list)


@dataclass
class BusinessProfile:
    url: str | None = None
    domain: str | None = None
    name: str | None = None
    organic_traffic: int | None = None
    organic_keywords: int | None = None
    ranked_keywords: set[str] = field(default_factory=set)

    @property
    def display_name(self) -> str:
        return self.domain or self.name or "your business"


@dataclass
class PipelineResult:
    business: BusinessProfile
    competitors: list[CompetitorProfile]
    topic_clusters: list[TopicCluster]
    all_keyword_metrics: dict[str, dict]
    keyword_intents: dict[str, ClassifiedKeyword] = field(default_factory=dict)
    total_keywords_found: int = 0
    total_clusters: int = 0


def _extract_domain(url: str) -> str:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    domain = parsed.hostname or parsed.path
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


async def _gather_competitor_data(profiles: list[CompetitorProfile]) -> None:
    async def fetch_one(profile: CompetitorProfile):
        try:
            overview = await seo_client.domain_overview(profile.domain)
            profile.organic_traffic = overview.organic_traffic
            profile.organic_keywords = overview.organic_keywords
            profile.top_pages = overview.pages[:50]
            for kw in overview.keywords:
                profile.ranked_keywords.add(kw.get("keyword", "").lower())
            logger.info("%s: %d ranked keywords from overview", profile.domain, len(profile.ranked_keywords))
        except Exception as e:
            logger.warning("domain_overview failed for %s: %s", profile.domain, e)

        try:
            suggestions = await seo_client.domain_suggestions(profile.domain)
            for kw in suggestions:
                profile.ranked_keywords.add(kw.get("keyword", "").lower())
            logger.info("%s: %d total ranked keywords after suggestions", profile.domain, len(profile.ranked_keywords))
        except Exception as e:
            logger.warning("domain_suggestions failed for %s: %s", profile.domain, e)

    await asyncio.gather(*[fetch_one(p) for p in profiles])


async def _gather_business_data(business: BusinessProfile) -> None:
    try:
        overview = await seo_client.domain_overview(business.domain)
        business.organic_traffic = overview.organic_traffic
        business.organic_keywords = overview.organic_keywords
        for kw in overview.keywords:
            business.ranked_keywords.add(kw.get("keyword", "").lower())
        logger.info("Business %s: %d ranked keywords", business.domain, len(business.ranked_keywords))
    except Exception as e:
        logger.warning("domain_overview failed for business %s: %s", business.domain, e)

    try:
        suggestions = await seo_client.domain_suggestions(business.domain)
        for kw in suggestions:
            business.ranked_keywords.add(kw.get("keyword", "").lower())
        logger.info("Business %s: %d total ranked keywords", business.domain, len(business.ranked_keywords))
    except Exception as e:
        logger.warning("domain_suggestions failed for business %s: %s", business.domain, e)


async def _enrich_keywords(keywords: list[str]) -> dict[str, dict]:
    metrics = {}
    for i in range(0, len(keywords), 700):
        batch = keywords[i : i + 700]
        try:
            results = await seo_client.keyword_overview(batch)
            for kw in results:
                metrics[kw.keyword.lower()] = {
                    "keyword": kw.keyword,
                    "searchVolume": kw.search_volume,
                    "cpc": kw.cpc,
                    "competition": kw.competition,
                    "keywordDifficulty": kw.keyword_difficulty,
                    "intent": kw.intent,
                }
        except Exception as e:
            logger.warning("keyword_overview failed for batch %d: %s", i, e)
    return metrics


async def _discover_competitors(
    business_name: str | None,
    business_category: str | None,
    business_location: str | None,
    limit: int = 5,
) -> list[str]:
    """Use Tavily search to find competitor website URLs when none are provided."""
    from tavily import TavilyClient
    from app.core.config import settings

    if not settings.tavily_api_key:
        return []

    parts = []
    if business_category:
        parts.append(f"best {business_category}")
    elif business_name:
        parts.append(f"businesses similar to {business_name}")
    else:
        return []

    if business_location:
        parts.append(f"in {business_location}")

    query = " ".join(parts) + " website"
    logger.info("Discovering competitors via Tavily: %s", query)

    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.search(query=query, max_results=limit * 2, search_depth="basic"),
        )

        urls = []
        seen_domains = set()
        skip_domains = {"google.com", "yelp.com", "tripadvisor.com", "facebook.com", "instagram.com", "maps.google.com", "wikipedia.org"}
        biz_domain = _extract_domain(f"https://{business_name}") if business_name else ""

        for r in response.get("results", []):
            url = r.get("url", "")
            if not url:
                continue
            domain = _extract_domain(url)
            if domain in seen_domains or domain in skip_domains:
                continue
            if biz_domain and domain == biz_domain:
                continue
            seen_domains.add(domain)
            urls.append(f"https://{domain}")
            if len(urls) >= limit:
                break

        logger.info("Discovered %d competitor URLs: %s", len(urls), urls)
        return urls
    except Exception as e:
        logger.warning("Competitor discovery failed: %s", e)
        return []


async def run_pipeline(
    competitor_urls: list[str],
    business_url: str | None = None,
    business_name: str | None = None,
    business_category: str | None = None,
    business_location: str | None = None,
    job_id: "uuid.UUID | None" = None,
) -> PipelineResult:
    from app.services.progress import update_progress
    import uuid as _uuid

    async def progress(stage: str, detail: str | None = None):
        if job_id:
            await update_progress(job_id, stage, detail)

    business = BusinessProfile(name=business_name)
    if business_url:
        business.url = business_url
        business.domain = _extract_domain(business_url)

    if not competitor_urls:
        await progress("discovering_competitors", f"Searching for competitors of {business_name or 'business'}")
        competitor_urls = await _discover_competitors(
            business_name=business_name,
            business_category=business_category,
            business_location=business_location,
        )
        if not competitor_urls:
            raise ValueError("No competitor URLs provided and auto-discovery found none. Provide at least one competitor URL.")
        await progress("discovering_competitors", f"Found {len(competitor_urls)} competitors")

    profiles = [
        CompetitorProfile(url=url, domain=_extract_domain(url))
        for url in competitor_urls
    ]

    await progress("crawling", f"Crawling {len(competitor_urls)} competitor websites")
    await progress("gathering_seo_data", f"Fetching SEO data for {len(profiles)} domains")

    gather_tasks = [
        crawl_all_competitors(competitor_urls),
        _gather_competitor_data(profiles),
    ]
    if business.domain:
        gather_tasks.append(_gather_business_data(business))

    results = await asyncio.gather(*gather_tasks)
    crawl_results = results[0]

    await progress("extracting_keywords", "Extracting keywords from crawled content")
    extracted = extract_all_keywords(crawl_results)
    for cr in crawl_results:
        for profile in profiles:
            if profile.url == cr.competitor_url:
                profile.crawl_result = cr
                profile.extracted_keywords = [
                    {"keyword": kw.keyword, "score": kw.score, "method": kw.method}
                    for kw in extracted.get(cr.competitor_url, [])
                ]
                break

    all_keywords_set: set[str] = set()
    for profile in profiles:
        all_keywords_set.update(profile.ranked_keywords)
        for kw in profile.extracted_keywords:
            all_keywords_set.add(kw["keyword"].lower())

    all_keywords = list(all_keywords_set)
    await progress("extracting_keywords", f"Found {len(all_keywords)} unique keywords")

    await progress("enriching_keywords", f"Enriching {len(all_keywords)} keywords with search metrics")
    keyword_metrics = await _enrich_keywords(all_keywords)
    await progress("enriching_keywords", f"Got metrics for {len(keyword_metrics)} keywords")

    await progress("classifying_intent", f"Classifying {len(all_keywords)} keywords with Pioneer GLiNER2")
    keyword_intents = await classify_keywords(all_keywords)
    await progress("classifying_intent", f"Classified {len(keyword_intents)} keywords by intent")

    await progress("embedding_keywords", f"Embedding {len(all_keywords)} keywords with Gemini")
    keywords_to_cluster = [kw for kw in all_keywords if kw in keyword_metrics]
    competitor_keywords = {p.url: p.ranked_keywords for p in profiles}

    topic_clusters = await build_topic_clusters(
        keywords=keywords_to_cluster,
        keyword_metrics=keyword_metrics,
        competitor_keywords=competitor_keywords,
        keyword_intents=keyword_intents,
        business_keywords=business.ranked_keywords,
        min_cluster_size=3,
    )
    await progress("clustering", f"Found {len(topic_clusters)} topic clusters")

    return PipelineResult(
        business=business,
        competitors=profiles,
        topic_clusters=topic_clusters,
        all_keyword_metrics=keyword_metrics,
        keyword_intents=keyword_intents,
        total_keywords_found=len(all_keywords),
        total_clusters=len(topic_clusters),
    )
