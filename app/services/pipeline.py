import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from app.services import seo_client
from app.services.crawler import CrawlResult, crawl_all_competitors
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
class PipelineResult:
    competitors: list[CompetitorProfile]
    topic_clusters: list[TopicCluster]
    all_keyword_metrics: dict[str, dict]
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


async def run_pipeline(urls: list[str]) -> PipelineResult:
    profiles = [
        CompetitorProfile(url=url, domain=_extract_domain(url))
        for url in urls
    ]

    # Phase 1a: Crawl + SEO data in parallel
    logger.info("Phase 1: Crawling competitors and gathering SEO data")
    crawl_results, _ = await asyncio.gather(
        crawl_all_competitors(urls),
        _gather_competitor_data(profiles),
    )

    # Phase 1b: Extract keywords from crawled content
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

    # Phase 1c: Merge and dedupe all keywords
    all_keywords_set: set[str] = set()
    for profile in profiles:
        all_keywords_set.update(profile.ranked_keywords)
        for kw in profile.extracted_keywords:
            all_keywords_set.add(kw["keyword"].lower())

    all_keywords = list(all_keywords_set)
    logger.info("Phase 1 complete: %d unique keywords across all competitors", len(all_keywords))

    # Phase 1d: Enrich with real metrics
    logger.info("Enriching %d keywords with search volume/difficulty", len(all_keywords))
    keyword_metrics = await _enrich_keywords(all_keywords)
    logger.info("Got metrics for %d keywords", len(keyword_metrics))

    # Phase 2: Topic clustering
    logger.info("Phase 2: Building topic clusters")
    keywords_to_cluster = [kw for kw in all_keywords if kw in keyword_metrics]
    competitor_keywords = {p.url: p.ranked_keywords for p in profiles}

    topic_clusters = await build_topic_clusters(
        keywords=keywords_to_cluster,
        keyword_metrics=keyword_metrics,
        competitor_keywords=competitor_keywords,
        min_cluster_size=3,
    )
    logger.info("Phase 2 complete: %d topic clusters", len(topic_clusters))

    return PipelineResult(
        competitors=profiles,
        topic_clusters=topic_clusters,
        all_keyword_metrics=keyword_metrics,
        total_keywords_found=len(all_keywords),
        total_clusters=len(topic_clusters),
    )
