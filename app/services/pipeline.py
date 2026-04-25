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
    """Find competitors by checking who ranks for seed keywords via DataForSEO SERP."""
    seed_keywords = []
    if business_category and business_location:
        seed_keywords.append(f"{business_category} {business_location}")
        seed_keywords.append(f"best {business_category} {business_location}")
        seed_keywords.append(f"{business_category} near me {business_location}")
    elif business_category:
        seed_keywords.append(f"best {business_category}")
    elif business_name and business_location:
        seed_keywords.append(f"{business_name} {business_location}")
    else:
        return []

    logger.info("Discovering competitors via SERP for seeds: %s", seed_keywords)

    skip_domains = {
        "google.com", "yelp.com", "yelp.de", "tripadvisor.com", "facebook.com",
        "instagram.com", "maps.google.com", "wikipedia.org", "youtube.com",
        "twitter.com", "x.com", "linkedin.com", "pinterest.com", "tiktok.com",
        "amazon.com", "ebay.com", "treatwell.de", "treatwell.com",
        "booksy.com", "fresha.com", "squareup.com",
    }
    biz_name_lower = (business_name or "").lower()

    seen_domains: dict[str, int] = {}

    for keyword in seed_keywords:
        try:
            serp_results = await seo_client.keyword_serp(keyword)
            for hit in serp_results:
                domain = hit.domain
                if not domain:
                    continue
                if domain in skip_domains:
                    continue
                if biz_name_lower and biz_name_lower.replace(" ", "") in domain.replace(".", ""):
                    continue
                seen_domains[domain] = seen_domains.get(domain, 0) + 1
        except Exception as e:
            logger.warning("SERP lookup failed for %r: %s", keyword, e)

    ranked = sorted(seen_domains.items(), key=lambda x: x[1], reverse=True)
    urls = [f"https://{domain}" for domain, _ in ranked[:limit]]

    logger.info("Discovered %d competitors from SERP: %s", len(urls), urls)
    return urls


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

    # Only enrich top 50 keywords to save DataForSEO credits
    # Prioritize: ranked keywords from SEO data first, then top extracted by score
    ranked_kws = set()
    for profile in profiles:
        ranked_kws.update(list(profile.ranked_keywords)[:20])
    extracted_by_score = sorted(
        [(kw["keyword"].lower(), kw["score"]) for p in profiles for kw in p.extracted_keywords],
        key=lambda x: x[1], reverse=True,
    )
    top_extracted = [kw for kw, _ in extracted_by_score if kw not in ranked_kws][:50 - len(ranked_kws)]
    keywords_to_enrich = list(ranked_kws | set(top_extracted))[:50]

    await progress("enriching_keywords", f"Enriching top {len(keywords_to_enrich)} keywords with search metrics")
    keyword_metrics = await _enrich_keywords(keywords_to_enrich)
    await progress("enriching_keywords", f"Got metrics for {len(keyword_metrics)} keywords")

    await progress("classifying_intent", f"Classifying {len(keywords_to_enrich)} keywords")
    keyword_intents = await classify_keywords(keywords_to_enrich)
    await progress("classifying_intent", f"Classified {len(keyword_intents)} keywords by intent")

    await progress("embedding_keywords", f"Embedding {len(keywords_to_enrich)} keywords with Gemini")
    keywords_to_cluster = [kw for kw in keywords_to_enrich if kw in keyword_metrics]
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
