import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.config import settings

_TIMEOUT = 60.0
_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 86400  # 24 hours


def _cache_key(func_name: str, **kwargs) -> str:
    raw = f"{func_name}:{sorted(kwargs.items())}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_get(key: str) -> Any | None:
    if key in _CACHE:
        ts, val = _CACHE[key]
        if time.time() - ts < _CACHE_TTL:
            return val
        del _CACHE[key]
    return None


def _cache_set(key: str, val: Any):
    _CACHE[key] = (time.time(), val)


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if settings.seo_api_token:
        h["Authorization"] = f"Bearer {settings.seo_api_token}"
    return h


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=settings.seo_api_url,
        headers=_headers(),
        timeout=_TIMEOUT,
    )


@dataclass
class KeywordMetric:
    keyword: str
    search_volume: int | None = None
    cpc: float | None = None
    competition: float | None = None
    keyword_difficulty: float | None = None
    intent: str | None = None
    trend: list[dict] | None = None


@dataclass
class SerpHit:
    rank: int
    title: str
    url: str
    domain: str
    description: str | None = None
    etv: float | None = None
    referring_domains: int | None = None
    backlinks: int | None = None


@dataclass
class DomainOverview:
    domain: str
    organic_traffic: int | None = None
    organic_keywords: int | None = None
    has_data: bool = False
    keywords: list[dict] = field(default_factory=list)
    pages: list[dict] = field(default_factory=list)


@dataclass
class BacklinksOverview:
    target: str
    summary: dict = field(default_factory=dict)
    backlinks: list[dict] = field(default_factory=list)
    trends: list[dict] = field(default_factory=list)


def _parse_keywords(items: list[dict]) -> list[KeywordMetric]:
    return [
        KeywordMetric(
            keyword=i["keyword"],
            search_volume=i.get("searchVolume"),
            cpc=i.get("cpc"),
            competition=i.get("competition"),
            keyword_difficulty=i.get("keywordDifficulty"),
            intent=i.get("intent"),
            trend=i.get("trend"),
        )
        for i in items
    ]


async def keyword_research(
    keywords: list[str],
    location_code: int = 2840,
    language_code: str = "en",
    mode: str = "auto",
) -> list[KeywordMetric]:
    async with _client() as c:
        r = await c.post("/keywords/research", json={
            "keywords": keywords,
            "locationCode": location_code,
            "languageCode": language_code,
            "mode": mode,
        })
        r.raise_for_status()
        return _parse_keywords(r.json().get("rows", []))


async def keyword_overview(
    keywords: list[str],
    location_code: int = 2840,
    language_code: str = "en",
) -> list[KeywordMetric]:
    ck = _cache_key("keyword_overview", keywords=tuple(sorted(keywords)), location_code=location_code, language_code=language_code)
    cached = _cache_get(ck)
    if cached is not None:
        return cached

    async with _client() as c:
        r = await c.post("/keywords/overview", json={
            "keywords": keywords,
            "locationCode": location_code,
            "languageCode": language_code,
        })
        r.raise_for_status()
        result = _parse_keywords(r.json().get("items", []))
        _cache_set(ck, result)
        return result


async def keyword_serp(
    keyword: str,
    location_code: int = 2840,
    language_code: str = "en",
    device: str = "desktop",
) -> list[SerpHit]:
    ck = _cache_key("keyword_serp", keyword=keyword, location_code=location_code, language_code=language_code, device=device)
    cached = _cache_get(ck)
    if cached is not None:
        return cached

    async with _client() as c:
        r = await c.post("/keywords/serp", json={
            "keyword": keyword,
            "locationCode": location_code,
            "languageCode": language_code,
            "device": device,
        })
        r.raise_for_status()
        data = r.json()
        result = [
            SerpHit(
                rank=i["rank"],
                title=i["title"],
                url=i["url"],
                domain=i["domain"],
                description=i.get("description"),
                etv=i.get("etv"),
                referring_domains=i.get("referringDomains"),
                backlinks=i.get("backlinks"),
            )
            for i in data.get("items", [])
        ]
        _cache_set(ck, result)
        return result


async def domain_overview(
    domain: str,
    include_subdomains: bool = True,
    location_code: int = 2840,
    language_code: str = "en",
) -> DomainOverview:
    ck = _cache_key("domain_overview", domain=domain, include_subdomains=include_subdomains, location_code=location_code, language_code=language_code)
    cached = _cache_get(ck)
    if cached is not None:
        return cached

    async with _client() as c:
        r = await c.post("/domain/overview", json={
            "domain": domain,
            "includeSubdomains": include_subdomains,
            "locationCode": location_code,
            "languageCode": language_code,
        })
        r.raise_for_status()
        d = r.json()
        result = DomainOverview(
            domain=d["domain"],
            organic_traffic=d.get("organicTraffic"),
            organic_keywords=d.get("organicKeywords"),
            has_data=d.get("hasData", False),
            keywords=d.get("keywords", []),
            pages=d.get("pages", []),
        )
        _cache_set(ck, result)
        return result


async def domain_suggestions(
    domain: str,
    location_code: int = 2840,
    language_code: str = "en",
) -> list[dict]:
    async with _client() as c:
        r = await c.post("/domain/suggestions", json={
            "domain": domain,
            "locationCode": location_code,
            "languageCode": language_code,
        })
        r.raise_for_status()
        return r.json().get("keywords", [])


async def backlinks_overview(
    target: str,
    scope: str = "domain",
    limit: int = 100,
) -> BacklinksOverview:
    async with _client() as c:
        r = await c.post("/backlinks/overview", json={
            "target": target,
            "scope": scope,
            "limit": limit,
        })
        r.raise_for_status()
        d = r.json()
        return BacklinksOverview(
            target=d["target"],
            summary=d.get("summary", {}),
            backlinks=d.get("backlinks", []),
            trends=d.get("trends", []),
        )


async def backlinks_referring_domains(
    target: str,
    scope: str = "domain",
    limit: int = 100,
) -> list[dict]:
    async with _client() as c:
        r = await c.post("/backlinks/referring-domains", json={
            "target": target,
            "scope": scope,
            "limit": limit,
        })
        r.raise_for_status()
        return r.json().get("rows", [])


async def backlinks_top_pages(
    target: str,
    scope: str = "domain",
    limit: int = 100,
) -> list[dict]:
    async with _client() as c:
        r = await c.post("/backlinks/top-pages", json={
            "target": target,
            "scope": scope,
            "limit": limit,
        })
        r.raise_for_status()
        return r.json().get("rows", [])
