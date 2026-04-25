import asyncio
import logging
import re
from dataclasses import dataclass, field

from tavily import TavilyClient

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CrawledPageData:
    url: str
    title: str | None = None
    full_text: str | None = None
    headings: dict | None = None
    metadata: dict | None = None
    schema_org: dict | None = None
    raw_content: dict | None = None


@dataclass
class CrawlResult:
    competitor_url: str
    pages: list[CrawledPageData] = field(default_factory=list)


def _extract_headings_from_markdown(md: str | None) -> dict | None:
    if not md:
        return None
    headings = {}
    for match in re.finditer(r"^(#{1,6})\s+(.+)$", md, re.MULTILINE):
        level = f"h{len(match.group(1))}"
        text = match.group(2).strip()
        headings.setdefault(level, []).append(text)
    return headings or None


def _extract_title_from_markdown(md: str | None) -> str | None:
    if not md:
        return None
    match = re.search(r"^#\s+(.+)$", md, re.MULTILINE)
    return match.group(1).strip() if match else None


def _markdown_to_plain_text(md: str | None) -> str | None:
    if not md:
        return None
    text = re.sub(r"!\[.*?\]\(.*?\)", "", md)
    text = re.sub(r"\[([^\]]+)\]\(.*?\)", r"\1", text)
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def crawl_competitor(url: str) -> CrawlResult:
    client = TavilyClient(api_key=settings.tavily_api_key)
    result = CrawlResult(competitor_url=url)

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.crawl(
                url=url,
                max_depth=settings.tavily_crawl_depth,
                limit=settings.tavily_max_pages,
            ),
        )
    except Exception as e:
        logger.error("Tavily crawl failed for %s: %s", url, e)
        return result

    pages = response.get("results", [])
    if not pages:
        logger.warning("Tavily returned 0 pages for %s (site may block crawlers)", url)
        return result

    logger.info("Crawled %s: %d pages in %.1fs", url, len(pages), response.get("response_time", 0))

    for page in pages:
        raw_md = page.get("raw_content") or ""
        plain_text = _markdown_to_plain_text(raw_md)

        page_data = CrawledPageData(
            url=page.get("url", url),
            title=page.get("title") or _extract_title_from_markdown(raw_md),
            full_text=plain_text,
            headings=_extract_headings_from_markdown(raw_md),
            metadata=page.get("metadata"),
            schema_org=None,
            raw_content={"markdown_length": len(raw_md), "markdown": raw_md[:50000] if raw_md else None},
        )
        result.pages.append(page_data)

    return result


async def crawl_all_competitors(urls: list[str]) -> list[CrawlResult]:
    tasks = [crawl_competitor(url) for url in urls]
    return await asyncio.gather(*tasks)
