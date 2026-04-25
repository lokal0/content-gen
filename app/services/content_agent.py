import json
import logging
from dataclasses import dataclass, field
from typing import Any

import anthropic

from app.core.config import settings
from app.services import seo_client
from app.services.pipeline import PipelineResult
from app.services.topic_clustering import TopicCluster

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 16000
MAX_ITERATIONS = 25

TOOLS = [
    {
        "name": "keyword_serp",
        "description": "Analyze the current Google SERP for a keyword. Returns the top ranking pages with their title, URL, domain, rank, estimated traffic, and backlink count. Use this to understand what content currently wins for a keyword and identify gaps.",
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "The keyword to analyze SERP for"},
                "device": {"type": "string", "enum": ["desktop", "mobile"], "description": "Device type", "default": "desktop"},
            },
            "required": ["keyword"],
        },
    },
    {
        "name": "keyword_research",
        "description": "Discover related keywords and their metrics (search volume, difficulty, CPC). Use this to find adjacent keyword opportunities within a topic cluster.",
        "input_schema": {
            "type": "object",
            "properties": {
                "keywords": {"type": "array", "items": {"type": "string"}, "description": "Seed keywords to research"},
                "mode": {"type": "string", "enum": ["auto", "related", "suggestions", "ideas"], "default": "auto"},
            },
            "required": ["keywords"],
        },
    },
    {
        "name": "backlinks_overview",
        "description": "Get backlink profile for a domain or page. Returns summary stats (total backlinks, referring domains, spam score) and trend data. Use this to assess link-building difficulty for a topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Domain or URL to analyze"},
                "scope": {"type": "string", "enum": ["domain", "page"], "default": "domain"},
            },
            "required": ["target"],
        },
    },
    {
        "name": "tavily_search",
        "description": "Search the web for current information about a topic. Use this to check content freshness, find recent developments, or verify facts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
]


async def _execute_tool(name: str, input_data: dict) -> str:
    try:
        if name == "keyword_serp":
            results = await seo_client.keyword_serp(
                keyword=input_data["keyword"],
                device=input_data.get("device", "desktop"),
            )
            return json.dumps([
                {"rank": r.rank, "title": r.title, "url": r.url, "domain": r.domain,
                 "description": r.description, "etv": r.etv,
                 "referring_domains": r.referring_domains, "backlinks": r.backlinks}
                for r in results[:10]
            ])

        elif name == "keyword_research":
            results = await seo_client.keyword_research(
                keywords=input_data["keywords"],
                mode=input_data.get("mode", "auto"),
            )
            return json.dumps([
                {"keyword": r.keyword, "search_volume": r.search_volume,
                 "keyword_difficulty": r.keyword_difficulty, "cpc": r.cpc, "intent": r.intent}
                for r in results[:20]
            ])

        elif name == "backlinks_overview":
            result = await seo_client.backlinks_overview(
                target=input_data["target"],
                scope=input_data.get("scope", "domain"),
            )
            return json.dumps({
                "target": result.target,
                "summary": result.summary,
                "top_backlinks": result.backlinks[:5],
            })

        elif name == "tavily_search":
            from tavily import TavilyClient
            import asyncio
            client = TavilyClient(api_key=settings.tavily_api_key)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.search(query=input_data["query"], max_results=5),
            )
            return json.dumps([
                {"title": r.get("title"), "url": r.get("url"), "content": r.get("content", "")[:500]}
                for r in response.get("results", [])
            ])

        return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        logger.error("Tool %s failed: %s", name, e)
        return json.dumps({"error": str(e)})


def _build_system_prompt(pipeline_result: PipelineResult) -> str:
    biz = pipeline_result.business
    biz_name = biz.display_name
    if biz.domain:
        business_summary = (
            f"{biz.domain}: {biz.organic_traffic or 'N/A'} organic traffic, "
            f"{len(biz.ranked_keywords)} ranked keywords"
        )
    else:
        business_summary = f"{biz_name} (no website yet — pre-launch)"

    competitors_summary = []
    for c in pipeline_result.competitors:
        competitors_summary.append(
            f"- {c.domain}: {c.organic_traffic or 'N/A'} organic traffic, "
            f"{len(c.ranked_keywords)} ranked keywords, "
            f"{len(c.top_pages)} top pages"
        )

    clusters_summary = []
    for cluster in pipeline_result.topic_clusters[:15]:
        coverage = ", ".join(
            f"{url.split('//')[1].split('/')[0]}: {cov:.0%}"
            for url, cov in cluster.competitor_coverage.items()
            if cov > 0
        )
        intent_breakdown = {}
        for m in cluster.keyword_metrics:
            intent = m.get("pioneer_intent", "unknown")
            intent_breakdown[intent] = intent_breakdown.get(intent, 0) + 1
        intent_str = ", ".join(f"{k}={v}" for k, v in intent_breakdown.items()) if intent_breakdown else "unclassified"

        clusters_summary.append(
            f"- Cluster #{cluster.id} (opportunity: {cluster.opportunity_score:.0f}): "
            f"{', '.join(cluster.keywords[:5])} | "
            f"vol={cluster.total_search_volume}, diff={cluster.avg_keyword_difficulty:.1f}, "
            f"cpc=${cluster.avg_cpc:.2f} | intents: {intent_str} | coverage: {coverage or 'none'}"
        )

    return f"""You are an expert SEO content strategist working for **{biz_name}**. You have analyzed their competitor websites and identified topic clusters ranked by opportunity — these are keyword gaps where competitors rank but {biz_name} does not.

## Your Client
{business_summary}

## Competitor Overview
{chr(10).join(competitors_summary)}

## Top Topic Clusters (ranked by opportunity score)
{chr(10).join(clusters_summary)}

Clusters are ranked by a composite score: search volume × competitor coverage gap × keyword difficulty × intent weight × novelty (keywords {biz_name} doesn't already rank for score higher).

## Your Task
For each of the top topic clusters, produce a detailed content brief and then write the full SEO-optimized content **for {biz_name}'s blog/website**.

**Process for each cluster:**
1. Use `keyword_serp` to analyze the current top-ranking pages for the primary keyword
2. Use `keyword_research` to find additional related keywords to target
3. Use `backlinks_overview` on the top-ranking domains to understand link difficulty
4. Use `tavily_search` to check for recent developments or angles competitors are missing

**Then produce for each cluster:**
A complete content brief followed by the full article in markdown, including:
- Target keyword + supporting keywords
- Search intent classification
- Meta title (≤60 chars) and meta description (≤155 chars)
- Full article with proper heading hierarchy (H1, H2, H3)
- Target word count based on SERP analysis
- Competitive angle (what {biz_name} does better than current top results)
- Schema.org markup suggestions
- Internal linking opportunities

Write content from {biz_name}'s perspective. Position them as the solution. Focus on the top 3-5 clusters with the highest opportunity scores. Write content that is genuinely better and more comprehensive than what currently ranks."""


@dataclass
class ContentPiece:
    cluster_id: int
    content: str
    thinking_steps: list[str] = field(default_factory=list)
    tools_used: list[dict] = field(default_factory=list)


@dataclass
class AgentResult:
    content_pieces: list[ContentPiece] = field(default_factory=list)
    full_response: str = ""
    thinking_blocks: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0


async def run_content_agent(pipeline_result: PipelineResult) -> AgentResult:
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    system_prompt = _build_system_prompt(pipeline_result)
    agent_result = AgentResult()

    messages = [
        {
            "role": "user",
            "content": "Analyze the top topic clusters and produce full SEO-optimized content for the best opportunities. Use all available tools to deeply research each topic before writing.",
        }
    ]

    for iteration in range(MAX_ITERATIONS):
        logger.info("Agent iteration %d/%d", iteration + 1, MAX_ITERATIONS)

        response = await client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            thinking={"type": "adaptive"},
            tools=TOOLS,
            messages=messages,
        )

        agent_result.total_input_tokens += response.usage.input_tokens
        agent_result.total_output_tokens += response.usage.output_tokens

        for block in response.content:
            if block.type == "thinking":
                agent_result.thinking_blocks.append(block.thinking)
            elif block.type == "text":
                agent_result.full_response += block.text

        if response.stop_reason == "end_turn":
            logger.info("Agent finished after %d iterations", iteration + 1)
            break

        if response.stop_reason == "pause_turn":
            messages = [
                {"role": "user", "content": messages[0]["content"]},
                {"role": "assistant", "content": response.content},
            ]
            continue

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tool in tool_use_blocks:
            logger.info("Calling tool: %s(%s)", tool.name, json.dumps(tool.input)[:100])
            result = await _execute_tool(tool.name, tool.input)
            agent_result.tool_calls.append({
                "name": tool.name,
                "input": tool.input,
                "output_preview": result[:200],
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

    logger.info(
        "Agent complete: %d tool calls, %d thinking blocks, %d input tokens, %d output tokens",
        len(agent_result.tool_calls),
        len(agent_result.thinking_blocks),
        agent_result.total_input_tokens,
        agent_result.total_output_tokens,
    )

    return agent_result
