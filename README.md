# Lokal — Local Business SEO Intelligence Platform

> Paste your Google Maps link. Get a fully optimized web presence. Live in 5 minutes.

## The Problem

Billions of local searches happen on Google daily. Businesses that show up win. Most local businesses can't compete — no SEO team, often no website. Just a Google Maps listing.

## What We Built

One input (Google Maps URL) triggers a 12-stage pipeline: competitive analysis, keyword research, intent classification, topic clustering, AI content generation, and a published SEO-optimized web presence.

## Pipeline (12 stages)

1. **Business Extraction** — Google Places API (name, category, location, rating, reviews, photos)
2. **Competitor Discovery** — DataForSEO keyword_research for real search terms, then SERP analysis with city-level geo (Berlin = code 1003854)
3. **Site Crawling** — Tavily (depth 3, 20 pages per competitor)
4. **Keyword Extraction** — TF-IDF + RAKE from crawled content
5. **Keyword Enrichment** — DataForSEO bulk lookup (volume, difficulty, CPC)
6. **Intent Classification** — Pioneer GLiNER2 with auto fine-tuning loop
7. **Embedding** — Gemini Embedding 2.0 (768-dim)
8. **Topic Clustering** — HDBSCAN with opportunity scoring
9. **Content Agent** — Claude Sonnet 4.6, adaptive thinking, 3 tools, SSE streaming
10. **Structured Output** — Pydantic via Anthropic messages.parse()
11. **Schema.org JSON-LD** — Deterministic Article + LocalBusiness markup
12. **Article Embedding** — Gemini + pgvector for similarity search

## Partner Technologies

| Technology | Usage |
|-----------|-------|
| **Pioneer (Fastino)** | GLiNER2 intent classification. Collects training samples from DataForSEO. Daily cron auto-triggers LORA fine-tuning at 50+ samples. Fine-tuned model auto-detected. Intent weights affect opportunity scoring (transactional=1.5x, navigational=0.5x). |
| **Google Gemini** | Embedding 2.0 (768-dim) for keyword clustering (HDBSCAN) and article similarity (pgvector cosine distance). Batched with retry + backoff. |
| **Tavily** | Site crawling (depth 3, 20 pages) and agent web search tool for trends/freshness. |
| **Entire** | Developer platform for agent-human collaboration. |

## What Makes It Different

- **Not a ChatGPT wrapper.** Content informed by real SERP data, keyword volumes, competitor coverage gaps.
- **Network effect.** Pioneer fine-tuning means every business makes the platform smarter.
- **One input, published output.** Google Maps link to live website at {business}.lokal0.app with real photos, schema.org, targeting verified keyword gaps.
- **Real-time transparency.** SSE streams every stage, tool call, agent thought. Custom UI per tool.

## Infrastructure

- Hetzner cpx22 via Kamal v2 (content-gen + seo-api on same server)
- Neon Postgres with pgvector
- 24h in-memory cache on all DataForSEO calls
- Docker networking (seo-api internal via network alias)

## SSE Events

| Event | Frontend Component |
|-------|-------------------|
| stage | Dot + label |
| tool_call | Custom per tool (search page / data grid / reader) |
| thinking | Faded italic |
| text | Streaming content |
| article | Highlighted card |
| complete/error | Status indicator |

## Setup

```bash
uv sync
cp .env.example .env
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Related Repos

- [lokal-next](https://github.com/aadilghani1/lokal-next) — Frontend (Next.js 16, Clerk, shadcn/ui, AI Elements)
- [seo-api](https://github.com/lokal0/seo-api) — DataForSEO proxy (Express/TypeScript)

Built at Big Berlin Hack 2026 by lokal0.
