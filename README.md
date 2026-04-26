# Lokal Content Engine

The backend intelligence engine for Lokal — a local business SEO platform that turns a Google Maps link into a fully optimized web presence.

## What It Does

Paste a Google Maps link. The engine analyzes your local market, discovers who outranks you, finds keyword gaps, and generates SEO-optimized content — published and live in minutes.

## Architecture

```
Google Maps URL → Competitor Discovery (DataForSEO SERP, city-level geo)
    → Site Crawling (Tavily, depth 3, 20 pages)
    → Keyword Extraction (TF-IDF + RAKE)
    → Keyword Enrichment (DataForSEO volume/difficulty/CPC)
    → Intent Classification (Pioneer GLiNER2 + auto fine-tuning)
    → Embedding (Gemini 2.0, 768-dim)
    → Topic Clustering (HDBSCAN)
    → Content Agent (Claude Sonnet 4.6, adaptive thinking + 3 tools)
    → Structured Output (Pydantic via Anthropic API)
    → Schema.org JSON-LD (deterministic)
    → Article Embedding (Gemini + pgvector similarity)
```

## Partner Technologies

| Technology | Usage |
|-----------|-------|
| **Pioneer (Fastino)** | GLiNER2 keyword intent classification. Auto fine-tunes daily when 50+ training samples accumulate from DataForSEO labels. |
| **Google Gemini** | Embedding 2.0 for article vectors + pgvector similarity search. 768-dim. |
| **Tavily** | Site crawling (depth 3, max 20 pages) and web search tool for the content agent. |
| **Entire** | Developer platform for agent-human collaboration. |

## API Endpoints

- `POST /api/v1/analyze` — Start async analysis job
- `GET /api/v1/analyze/{job_id}` — Poll job status + results
- `GET /api/v1/analyze/{job_id}/stream` — SSE real-time event stream
- `POST /api/v1/discover-competitors` — SERP-based competitor discovery
- `GET /api/v1/intent-model/status` — Pioneer model status
- `POST /api/v1/intent-model/train` — Trigger fine-tuning

## SSE Events

Real-time streaming via Server-Sent Events: `stage`, `tool_call`, `thinking`, `text`, `article`, `complete`, `error`.

## Content Agent

Claude Sonnet 4.6 with adaptive thinking. 3 tools: `keyword_serp` (Google SERP), `keyword_research` (related keywords), `tavily_search` (web search). Output structured via Pydantic + `messages.parse()`.

## Tech Stack

FastAPI, SQLAlchemy + asyncpg (Neon Postgres), Anthropic SDK, Google GenAI, Tavily, Pioneer, HDBSCAN, scikit-learn, pgvector.

## Setup

```bash
uv sync
cp .env.example .env
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Deployment

Hetzner cpx22 via Kamal v2. `kamal deploy`.

Built at Big Berlin Hack 2026 by lokal0.
