# Lokal — Local Business SEO Intelligence Platform

> Paste your Google Maps link. Get a fully optimized web presence. Live in 5 minutes.

## The Problem

Billions of local searches happen on Google daily. Businesses that show up win. Most local businesses can't compete — no SEO team, often no website. Just a Google Maps listing.

## What We Built

One input (Google Maps URL) triggers a 12-stage pipeline: competitive analysis, keyword research, intent classification, topic clustering, AI content generation, and a published SEO-optimized web presence.

## Pipeline

```
                         Google Maps URL
                              |
                    +---------+---------+
                    |   Places API      |
                    |   name, category  |
                    |   location, rating|
                    |   reviews, photos |
                    +---------+---------+
                              |
              +---------------+---------------+
              |                               |
   +----------+----------+       +-----------+-----------+
   | keyword_research()  |       | keyword_serp()        |
   | "Salon Berlin"      |       | per discovered keyword|
   | -> real search terms|       | -> who ranks on Google|
   +----------+----------+       +-----------+-----------+
              |                               |
              +--------> COMPETITORS <--------+
                         (SERP-based,
                          city-level geo:
                          Berlin = 1003854)
                              |
                    +---------+---------+
                    |  Tavily Crawl     |
                    |  depth 3          |
                    |  20 pages/site    |
                    |  x5 competitors   |
                    +---------+---------+
                              |
              +---------------+---------------+
              |                               |
   +----------+----------+       +-----------+-----------+
   | TF-IDF              |       | RAKE                  |
   | cross-doc scoring   |       | multi-word phrases    |
   | ngram (1,3)         |       | max_length=4          |
   +----------+----------+       +-----------+-----------+
              |                               |
              +-----> MERGED KEYWORDS <-------+
                      (top 50 selected)
                              |
              +---------------+---------------+
              |               |               |
   +----------+--+  +---------+---+  +--------+--------+
   | DataForSEO  |  | Pioneer     |  | Gemini          |
   | volume      |  | GLiNER2     |  | Embedding 2.0   |
   | difficulty   |  | intent      |  | 768-dim vectors |
   | CPC         |  | classify    |  |                 |
   +----------+--+  +---------+---+  +--------+--------+
              |               |               |
              +-------+-------+-------+-------+
                      |               |
               +------+------+  +-----+------+
               | Opportunity |  | HDBSCAN    |
               | Scoring     |  | Clustering |
               | vol x gap x |  | semantic   |
               | diff x      |  | topic      |
               | intent x    |  | groups     |
               | novelty     |  |            |
               +------+------+  +-----+------+
                      |               |
                      +-------+-------+
                              |
                    +---------+---------+
                    |  Claude Agent     |
                    |  Sonnet 4.6       |
                    |  adaptive thinking|
                    |                   |
                    |  Tools:           |
                    |   keyword_serp    |
                    |   keyword_research|
                    |   tavily_search   |
                    |                   |
                    |  SSE streaming    |
                    |  stage/tool_call/ |
                    |  thinking/text/   |
                    |  article/complete |
                    +---------+---------+
                              |
              +---------------+---------------+
              |               |               |
   +----------+--+  +---------+---+  +--------+--------+
   | Pydantic    |  | Schema.org  |  | Gemini          |
   | Structured  |  | JSON-LD     |  | Article         |
   | Output via  |  | Article +   |  | Embedding       |
   | messages.   |  | LocalBiz    |  | -> pgvector     |
   | parse()     |  | + Rating    |  | -> similarity   |
   +----------+--+  +---------+---+  +--------+--------+
              |               |               |
              +-------+-------+-------+-------+
                              |
                    +---------+---------+
                    | Published Article |
                    | {biz}.lokal0.app  |
                    |                   |
                    | - SEO content     |
                    | - Real photos     |
                    | - Schema markup   |
                    | - Related articles|
                    +-------------------+
```

### Stage Details

| # | Stage | Service | Output |
|---|-------|---------|--------|
| 1 | Business Extraction | Google Places API | name, category, location, rating, reviews, photos |
| 2 | Competitor Discovery | DataForSEO SERP | top 5 competitor domains (city-level geo) |
| 3 | Site Crawling | Tavily | markdown content, headings, metadata per page |
| 4 | Keyword Extraction | TF-IDF + RAKE | merged, deduped keyword list |
| 5 | Keyword Enrichment | DataForSEO | search volume, difficulty, CPC per keyword |
| 6 | Intent Classification | Pioneer GLiNER2 | informational / transactional / commercial / navigational |
| 7 | Embedding | Gemini 2.0 | 768-dim vectors for clustering |
| 8 | Topic Clustering | HDBSCAN | semantic keyword groups with opportunity scores |
| 9 | Content Agent | Claude Sonnet 4.6 | researched, written SEO articles |
| 10 | Structured Output | Anthropic API | typed article fields (keyword, meta, markdown) |
| 11 | Schema.org | Deterministic | Article + LocalBusiness JSON-LD |
| 12 | Article Embedding | Gemini + pgvector | similarity search for related articles |

## Partner Technologies

| Technology | Usage |
|-----------|-------|
| **Pioneer (Fastino)** | GLiNER2 intent classification. Collects training samples from DataForSEO. Daily cron auto-triggers LORA fine-tuning at 50+ samples. Fine-tuned model auto-detected. Intent weights affect opportunity scoring (transactional=1.5x, navigational=0.5x). |
| **Google Gemini** | Embedding 2.0 (768-dim) for keyword clustering (HDBSCAN) and article similarity (pgvector cosine distance). Batched with retry + backoff. |
| **Tavily** | Site crawling (depth 3, 20 pages) and agent web search tool for trends/freshness. |
| **Entire** | Developer platform for agent-human collaboration. |

## Pioneer Fine-Tuning Loop

```
  Every pipeline run:
  +------------------+     +------------------+     +------------------+
  | DataForSEO       | --> | Collect training | --> | Store in DB      |
  | returns intent   |     | samples (keyword |     | (intent_training |
  | labels per       |     | + intent pairs)  |     | _samples table)  |
  | keyword          |     |                  |     |                  |
  +------------------+     +------------------+     +--------+---------+
                                                             |
  Daily cron (24h):                                          |
  +------------------+     +------------------+     +--------+---------+
  | Pioneer API      | <-- | Upload JSONL     | <-- | >= 50 samples?   |
  | LORA fine-tune   |     | dataset          |     | Check count      |
  | 10 epochs        |     |                  |     |                  |
  | 5e-5 LR          |     |                  |     |                  |
  +--------+---------+     +------------------+     +------------------+
           |
  Next pipeline run:
  +--------+---------+     +------------------+
  | Auto-detect      | --> | Use fine-tuned   |
  | fine-tuned model |     | model for intent |
  | via Pioneer API  |     | classification   |
  +------------------+     +------------------+
```

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

### Prerequisites

- Python 3.14+ with [uv](https://docs.astral.sh/uv/)
- Docker (for local Postgres + seo-api)
- API keys: Tavily, Gemini, Anthropic, Pioneer (see `.env.example`)

### Quick Start

```bash
# 1. Clone with seo-api sibling
git clone https://github.com/lokal0/content-gen.git
git clone https://github.com/lokal0/seo-api.git

# 2. Install Python dependencies
cd content-gen
uv sync

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Start Postgres (pgvector) + seo-api
docker compose up -d

# 5. Run the engine
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The engine starts at `http://localhost:8000`. API docs at `/docs`.

### Without Docker (using Neon)

If using Neon Postgres instead of local Docker:

```bash
# Set DATABASE_URL to your Neon connection string in .env
# The app auto-converts postgresql:// to postgresql+asyncpg://
# and handles sslmode -> ssl parameter conversion

# Run seo-api separately
cd ../seo-api && npm install && npm run build && npm start &

# Run content-gen
cd ../content-gen && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Related Repos

- [lokal-next](https://github.com/aadilghani1/lokal-next) — Frontend (Next.js 16, Clerk, shadcn/ui, AI Elements)
- [seo-api](https://github.com/lokal0/seo-api) — DataForSEO proxy (Express/TypeScript)

Built at Big Berlin Hack 2026 by lokal0.
