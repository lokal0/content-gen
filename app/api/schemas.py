import uuid
from datetime import datetime

from pydantic import BaseModel, HttpUrl, field_validator


class AnalyzeRequest(BaseModel):
    business_url: HttpUrl | None = None
    business_name: str | None = None
    business_category: str | None = None
    business_location: str | None = None
    business_rating: float | None = None
    business_review_count: int | None = None
    competitor_urls: list[HttpUrl] = []
    skip_domain_enrichment: bool = False

    @field_validator("competitor_urls")
    @classmethod
    def max_five_urls(cls, v):
        if len(v) > 5:
            raise ValueError("Maximum 5 competitor URLs allowed")
        return v


class KeywordOut(BaseModel):
    keyword: str
    score: float
    method: str


class CrawledPageOut(BaseModel):
    url: str
    title: str | None
    headings: dict | None
    metadata: dict | None
    schema_org: dict | None


class CompetitorOut(BaseModel):
    url: str
    domain: str
    pages_crawled: int
    organic_traffic: int | None = None
    organic_keywords: int | None = None
    ranked_keywords_count: int = 0
    extracted_keywords: list[KeywordOut] = []
    top_pages: list[dict] = []


class TopicClusterOut(BaseModel):
    id: int
    label: str = ""
    keywords: list[str]
    total_search_volume: int = 0
    avg_keyword_difficulty: float = 0.0
    avg_cpc: float = 0.0
    competitor_coverage: dict[str, float] = {}
    opportunity_score: float = 0.0
    keyword_metrics: list[dict] = []


class AgentToolCallOut(BaseModel):
    name: str
    input: dict
    output_preview: str = ""


class ContentAgentOut(BaseModel):
    full_response: str = ""
    thinking_blocks: list[str] = []
    tool_calls: list[AgentToolCallOut] = []
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class BusinessProfileOut(BaseModel):
    url: str | None = None
    domain: str | None = None
    name: str | None = None
    organic_traffic: int | None = None
    organic_keywords: int | None = None
    ranked_keywords_count: int = 0


class AnalyzeResponse(BaseModel):
    submission_id: uuid.UUID
    status: str
    created_at: datetime
    business: BusinessProfileOut
    total_keywords_found: int = 0
    total_clusters: int = 0
    competitors: list[CompetitorOut]
    topic_clusters: list[TopicClusterOut] = []
    content: ContentAgentOut | None = None
