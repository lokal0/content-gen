import uuid
from datetime import datetime

from pydantic import BaseModel, HttpUrl, field_validator


class AnalyzeRequest(BaseModel):
    urls: list[HttpUrl]

    @field_validator("urls")
    @classmethod
    def exactly_five_urls(cls, v):
        if len(v) != 5:
            raise ValueError("Exactly 5 competitor URLs are required")
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
    pages_crawled: int
    keywords: list[KeywordOut]
    pages: list[CrawledPageOut]


class AnalyzeResponse(BaseModel):
    submission_id: uuid.UUID
    status: str
    created_at: datetime
    competitors: list[CompetitorOut]
