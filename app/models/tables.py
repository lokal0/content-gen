import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    competitors: Mapped[list["Competitor"]] = relationship(back_populates="submission", cascade="all, delete-orphan")


class Competitor(Base):
    __tablename__ = "competitors"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    submission_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("submissions.id"))
    url: Mapped[str] = mapped_column(String(2048))

    submission: Mapped["Submission"] = relationship(back_populates="competitors")
    crawled_pages: Mapped[list["CrawledPage"]] = relationship(back_populates="competitor", cascade="all, delete-orphan")
    keywords: Mapped[list["Keyword"]] = relationship(back_populates="competitor", cascade="all, delete-orphan")


class CrawledPage(Base):
    __tablename__ = "crawled_pages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    competitor_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("competitors.id"))
    url: Mapped[str] = mapped_column(String(2048))
    title: Mapped[str | None] = mapped_column(String(1024))
    full_text: Mapped[str | None] = mapped_column(Text)
    headings: Mapped[dict | None] = mapped_column(JSON)
    page_metadata: Mapped[dict | None] = mapped_column(JSON)
    schema_org: Mapped[dict | None] = mapped_column(JSON)
    raw_content: Mapped[dict | None] = mapped_column(JSON)

    competitor: Mapped["Competitor"] = relationship(back_populates="crawled_pages")


class Keyword(Base):
    __tablename__ = "keywords"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    competitor_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("competitors.id"))
    keyword: Mapped[str] = mapped_column(String(512))
    score: Mapped[float] = mapped_column(Float)
    method: Mapped[str] = mapped_column(String(50))

    competitor: Mapped["Competitor"] = relationship(back_populates="keywords")


class IntentTrainingSample(Base):
    __tablename__ = "intent_training_samples"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    keyword: Mapped[str] = mapped_column(String(512), index=True)
    intent: Mapped[str] = mapped_column(String(50))
    source: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
