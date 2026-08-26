"""
Every request and response shape, in one file.

Field names are snake_case on the wire. That is a deliberate carry-over from the
sibling project's FastAPI era and matches what the Next.js lib/types.ts expects, so
neither side has to translate.
"""

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field


# --- auth ------------------------------------------------------------------

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=200)


class RegisterRequest(LoginRequest):
    full_name: str | None = Field(default=None, max_length=200)


class UserOut(BaseModel):
    id: str
    email: str
    full_name: str | None = None
    role: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: str | None = None
    user: UserOut


class RefreshRequest(BaseModel):
    refresh_token: str


# --- filters ---------------------------------------------------------------

class FilterSpec(BaseModel):
    source_slugs: list[str] = Field(default_factory=list)
    doc_types: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    date_from: date | None = None
    date_to: date | None = None


# --- search ----------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"
    filters: FilterSpec | None = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class SearchHit(BaseModel):
    article_id: str
    title: str
    publisher: str | None = None
    source: str | None = None
    source_slug: str | None = None
    published_at: str | None = None
    url: str | None = None
    doc_type: str | None = None
    language: str | None = None
    topic: str | None = None
    authors: list[str] = Field(default_factory=list)
    snippet: str = ""
    best_chunk_id: int | None = None
    score: float = 0.0
    scores: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    mode: str
    total: int
    took_ms: int
    index_version: int
    results: list[SearchHit]


# --- chat ------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    filters: FilterSpec | None = None
    stream: bool = True
    top_k: int = Field(default=8, ge=1, le=20)


class Citation(BaseModel):
    index: int
    article_id: str | None = None
    chunk_id: int | None = None
    title: str | None = None
    publisher: str | None = None
    source: str | None = None
    published_at: str | None = None
    url: str | None = None
    doc_type: str | None = None
    language: str | None = None
    section: str | None = None
    snippet: str = ""
    char_start: int | None = None
    char_end: int | None = None
    score: float = 0.0


class ChatResponse(BaseModel):
    message_id: str
    session_id: str
    answer: str
    key_findings: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    no_evidence: bool = False
    grounded: bool = True
    model: str | None = None
    latency_ms: int = 0
    usage: dict[str, Any] = Field(default_factory=dict)


class SessionOut(BaseModel):
    id: str
    title: str | None = None
    created_at: str
    updated_at: str
    last_message_at: str | None = None
    message_count: int = 0


class SessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    filters: FilterSpec | None = None


class SessionUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    filters: FilterSpec | None = None


class FeedbackRequest(BaseModel):
    rating: Literal["helpful", "needs_review"]
    reason: str | None = Field(default=None, max_length=200)
    comment: str | None = Field(default=None, max_length=2000)


# --- articles / admin ------------------------------------------------------

class ArticleOut(BaseModel):
    id: str
    title: str
    abstract: str | None = None
    full_text: str | None = None
    authors: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    publisher: str | None = None
    journal: str | None = None
    doi: str | None = None
    published_at: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    doc_type: str | None = None
    language: str | None = None
    source_slug: str | None = None
    topic: str | None = None
    processing_status: str | None = None
    full_text_status: str | None = None
    chunk_count: int = 0
    highlight: dict[str, int] | None = None


class ArticlePatch(BaseModel):
    title: str | None = Field(default=None, max_length=1000)
    abstract: str | None = None
    publisher: str | None = Field(default=None, max_length=300)
    published_at: date | None = None
    doc_type: str | None = Field(default=None, max_length=60)
    language: str | None = Field(default=None, pattern="^(id|en)$")
    url: str | None = Field(default=None, max_length=1000)
    authors: list[str] | None = None
    keywords: list[str] | None = None
    topic: str | None = Field(default=None, max_length=60)
