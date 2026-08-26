"""GET /health and GET /api/meta."""

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.config import ANTHROPIC_MODEL, EMBEDDING_MODEL
from app.deps import get_db

router = APIRouter(tags=["meta"])


@router.get("/health")
def health(db: Session = Depends(get_db)) -> dict[str, Any]:
    """
    Liveness, not corpus health.

    Deliberately returns 200 even when the last ingest failed: a failed crawl must not
    make the platform restart the API mid-conversation. The numbers are informational.
    """
    from db.engine import check_connection
    from rag import lexical

    ok = check_connection()
    ready = 0
    chunks = 0
    if ok:
        ready = db.execute(
            sql("SELECT count(*) FROM articles WHERE processing_status = 'ready' AND duplicate_of_id IS NULL")
        ).scalar() or 0
        chunks = db.execute(sql("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")).scalar() or 0

    return {
        "status": "ok" if ok else "degraded",
        "db": "ok" if ok else "unreachable",
        "articles_ready": ready,
        "chunks_indexed": chunks,
        "index_version": lexical.index_version(),
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": ANTHROPIC_MODEL,
    }


@router.get("/api/meta")
def meta(db: Session = Depends(get_db)) -> dict[str, Any]:
    """
    Everything the frontend needs to build its filter dropdowns.

    Exists so the UI hardcodes nothing: adding a topic or a document type is a data
    change, not a frontend release. Deriving these client-side over 1,400 documents
    would also mean shipping the whole corpus to the browser.
    """
    sources = db.execute(
        sql(
            "SELECT s.slug, s.name, count(a.id) AS count "
            "FROM sources s LEFT JOIN articles a ON a.source_id = s.id "
            "  AND a.processing_status = 'ready' AND a.duplicate_of_id IS NULL "
            "GROUP BY s.slug, s.name ORDER BY count DESC"
        )
    ).mappings().all()

    doc_types = db.execute(
        sql(
            "SELECT doc_type AS slug, count(*) AS count FROM articles "
            "WHERE processing_status = 'ready' AND duplicate_of_id IS NULL "
            "GROUP BY doc_type ORDER BY count DESC"
        )
    ).mappings().all()

    topics = db.execute(
        sql(
            "SELECT t.slug, t.label_en, t.label_id, count(a.id) AS count "
            "FROM topics t LEFT JOIN articles a ON a.topic_primary_id = t.id "
            "  AND a.processing_status = 'ready' AND a.duplicate_of_id IS NULL "
            "WHERE t.is_active GROUP BY t.slug, t.label_en, t.label_id, t.sort_order "
            "ORDER BY t.sort_order"
        )
    ).mappings().all()

    languages = db.execute(
        sql(
            "SELECT language AS code, count(*) AS count FROM articles "
            "WHERE processing_status = 'ready' AND duplicate_of_id IS NULL AND language IS NOT NULL "
            "GROUP BY language ORDER BY count DESC"
        )
    ).mappings().all()

    years = db.execute(
        sql(
            "SELECT min(published_year) AS min, max(published_year) AS max FROM articles "
            "WHERE processing_status = 'ready' AND duplicate_of_id IS NULL AND published_year IS NOT NULL"
        )
    ).mappings().first()

    total = db.execute(
        sql("SELECT count(*) FROM articles WHERE processing_status = 'ready' AND duplicate_of_id IS NULL")
    ).scalar() or 0

    # When the corpus was last refreshed -- the finish time of the last successful
    # crawl cycle. Researchers see this in the workspace, so it answers "how fresh is
    # what I am reading?" without a trip to the admin dashboard.
    last_crawl = db.execute(
        sql(
            "SELECT finished_at, status FROM ingest_runs "
            "WHERE kind IN ('full', 'incremental') AND status IN ('ok', 'partial') "
            "AND finished_at IS NOT NULL ORDER BY finished_at DESC LIMIT 1"
        )
    ).first()

    return {
        "article_count": total,
        "last_crawled_at": last_crawl[0].isoformat() if last_crawl else None,
        "last_crawl_status": last_crawl[1] if last_crawl else None,
        "sources": [dict(r) for r in sources],
        "doc_types": [dict(r) for r in doc_types],
        "topics": [dict(r) for r in topics],
        "languages": [dict(r) for r in languages],
        "year_range": dict(years) if years else {"min": None, "max": None},
        # Two Indonesian, two English -- the wireframe's suggested questions, and a
        # deliberate signal that the system answers in both.
        "suggested_questions": [
            "Apa tantangan utama transisi energi Indonesia dalam tiga tahun terakhir?",
            "Bagaimana pembiayaan proyek energi terbarukan di Indonesia?",
            "What are the barriers to renewable energy investment in Indonesia?",
            "How does carbon pricing affect Indonesian industry?",
        ],
    }
