"""
Upsert normalised articles.

Batch-then-row-by-row on failure, the sibling crawler's convention: one malformed
record must not cost the other 49 in its batch. The row-by-row replay is what turns
"the run died" into "49 rows landed and one ingest_errors row explains the 50th".

`metadata_locked_fields` is honoured here and nowhere else: any field an admin edited
is stripped from the incoming update, which is what makes admin edits survive
re-harvesting.
"""

from typing import Any, Iterable

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.logging_utils import _log

BATCH_SIZE = 50

# Columns ingestion is allowed to write. Note `first_seen_at` is insert-only, and
# processing_status is NOT overwritten on conflict -- re-harvesting an unchanged
# article must not knock it back to 'pending' and force a needless re-embed.
_UPSERT = """
INSERT INTO articles (
    source_id, external_id, doc_type, title, title_normalized, abstract, full_text,
    full_text_status, authors, keywords, publisher, journal, volume, issue, pages,
    doi, issn, published_at, url, pdf_url, galley_id, language, language_confidence,
    language_source, word_count, char_count, content_hash, dedup_key,
    processing_status, processing_error, raw, last_harvested_at
) VALUES (
    :source_id, :external_id, :doc_type, :title, :title_normalized, :abstract, :full_text,
    :full_text_status, :authors, :keywords, :publisher, :journal, :volume, :issue, :pages,
    :doi, :issn, :published_at, :url, :pdf_url, :galley_id, :language, :language_confidence,
    :language_source, :word_count, :char_count, :content_hash, :dedup_key,
    :processing_status, :processing_error, CAST(:raw AS jsonb), now()
)
ON CONFLICT (source_id, external_id) DO UPDATE SET
    doc_type         = CASE WHEN 'doc_type'  = ANY(articles.metadata_locked_fields) THEN articles.doc_type  ELSE EXCLUDED.doc_type END,
    title            = CASE WHEN 'title'     = ANY(articles.metadata_locked_fields) THEN articles.title     ELSE EXCLUDED.title END,
    title_normalized = CASE WHEN 'title'     = ANY(articles.metadata_locked_fields) THEN articles.title_normalized ELSE EXCLUDED.title_normalized END,
    abstract         = CASE WHEN 'abstract'  = ANY(articles.metadata_locked_fields) THEN articles.abstract  ELSE EXCLUDED.abstract END,
    full_text        = COALESCE(EXCLUDED.full_text, articles.full_text),
    full_text_status = CASE WHEN EXCLUDED.full_text IS NOT NULL THEN EXCLUDED.full_text_status ELSE articles.full_text_status END,
    authors          = CASE WHEN 'authors'   = ANY(articles.metadata_locked_fields) THEN articles.authors   ELSE EXCLUDED.authors END,
    keywords         = CASE WHEN 'keywords'  = ANY(articles.metadata_locked_fields) THEN articles.keywords  ELSE EXCLUDED.keywords END,
    publisher        = CASE WHEN 'publisher' = ANY(articles.metadata_locked_fields) THEN articles.publisher ELSE EXCLUDED.publisher END,
    journal = EXCLUDED.journal, volume = EXCLUDED.volume, issue = EXCLUDED.issue,
    pages = EXCLUDED.pages, doi = COALESCE(EXCLUDED.doi, articles.doi),
    issn = EXCLUDED.issn,
    published_at = CASE WHEN 'published_at' = ANY(articles.metadata_locked_fields) THEN articles.published_at ELSE EXCLUDED.published_at END,
    url = EXCLUDED.url, pdf_url = EXCLUDED.pdf_url, galley_id = EXCLUDED.galley_id,
    language = CASE WHEN 'language' = ANY(articles.metadata_locked_fields) THEN articles.language ELSE EXCLUDED.language END,
    language_confidence = EXCLUDED.language_confidence,
    language_source     = EXCLUDED.language_source,
    word_count = EXCLUDED.word_count, char_count = EXCLUDED.char_count,
    -- A changed content_hash is the signal the chunker uses to re-chunk.
    content_hash = EXCLUDED.content_hash,
    dedup_key    = EXCLUDED.dedup_key,
    raw          = EXCLUDED.raw,
    last_harvested_at = now(),
    updated_at   = now()
RETURNING (xmax = 0) AS inserted
"""

_DEFAULTS: dict[str, Any] = {
    "abstract": None, "full_text": None, "full_text_status": "none",
    "authors": [], "keywords": [], "publisher": None, "journal": None,
    "volume": None, "issue": None, "pages": None, "doi": None, "issn": [],
    "published_at": None, "url": None, "pdf_url": None, "galley_id": None,
    "language": "en", "language_confidence": 0.0, "language_source": "detected",
    "word_count": None, "char_count": None, "content_hash": None, "dedup_key": None,
    "processing_status": "pending", "processing_error": None,
    "title_normalized": None, "raw": None,
}


def _params(article: dict[str, Any], source_id: int) -> dict[str, Any]:
    import json

    row = {**_DEFAULTS, **article, "source_id": source_id}
    row["raw"] = json.dumps(row.get("raw"), ensure_ascii=False, default=str)
    return {k: row.get(k) for k in _DEFAULTS | {"source_id": 0, "external_id": "", "doc_type": "", "title": ""}}


def upsert_articles(
    db: Session, articles: Iterable[dict[str, Any]], source_id: int, run_id: str | None = None
) -> dict[str, int]:
    stats = {"inserted": 0, "updated": 0, "failed": 0}
    batch: list[dict[str, Any]] = []

    def flush(rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        try:
            for row in rows:
                inserted = db.execute(text(_UPSERT), row).scalar()
                stats["inserted" if inserted else "updated"] += 1
            db.commit()
        except Exception:  # noqa: BLE001 -- batch failed; replay row by row so one bad
                           # record costs one record, not the whole batch.
            db.rollback()
            for row in rows:
                try:
                    inserted = db.execute(text(_UPSERT), row).scalar()
                    stats["inserted" if inserted else "updated"] += 1
                    db.commit()
                except Exception as exc:  # noqa: BLE001
                    db.rollback()
                    stats["failed"] += 1
                    _record_error(db, run_id, "store", source_id, row.get("external_id"), str(exc))

    for article in articles:
        batch.append(_params(article, source_id))
        if len(batch) >= BATCH_SIZE:
            flush(batch)
            batch = []
    flush(batch)
    return stats


def _record_error(
    db: Session, run_id: str | None, stage: str, source_id: int | None,
    external_id: str | None, message: str,
) -> None:
    try:
        db.execute(
            text(
                "INSERT INTO ingest_errors (run_id, stage, source_id, external_id, message) "
                "VALUES (CAST(:run_id AS uuid), :stage, :source_id, :external_id, :message)"
            ),
            {"run_id": run_id, "stage": stage, "source_id": source_id,
             "external_id": external_id, "message": message[:2000]},
        )
        db.commit()
    except Exception:  # noqa: BLE001 -- logging must never break the run
        db.rollback()
