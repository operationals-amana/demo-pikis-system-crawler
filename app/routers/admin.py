"""
Admin: dataset oversight -- issue #3's whole surface.

Everything here is gated by require_admin. The dashboard's numbers come from ONE
stats call, because a dashboard that issues eight queries in parallel from the
browser is eight chances to render half a page.
"""

import json
import threading
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.deps import get_db, require_admin
from app.errors import ApiError, bad_request, not_found
from app.logging_utils import _log
from app.schemas import ArticlePatch

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/stats")
def stats(db: Session = Depends(get_db), admin: dict = Depends(require_admin)) -> dict[str, Any]:
    def rows(query: str) -> list[dict[str, Any]]:
        return [dict(r) for r in db.execute(sql(query)).mappings().all()]

    def one(query: str) -> Any:
        return db.execute(sql(query)).scalar()

    base = "FROM articles WHERE duplicate_of_id IS NULL AND processing_status <> 'archived'"
    last_run = db.execute(
        sql(
            "SELECT id::text, kind, status, started_at, finished_at, stats "
            "FROM ingest_runs ORDER BY started_at DESC LIMIT 1"
        )
    ).mappings().first()

    from rag import lexical

    return {
        "total_articles": one(f"SELECT count(*) {base}"),
        "by_status": rows(
            "SELECT processing_status AS key, count(*) AS count FROM articles "
            "WHERE processing_status <> 'archived' GROUP BY 1 ORDER BY 2 DESC"
        ),
        "by_year": rows(
            f"SELECT published_year AS year, count(*) AS count {base} "
            "AND published_year IS NOT NULL GROUP BY 1 ORDER BY 1"
        ),
        "by_topic": rows(
            "SELECT t.slug, t.label_en, t.label_id, count(a.id) AS count "
            "FROM topics t LEFT JOIN articles a ON a.topic_primary_id = t.id "
            "  AND a.duplicate_of_id IS NULL AND a.processing_status <> 'archived' "
            "WHERE t.is_active GROUP BY t.slug, t.label_en, t.label_id, t.sort_order "
            "ORDER BY t.sort_order"
        ),
        "by_source": rows(
            "SELECT s.slug, s.name, count(a.id) AS count FROM sources s "
            "LEFT JOIN articles a ON a.source_id = s.id AND a.duplicate_of_id IS NULL "
            "  AND a.processing_status <> 'archived' GROUP BY s.slug, s.name ORDER BY 3 DESC"
        ),
        "by_language": rows(f"SELECT language AS key, count(*) AS count {base} GROUP BY 1 ORDER BY 2 DESC"),
        "by_doc_type": rows(f"SELECT doc_type AS key, count(*) AS count {base} GROUP BY 1 ORDER BY 2 DESC"),
        "publisher_count": one(f"SELECT count(DISTINCT publisher) {base} AND publisher IS NOT NULL"),
        "source_count": one("SELECT count(*) FROM sources"),
        "chunk_count": one("SELECT count(*) FROM chunks"),
        "embedded_chunk_count": one("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"),
        "full_text": rows(
            f"SELECT full_text_status AS key, count(*) AS count {base} GROUP BY 1 ORDER BY 2 DESC"
        ),
        "duplicates": one("SELECT count(*) FROM articles WHERE processing_status = 'duplicate'"),
        "index_version": lexical.index_version(),
        "last_ingest": dict(last_run) if last_run else None,
        "sessions": one("SELECT count(*) FROM chat_sessions WHERE deleted_at IS NULL"),
        "messages": one("SELECT count(*) FROM chat_messages"),
        "feedback": rows("SELECT rating AS key, count(*) AS count FROM message_feedback GROUP BY 1"),
        "llm_cost_usd": float(one("SELECT COALESCE(sum(cost_usd), 0) FROM chat_messages") or 0),
    }


_SORTS = {
    "published_at": "a.published_at",
    "title": "a.title_normalized",
    "status": "a.processing_status",
    "updated_at": "a.updated_at",
}


@router.get("/articles")
def list_articles(
    q: str | None = None,
    source: str | None = None,
    doc_type: str | None = None,
    status: str | None = None,
    topic: str | None = None,
    language: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    has_full_text: bool | None = None,
    sort: str = Query(default="published_at"),
    order: str = Query(default="desc", pattern="^(asc|desc)$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    db: Session = Depends(get_db),
    admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    where = ["a.processing_status <> 'archived'"]
    params: dict[str, Any] = {}

    if q:
        # search_vector (GIN) with a trigram fallback for typos and partial words --
        # websearch_to_tsquery alone misses "renewab".
        where.append(
            "(a.search_vector @@ websearch_to_tsquery('english', :q) "
            " OR a.search_vector @@ websearch_to_tsquery('indonesian', :q) "
            " OR a.title_normalized % lower(:q))"
        )
        params["q"] = q
    if source:
        where.append("s.slug = :source"); params["source"] = source
    if doc_type:
        where.append("a.doc_type = :doc_type"); params["doc_type"] = doc_type
    if status:
        where.append("a.processing_status = :status"); params["status"] = status
    if topic:
        where.append("t.slug = :topic"); params["topic"] = topic
    if language:
        where.append("a.language = :language"); params["language"] = language
    if year_from:
        where.append("a.published_year >= :yf"); params["yf"] = year_from
    if year_to:
        where.append("a.published_year <= :yt"); params["yt"] = year_to
    if has_full_text is True:
        where.append("a.full_text_status IN ('extracted')")
    elif has_full_text is False:
        where.append("a.full_text_status NOT IN ('extracted')")

    clause = " AND ".join(where)
    sort_col = _SORTS.get(sort, "a.published_at")
    direction = "ASC" if order == "asc" else "DESC"

    total = db.execute(
        sql(
            f"SELECT count(*) FROM articles a JOIN sources s ON s.id = a.source_id "
            f"LEFT JOIN topics t ON t.id = a.topic_primary_id WHERE {clause}"
        ),
        params,
    ).scalar() or 0

    rows = db.execute(
        sql(
            "SELECT a.id::text, a.title, a.publisher, a.published_at, a.doc_type, "
            "a.language, a.processing_status, a.full_text_status, a.chunk_count, "
            "a.topic_confidence, a.topic_source, a.duplicate_of_id::text, "
            "s.slug AS source_slug, t.slug AS topic, t.label_en AS topic_label "
            "FROM articles a JOIN sources s ON s.id = a.source_id "
            "LEFT JOIN topics t ON t.id = a.topic_primary_id "
            f"WHERE {clause} ORDER BY {sort_col} {direction} NULLS LAST "
            "LIMIT :limit OFFSET :offset"
        ),
        {**params, "limit": page_size, "offset": (page - 1) * page_size},
    ).mappings().all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "results": [
            {**dict(r), "published_at": r["published_at"].isoformat() if r["published_at"] else None}
            for r in rows
        ],
    }


@router.get("/articles/{article_id}")
def article_detail(
    article_id: str, db: Session = Depends(get_db), admin: dict = Depends(require_admin)
) -> dict[str, Any]:
    row = db.execute(
        sql(
            "SELECT a.*, a.id::text AS id_text, s.slug AS source_slug, "
            "t.slug AS topic, t.label_en AS topic_label "
            "FROM articles a JOIN sources s ON s.id = a.source_id "
            "LEFT JOIN topics t ON t.id = a.topic_primary_id "
            "WHERE a.id = CAST(:id AS uuid)"
        ),
        {"id": article_id},
    ).mappings().first()
    if not row:
        raise not_found("Article")
    out = {k: v for k, v in dict(row).items() if k not in ("id", "raw", "search_vector")}
    out["id"] = row["id_text"]
    out.pop("id_text", None)
    for key in ("published_at", "first_seen_at", "updated_at", "last_harvested_at", "processed_at", "embedded_at"):
        if out.get(key) is not None:
            out[key] = str(out[key])
    out["duplicate_of_id"] = str(row["duplicate_of_id"]) if row["duplicate_of_id"] else None
    out["topic_primary_id"] = row["topic_primary_id"]

    out["revisions"] = [
        {**dict(r), "created_at": str(r["created_at"])}
        for r in db.execute(
            sql(
                "SELECT r.changed, r.created_at, u.email AS edited_by "
                "FROM article_revisions r LEFT JOIN users u ON u.id = r.user_id "
                "WHERE r.article_id = CAST(:id AS uuid) ORDER BY r.created_at DESC LIMIT 20"
            ),
            {"id": article_id},
        ).mappings().all()
    ]
    out["chunks"] = [
        dict(r)
        for r in db.execute(
            sql(
                "SELECT id, chunk_index, token_count, language, section, "
                "(embedding IS NOT NULL) AS embedded FROM chunks "
                "WHERE article_id = CAST(:id AS uuid) ORDER BY chunk_index"
            ),
            {"id": article_id},
        ).mappings().all()
    ]
    return out


@router.patch("/articles/{article_id}")
def patch_article(
    article_id: str,
    payload: ArticlePatch,
    db: Session = Depends(get_db),
    admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    """
    Edit metadata. Three things happen atomically, and each is load-bearing:

    * the change lands in `article_revisions` (who, what, before/after);
    * every edited field is appended to `metadata_locked_fields`, which is what stops
      the next harvest silently reverting the edit;
    * a topic change also rewrites `article_topics` so the filter keeps working.
    """
    current = db.execute(
        sql("SELECT * FROM articles WHERE id = CAST(:id AS uuid)"), {"id": article_id}
    ).mappings().first()
    if not current:
        raise not_found("Article")

    edits = payload.model_dump(exclude_unset=True)
    if not edits:
        raise bad_request("Nothing to update")

    changed: dict[str, Any] = {}
    sets: list[str] = []
    params: dict[str, Any] = {"id": article_id}
    lockable = {"title", "abstract", "publisher", "published_at", "doc_type",
                "language", "url", "authors", "keywords"}
    locked: list[str] = []

    for field, value in edits.items():
        if field == "topic":
            topic_id = db.execute(
                sql("SELECT id FROM topics WHERE slug = :s AND is_active"), {"s": value}
            ).scalar()
            if topic_id is None:
                raise bad_request(f"Unknown topic {value!r}")
            changed["topic"] = {"before": None, "after": value}
            sets.append("topic_primary_id = :topic_id")
            sets.append("topic_source = 'manual'")
            sets.append("topic_confidence = 1.0")
            params["topic_id"] = topic_id
            db.execute(
                sql("DELETE FROM article_topics WHERE article_id = CAST(:id AS uuid)"),
                {"id": article_id},
            )
            db.execute(
                sql(
                    "INSERT INTO article_topics (article_id, topic_id, rank, confidence) "
                    "VALUES (CAST(:id AS uuid), :t, 1, 1.0)"
                ),
                {"id": article_id, "t": topic_id},
            )
            continue

        before = current.get(field)
        if before == value:
            continue
        changed[field] = {"before": str(before) if before is not None else None,
                          "after": str(value) if value is not None else None}
        sets.append(f"{field} = :{field}")
        params[field] = value
        if field in lockable:
            locked.append("title" if field == "title" else field)
        if field == "title":
            from ingest.html_text import normalize_title

            sets.append("title_normalized = :title_norm")
            params["title_norm"] = normalize_title(value)

    if not changed:
        return {"updated": [], "detail": "No changes"}

    if locked:
        sets.append(
            "metadata_locked_fields = (SELECT ARRAY(SELECT DISTINCT unnest(metadata_locked_fields || CAST(:locked AS text[]))))"
        )
        params["locked"] = locked

    db.execute(sql(f"UPDATE articles SET {', '.join(sets)} WHERE id = CAST(:id AS uuid)"), params)
    db.execute(
        sql(
            "INSERT INTO article_revisions (article_id, user_id, changed) "
            "VALUES (CAST(:a AS uuid), CAST(:u AS uuid), CAST(:c AS jsonb))"
        ),
        {"a": article_id, "u": admin["id"], "c": json.dumps(changed)},
    )
    db.commit()
    return {"updated": sorted(changed.keys())}


@router.post("/articles/{article_id}/reprocess")
def reprocess_article(
    article_id: str, db: Session = Depends(get_db), admin: dict = Depends(require_admin)
) -> dict[str, Any]:
    """Re-chunk and re-embed one article inline -- a few seconds for a single paper."""
    exists = db.execute(
        sql("SELECT 1 FROM articles WHERE id = CAST(:id AS uuid)"), {"id": article_id}
    ).scalar()
    if not exists:
        raise not_found("Article")

    # Clearing the fingerprint is the supported way to force a re-chunk.
    db.execute(
        sql(
            "UPDATE articles SET chunked_content_hash = NULL, "
            "processing_status = CASE WHEN processing_status = 'ready' THEN 'processing' "
            "ELSE processing_status END WHERE id = CAST(:id AS uuid)"
        ),
        {"id": article_id},
    )
    db.commit()

    from ingest.chunker import chunk_pending
    from ingest.embedder import embed_pending

    chunk_stats = chunk_pending(db)
    embed_stats = embed_pending(db)
    return {"chunked": chunk_stats, "embedded": embed_stats}


@router.post("/articles/{article_id}/duplicate")
def set_duplicate(
    article_id: str,
    payload: dict[str, Any],
    db: Session = Depends(get_db),
    admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    """Manual duplicate link/unlink. Manual links are never overwritten by dedupe."""
    target = payload.get("duplicate_of_id")
    if target == article_id:
        raise bad_request("An article cannot duplicate itself")
    if target:
        exists = db.execute(
            sql("SELECT 1 FROM articles WHERE id = CAST(:id AS uuid)"), {"id": target}
        ).scalar()
        if not exists:
            raise not_found("Target article")
        db.execute(
            sql(
                "UPDATE articles SET duplicate_of_id = CAST(:t AS uuid), "
                "duplicate_reason = 'manual', processing_status = 'duplicate' "
                "WHERE id = CAST(:id AS uuid)"
            ),
            {"id": article_id, "t": target},
        )
    else:
        db.execute(
            sql(
                "UPDATE articles SET duplicate_of_id = NULL, duplicate_reason = 'manual', "
                "processing_status = CASE WHEN processing_status = 'duplicate' "
                "THEN 'ready' ELSE processing_status END WHERE id = CAST(:id AS uuid)"
            ),
            {"id": article_id},
        )
    db.commit()
    return {"duplicate_of_id": target}


@router.delete("/articles/{article_id}", status_code=204)
def archive_article(
    article_id: str, db: Session = Depends(get_db), admin: dict = Depends(require_admin)
) -> None:
    """Soft: chat_messages.citations reference articles, so a hard delete would orphan
    the audit trail. Chunks go, so the article leaves retrieval immediately."""
    result = db.execute(
        sql("UPDATE articles SET processing_status = 'archived' WHERE id = CAST(:id AS uuid)"),
        {"id": article_id},
    )
    if not result.rowcount:
        raise not_found("Article")
    db.execute(sql("DELETE FROM chunks WHERE article_id = CAST(:id AS uuid)"), {"id": article_id})
    db.commit()


# --- ingest control --------------------------------------------------------

_ingest_thread: threading.Thread | None = None


@router.get("/ingest/runs")
def ingest_runs(
    db: Session = Depends(get_db), admin: dict = Depends(require_admin), limit: int = 20
) -> list[dict[str, Any]]:
    rows = db.execute(
        sql(
            "SELECT r.id::text, r.kind, r.status, r.triggered_by, r.started_at, "
            "r.finished_at, r.stats, r.error, "
            "(SELECT count(*) FROM ingest_errors e WHERE e.run_id = r.id) AS error_count "
            "FROM ingest_runs r ORDER BY r.started_at DESC LIMIT :l"
        ),
        {"l": min(limit, 100)},
    ).mappings().all()
    return [
        {**dict(r), "started_at": str(r["started_at"]),
         "finished_at": str(r["finished_at"]) if r["finished_at"] else None}
        for r in rows
    ]


@router.post("/ingest/run", status_code=202)
def trigger_ingest(
    payload: dict[str, Any] | None = None,
    db: Session = Depends(get_db),
    admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    """
    Kick a harvest cycle. 409 with the running job's details when one is going --
    the useful answer to "ingest now" during an ingest is "one is running, here is
    how long it has been at it", the sibling's exact semantics.
    """
    global _ingest_thread

    from app.config import INGEST_STALE_MINUTES

    # Only a row younger than the stale window counts as live -- a process that died
    # mid-run leaves 'running' behind forever, and without this cutoff one crashed
    # run would 409 every trigger for hours. The runner itself sweeps corpses once it
    # holds the advisory lock, so the table self-heals on the next successful start.
    running = db.execute(
        sql(
            "SELECT id::text, started_at FROM ingest_runs WHERE status = 'running' "
            "AND started_at > now() - make_interval(mins => :stale) "
            "ORDER BY started_at DESC LIMIT 1"
        ),
        {"stale": INGEST_STALE_MINUTES},
    ).mappings().first()
    if running or (_ingest_thread and _ingest_thread.is_alive()):
        detail = "An ingest run is already in progress"
        if running:
            detail += f" (run {running['id']}, started {running['started_at']})"
        raise ApiError(409, detail)

    incremental = bool((payload or {}).get("incremental", True))

    def _work() -> None:
        try:
            from ingest.runner import run

            since = None
            if incremental:
                from scripts.run_ingest import _last_successful_run_date

                since = _last_successful_run_date()
            run(since=since, triggered_by=f"admin:{admin['email']}")
        except Exception as exc:  # noqa: BLE001 -- the run records its own failure;
                                  # this thread must simply not die loudly.
            _log(f"admin ingest: {exc}")

    _ingest_thread = threading.Thread(target=_work, daemon=True, name="admin-ingest")
    _ingest_thread.start()
    return {"detail": "Ingest started", "incremental": incremental}


@router.post("/index/rebuild")
def rebuild_index_endpoint(
    db: Session = Depends(get_db), admin: dict = Depends(require_admin)
) -> dict[str, Any]:
    from ingest.index_builder import build_index

    return build_index(db)


# --- audit -----------------------------------------------------------------

@router.get("/audit")
def audit(
    db: Session = Depends(get_db),
    admin: dict = Depends(require_admin),
    endpoint: str | None = None,
    rating: str | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
) -> dict[str, Any]:
    """The issue's audit view: query, sources used, response, feedback, cost."""
    where = ["1=1"]
    params: dict[str, Any] = {}
    if endpoint:
        where.append("q.endpoint = :endpoint"); params["endpoint"] = endpoint
    if rating:
        where.append("f.rating = :rating"); params["rating"] = rating
    clause = " AND ".join(where)

    total = db.execute(
        sql(
            f"SELECT count(*) FROM query_log q "
            f"LEFT JOIN chat_messages m ON m.id = q.message_id "
            f"LEFT JOIN message_feedback f ON f.message_id = m.id WHERE {clause}"
        ),
        params,
    ).scalar() or 0

    rows = db.execute(
        sql(
            "SELECT q.id, q.created_at, q.endpoint, q.query, q.query_rewritten, "
            "q.filters, q.result_count, q.had_evidence, q.latency_ms, u.email AS user_email, "
            "left(m.content, 400) AS answer_preview, m.grounded, m.cost_usd, "
            "m.citations, f.rating AS feedback, f.comment AS feedback_comment "
            f"FROM query_log q "
            "LEFT JOIN users u ON u.id = q.user_id "
            "LEFT JOIN chat_messages m ON m.id = q.message_id "
            "LEFT JOIN message_feedback f ON f.message_id = m.id "
            f"WHERE {clause} ORDER BY q.created_at DESC LIMIT :limit OFFSET :offset"
        ),
        {**params, "limit": page_size, "offset": (page - 1) * page_size},
    ).mappings().all()

    return {
        "total": total, "page": page, "page_size": page_size,
        "results": [{**dict(r), "created_at": str(r["created_at"])} for r in rows],
    }
