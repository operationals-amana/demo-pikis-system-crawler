"""POST /api/search -- hybrid retrieval, no LLM."""

import time
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.deps import get_db, optional_user
from app.schemas import FilterSpec, SearchHit, SearchRequest, SearchResponse
from rag.citations import pick_snippet
from rag.filters import Filters

router = APIRouter(prefix="/api", tags=["search"])


def to_filters(spec: FilterSpec | None) -> Filters | None:
    if spec is None:
        return None
    return Filters(
        source_slugs=spec.source_slugs,
        authors=spec.authors,
        doc_types=spec.doc_types,
        topics=spec.topics,
        languages=spec.languages,
        date_from=spec.date_from,
        date_to=spec.date_to,
    )


def _collapse_to_articles(candidates: list[dict[str, Any]], query_terms: set[str]) -> list[SearchHit]:
    """
    Results are ARTICLE-level (the wireframe says "42 results", not "42 passages").

    Each article is represented by its best-scoring chunk, which is also the snippet
    the user sees -- so clicking through lands on the passage that actually matched.
    """
    best: dict[str, dict[str, Any]] = {}
    for c in candidates:
        article_id = c.get("article_id")
        if article_id not in best or c.get("rerank", 0) > best[article_id].get("rerank", 0):
            best[article_id] = c

    hits = []
    for c in sorted(best.values(), key=lambda r: r.get("rerank", 0), reverse=True):
        published = c.get("published_at")
        hits.append(
            SearchHit(
                article_id=c["article_id"],
                title=c.get("title") or "(untitled)",
                publisher=c.get("publisher"),
                source=c.get("journal") or c.get("source_name"),
                source_slug=c.get("source_slug"),
                published_at=published.isoformat() if published else None,
                url=c.get("url"),
                doc_type=c.get("doc_type"),
                language=c.get("language"),
                topic=c.get("topic_slug"),
                authors=list(c.get("authors") or []),
                snippet=pick_snippet(c.get("content", ""), query_terms),
                best_chunk_id=c.get("id"),
                score=round(float(c.get("rerank") or 0.0), 5),
                scores={
                    "semantic": c.get("semantic"),
                    "lexical": c.get("lexical"),
                    "rrf": c.get("rrf"),
                    "rerank": c.get("rerank"),
                },
            )
        )
    return hits


@router.post("/search", response_model=SearchResponse)
def search(
    payload: SearchRequest,
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(optional_user),
) -> SearchResponse:
    from rag import lexical
    from rag.generator import rewrite_query
    from rag.retriever import retrieve

    started = time.perf_counter()
    filters = to_filters(payload.filters)
    from rag.metadata_filters import extract_explicit_filters

    # Metadata stated in the query is promoted to the same hard constraints as the
    # filter bar before either retrieval channel is allowed to rank candidates.
    filters = extract_explicit_filters(db, payload.query, filters)

    # Search is not streamed, so the extra ~400 ms of a translation call is hidden by
    # the spinner -- and without it an Indonesian query cannot reach the mostly-English
    # journal corpus at all (measured: recall@5 0.00 -> 0.88).
    extra: list[str] = []
    from app.config import ENABLE_QUERY_TRANSLATION, TRANSLATE_FROM_LANGUAGES

    if ENABLE_QUERY_TRANSLATION:
        rewritten = rewrite_query(payload.query)
        # Directional -- see TRANSLATE_FROM_LANGUAGES in app/config.py for the measured
        # reason an English query must NOT be translated into Indonesian here.
        if rewritten.query_translated and rewritten.language in TRANSLATE_FROM_LANGUAGES:
            extra.append(rewritten.query_translated)

    # Channel weights express the mode. Zeroing a weight rather than skipping the
    # channel keeps one code path for all three modes.
    from app.config import W_LEXICAL, W_SEMANTIC
    import rag.fusion as fusion_module

    original = (fusion_module.W_SEMANTIC, fusion_module.W_LEXICAL)
    if payload.mode == "semantic":
        fusion_module.W_SEMANTIC, fusion_module.W_LEXICAL = 1.0, 0.0
    elif payload.mode == "keyword":
        fusion_module.W_SEMANTIC, fusion_module.W_LEXICAL = 0.0, 1.0
    try:
        result = retrieve(
            db,
            payload.query,
            filters=filters,
            top_k=min(payload.limit + payload.offset, 60),
            extra_queries=extra or None,
        )
    finally:
        fusion_module.W_SEMANTIC, fusion_module.W_LEXICAL = original

    hits = _collapse_to_articles(result.candidates, result.query_terms)
    window = hits[payload.offset : payload.offset + payload.limit]
    took = int((time.perf_counter() - started) * 1000)

    # Audit log -- issue #1 requires query, sources used and response to be recorded.
    from sqlalchemy import text as sql

    db.execute(
        sql(
            "INSERT INTO query_log (user_id, endpoint, query, query_rewritten, filters, "
            "mode, retrieved_chunk_ids, result_count, latency_ms, index_version, had_evidence) "
            "VALUES (CAST(:u AS uuid), 'search', :q, :qr, CAST(:f AS jsonb), :m, :ids, :n, :ms, :iv, :he)"
        ),
        {
            "u": (user or {}).get("id"),
            "q": payload.query,
            "qr": extra[0] if extra else None,
            "f": __import__("json").dumps(filters.describe() if filters else {}),
            "m": payload.mode,
            "ids": [c.get("id") for c in result.candidates][:50],
            "n": len(hits),
            "ms": took,
            "iv": result.index_version,
            "he": bool(hits),
        },
    )
    db.commit()

    return SearchResponse(
        query=payload.query,
        mode=payload.mode,
        total=len(hits),
        took_ms=took,
        index_version=result.index_version,
        filters_applied=filters.describe() if filters else {},
        results=window,
    )
