"""
The retrieval pipeline: filter -> lexical + semantic -> fuse -> hydrate -> rerank.

Returns candidates carrying everything downstream needs (article metadata, offsets,
per-channel scores), so the API layer never has to issue follow-up queries per result.

The no-evidence decision lives here rather than in the prompt, because a prompt
instruction is a request and a score threshold is a guarantee. See `has_evidence`.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.config import (
    MIN_EVIDENCE_CHUNKS,
    MIN_EVIDENCE_SCORE,
    MIN_SEMANTIC_SIMILARITY,
    RETRIEVAL_CANDIDATES,
    TOP_K_CONTEXT,
)
from rag import fusion, lexical, semantic
from rag.filters import Filters
from rag.rerank import rerank

# One hydration query for the whole candidate set. Doing this per chunk would be ~100
# round trips per search.
_HYDRATE = """
SELECT c.id, c.article_id::text, c.content, c.char_start, c.char_end, c.section,
       c.language, c.lexemes, c.chunk_index,
       a.title, a.publisher, a.published_at, a.published_year, a.url, a.doc_type,
       a.doi, a.journal, a.keywords, a.authors, a.abstract,
       s.slug AS source_slug, s.name AS source_name,
       t.slug AS topic_slug, t.label_en AS topic_label_en, t.label_id AS topic_label_id
FROM chunks c
JOIN articles a ON a.id = c.article_id
JOIN sources s ON s.id = a.source_id
LEFT JOIN topics t ON t.id = a.topic_primary_id
WHERE c.id = ANY(:ids)
"""


@dataclass
class RetrievalResult:
    candidates: list[dict[str, Any]] = field(default_factory=list)
    query: str = ""
    query_terms: set[str] = field(default_factory=set)
    timings: dict[str, int] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    index_version: int = 0

    def best_similarity(self) -> float:
        """Highest raw cosine among the candidates -- the absolute relevance signal."""
        return max((float(c.get("semantic") or 0.0) for c in self.candidates), default=0.0)

    def has_evidence(self) -> bool:
        """
        The pre-LLM gate: no LLM call at all when evidence is thin.

        TWO conditions, because they catch different failures:

        * best_similarity() >= MIN_SEMANTIC_SIMILARITY -- an ABSOLUTE check. RRF is
          rank-based and its top score is ~the same whether the match is perfect or
          merely the least-bad, so it cannot answer "is anything here relevant?".
          Cosine can. See the measured bands in app/config.py.
        * at least MIN_EVIDENCE_CHUNKS candidates above MIN_EVIDENCE_SCORE -- guards
          the case where exactly one chunk happens to match: a single passage is a
          quote, not a body of evidence, and answers built on one are brittle.
        """
        if self.best_similarity() < MIN_SEMANTIC_SIMILARITY:
            return False
        strong = [c for c in self.candidates if c.get("rerank", 0.0) >= MIN_EVIDENCE_SCORE]
        return len(strong) >= MIN_EVIDENCE_CHUNKS

    def best_score(self) -> float:
        return max((c.get("rerank", 0.0) for c in self.candidates), default=0.0)


def _hydrate(db: Session, fused: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not fused:
        return []
    ids = [row["chunk_id"] for row in fused]
    rows = db.execute(sql(_HYDRATE), {"ids": ids}).mappings().all()
    by_id = {row["id"]: dict(row) for row in rows}

    hydrated = []
    for row in fused:
        meta = by_id.get(row["chunk_id"])
        if not meta:
            # The chunk vanished between ranking and hydration (a concurrent re-index).
            # Dropping it is correct: we cannot cite what no longer exists.
            continue
        hydrated.append({**meta, **row})
    return hydrated


def retrieve(
    db: Session,
    query: str,
    filters: Filters | None = None,
    top_k: int = TOP_K_CONTEXT,
    candidates: int = RETRIEVAL_CANDIDATES,
    query_vector: list[float] | None = None,
    extra_queries: list[str] | None = None,
) -> RetrievalResult:
    """
    `extra_queries` carries the translated query for cross-lingual search. The lexical
    channel runs once per query text and the per-chunk scores are merged by max, which
    is what lets an Indonesian question reach an English paper through the keyword
    channel at all (their stems do not overlap).
    """
    timings: dict[str, int] = {}
    counts: dict[str, int] = {}

    t0 = time.perf_counter()
    allowed = None
    from rag.filters import allowed_chunk_ids

    if filters is not None and not filters.is_empty():
        allowed = allowed_chunk_ids(db, filters)
        counts["filtered_chunks"] = len(allowed)
        if not allowed:
            # A filter that matches nothing is a legitimate answer, not an error.
            return RetrievalResult(query=query, counts=counts, index_version=lexical.index_version())
    timings["filter_ms"] = int((time.perf_counter() - t0) * 1000)

    # --- lexical -----------------------------------------------------------
    t0 = time.perf_counter()
    merged_lexical: dict[int, float] = {}
    for text_variant in [query, *(extra_queries or [])]:
        if not text_variant or not text_variant.strip():
            continue
        for chunk_id, score in lexical.search(db, text_variant, limit=candidates, allowed=allowed):
            if score > merged_lexical.get(chunk_id, 0.0):
                merged_lexical[chunk_id] = score
    lexical_hits = sorted(merged_lexical.items(), key=lambda kv: kv[1], reverse=True)[:candidates]
    timings["lexical_ms"] = int((time.perf_counter() - t0) * 1000)
    counts["lexical"] = len(lexical_hits)

    # --- semantic ----------------------------------------------------------
    # The translated query must go through the SEMANTIC channel too, not just the
    # lexical one. Measured on the golden set: multilingual-e5-small ranks an English
    # paper at 1 for an English query but at 7-or-missing for the Indonesian
    # equivalent -- its cross-lingual alignment is simply not strong enough to carry
    # a mostly-English corpus on its own. Embedding both phrasings and merging by max
    # recovers that, and costs one extra forward pass (~20 ms).
    t0 = time.perf_counter()
    from ingest.embedder import embed_query

    if query_vector is None:
        query_vector = embed_query(query)

    merged_semantic: dict[int, float] = {}
    for chunk_id, score in semantic.search(db, query_vector, limit=candidates, filters=filters):
        merged_semantic[chunk_id] = score
    for text_variant in extra_queries or []:
        if not text_variant or not text_variant.strip() or text_variant.strip() == query.strip():
            continue
        for chunk_id, score in semantic.search(
            db, embed_query(text_variant), limit=candidates, filters=filters
        ):
            if score > merged_semantic.get(chunk_id, 0.0):
                merged_semantic[chunk_id] = score
    semantic_hits = sorted(merged_semantic.items(), key=lambda kv: kv[1], reverse=True)[:candidates]
    timings["semantic_ms"] = int((time.perf_counter() - t0) * 1000)
    counts["semantic"] = len(semantic_hits)

    # --- fuse, hydrate, rerank --------------------------------------------
    t0 = time.perf_counter()
    fused = fusion.reciprocal_rank_fusion(semantic_hits, lexical_hits)
    # Rerank only a bounded window: signals cost real work and the tail cannot win.
    fused = fused[: max(candidates, top_k * 4)]
    hydrated = _hydrate(db, fused)
    counts["fused"] = len(hydrated)

    query_terms = set(lexical.lex_query(db, query))
    for extra in extra_queries or []:
        query_terms |= set(lexical.lex_query(db, extra))

    has_date_filter = bool(filters and (filters.date_from or filters.date_to))
    top = rerank(hydrated, query, query_terms, top_k=top_k, has_date_filter=has_date_filter)
    timings["rerank_ms"] = int((time.perf_counter() - t0) * 1000)
    counts["returned"] = len(top)

    return RetrievalResult(
        candidates=top,
        query=query,
        query_terms=query_terms,
        timings=timings,
        counts=counts,
        index_version=lexical.index_version(),
    )
