"""
ONE filter predicate builder, used two ways.

The four filters the issues mandate -- source, date range, topic, language -- have to
apply identically to both retrieval channels, or a filtered search silently means two
different things depending on which channel surfaced a result.

So this module emits the predicate once, and exposes it in the two shapes the channels
need: inline SQL for the pgvector query, and an id-set for masking the BM25 score
vector. The alternative (writing the WHERE clause twice) is the kind of duplication
that drifts within a week.

Filters are applied BEFORE ranking, not as a post-hoc trim -- otherwise a narrow filter
returns the top-k of everything, filtered down to three results.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class Filters:
    source_slugs: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    doc_types: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)      # topic slugs
    languages: list[str] = field(default_factory=list)   # 'id' / 'en'
    date_from: date | None = None
    date_to: date | None = None

    def is_empty(self) -> bool:
        return not any(
            [self.source_slugs, self.authors, self.doc_types, self.topics, self.languages,
             self.date_from, self.date_to]
        )

    def describe(self) -> dict[str, Any]:
        """Serialisable form for query_log and the no-evidence notice."""
        out: dict[str, Any] = {}
        if self.source_slugs:
            out["sources"] = self.source_slugs
        if self.authors:
            out["authors"] = self.authors
        if self.doc_types:
            out["doc_types"] = self.doc_types
        if self.topics:
            out["topics"] = self.topics
        if self.languages:
            out["languages"] = self.languages
        if self.date_from:
            out["date_from"] = self.date_from.isoformat()
        if self.date_to:
            out["date_to"] = self.date_to.isoformat()
        return out


# Every retrieval query carries this base predicate. 'ready' excludes work in progress
# and failures; duplicate_of_id IS NULL is the show-each-document-once rule.
BASE_PREDICATE = "a.processing_status = 'ready' AND a.duplicate_of_id IS NULL"


def build(filters: Filters | None) -> tuple[str, dict[str, Any]]:
    """
    Return (sql_fragment, params). The fragment always starts with ' AND ' when
    non-empty, so callers can concatenate it after BASE_PREDICATE unconditionally.
    """
    if filters is None or filters.is_empty():
        return "", {}

    clauses: list[str] = []
    params: dict[str, Any] = {}

    if filters.source_slugs:
        clauses.append("s.slug = ANY(:f_sources)")
        params["f_sources"] = filters.source_slugs
    if filters.authors:
        # An explicitly named author is an identity constraint, not a fuzzy text
        # boost. Case-folding tolerates ingestion differences while retaining exact
        # author-name matching.
        clauses.append(
            "EXISTS (SELECT 1 FROM unnest(a.authors) AS article_author "
            "        WHERE lower(article_author) = ANY(:f_authors))"
        )
        params["f_authors"] = [author.lower() for author in filters.authors]
    if filters.doc_types:
        clauses.append("a.doc_type = ANY(:f_doc_types)")
        params["f_doc_types"] = filters.doc_types
    if filters.languages:
        clauses.append("a.language = ANY(:f_languages)")
        params["f_languages"] = filters.languages
    if filters.date_from:
        clauses.append("a.published_at >= :f_from")
        params["f_from"] = filters.date_from
    if filters.date_to:
        clauses.append("a.published_at <= :f_to")
        params["f_to"] = filters.date_to
    if filters.topics:
        # EXISTS over article_topics rather than a join, so an article carrying two
        # matching topics is not returned twice.
        clauses.append(
            "EXISTS (SELECT 1 FROM article_topics at JOIN topics t ON t.id = at.topic_id "
            "        WHERE at.article_id = a.id AND t.slug = ANY(:f_topics))"
        )
        params["f_topics"] = filters.topics

    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def allowed_chunk_ids(db, filters: Filters | None) -> set[int] | None:
    """
    The id-set shape, for masking BM25 scores.

    Returns None when there is no filter, which callers treat as "everything allowed"
    -- materialising 15k ids for the unfiltered case would be pure waste.
    """
    from sqlalchemy import text as sql

    if filters is None or filters.is_empty():
        return None
    fragment, params = build(filters)
    rows = db.execute(
        sql(
            "SELECT c.id FROM chunks c "
            "JOIN articles a ON a.id = c.article_id "
            "JOIN sources s ON s.id = a.source_id "
            f"WHERE {BASE_PREDICATE}{fragment}"
        ),
        params,
    ).all()
    return {r[0] for r in rows}
