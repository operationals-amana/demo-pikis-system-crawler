"""
pgvector cosine search -- exact scan, no ANN index. See db/migrations/0007 for why.

At ~15k chunks and 384 dimensions this is ~25 MB of vectors: a sequential scan runs in
tens of milliseconds and gives 100% recall, which is what a citation-grounded product
needs. An HNSW index on pgvector < 0.8.0 would post-filter after collecting ef_search
candidates and silently return fewer rows than asked for whenever a filter is narrow --
and narrow filters are exactly what the UI encourages.
"""

from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from rag.filters import BASE_PREDICATE, Filters, build


def search(
    db: Session, query_vector: list[float], limit: int = 60, filters: Filters | None = None
) -> list[tuple[int, float]]:
    """Return [(chunk_id, cosine_similarity)] best first."""
    fragment, params = build(filters)
    params = {**params, "qv": str(query_vector), "lim": limit}
    rows = db.execute(
        sql(
            "SELECT c.id, 1 - (c.embedding <=> CAST(:qv AS vector)) AS score "
            "FROM chunks c "
            "JOIN articles a ON a.id = c.article_id "
            "JOIN sources s ON s.id = a.source_id "
            f"WHERE {BASE_PREDICATE}{fragment} AND c.embedding IS NOT NULL "
            "ORDER BY c.embedding <=> CAST(:qv AS vector) "
            "LIMIT :lim"
        ),
        params,
    ).all()
    return [(int(r[0]), float(r[1])) for r in rows]
