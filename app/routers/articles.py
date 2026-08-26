"""Researcher-facing article access -- the workspace's article drawer reads these."""

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.deps import current_user, get_db
from app.errors import not_found

router = APIRouter(prefix="/api/articles", tags=["articles"])

_DETAIL = """
SELECT a.id::text, a.title, a.abstract, a.authors, a.keywords, a.publisher,
       a.journal, a.volume, a.issue, a.pages, a.doi, a.published_at, a.url,
       a.pdf_url, a.doc_type, a.language, a.processing_status, a.full_text_status,
       a.chunk_count, a.word_count, a.char_count, a.full_text,
       s.slug AS source_slug, s.name AS source_name,
       t.slug AS topic, t.label_en AS topic_label_en, t.label_id AS topic_label_id
FROM articles a
JOIN sources s ON s.id = a.source_id
LEFT JOIN topics t ON t.id = a.topic_primary_id
WHERE a.id = CAST(:id AS uuid)
"""


@router.get("/{article_id}")
def get_article(
    article_id: str,
    include: str | None = Query(default=None, description="'full_text' to include the body"),
    highlight_chunk: int | None = Query(default=None),
    db: Session = Depends(get_db),
    user: dict = Depends(current_user),
) -> dict[str, Any]:
    row = db.execute(sql(_DETAIL), {"id": article_id}).mappings().first()
    if not row:
        raise not_found("Article")
    out = dict(row)
    out["published_at"] = out["published_at"].isoformat() if out["published_at"] else None

    # The body is heavy (IJE averages ~35k chars) and the drawer usually only needs
    # metadata + abstract, so it is opt-in.
    if include != "full_text":
        out.pop("full_text", None)

    # Where to scroll and highlight, so clicking a citation needs no second retrieval.
    if highlight_chunk is not None:
        chunk = db.execute(
            sql(
                "SELECT char_start, char_end, content FROM chunks "
                "WHERE id = :c AND article_id = CAST(:a AS uuid)"
            ),
            {"c": highlight_chunk, "a": article_id},
        ).mappings().first()
        out["highlight"] = dict(chunk) if chunk else None

    return out


@router.get("/{article_id}/chunks")
def get_chunks(
    article_id: str,
    ids: str | None = Query(default=None, description="comma-separated chunk ids"),
    db: Session = Depends(get_db),
    user: dict = Depends(current_user),
) -> list[dict[str, Any]]:
    """Resolve specific evidence passages (the citation objects carry chunk ids)."""
    wanted: list[int] = []
    if ids:
        for piece in ids.split(","):
            piece = piece.strip()
            if piece.isdigit():
                wanted.append(int(piece))
    rows = db.execute(
        sql(
            "SELECT id, chunk_index, content, char_start, char_end, section, language "
            "FROM chunks WHERE article_id = CAST(:a AS uuid) "
            + ("AND id = ANY(:ids) " if wanted else "")
            + "ORDER BY chunk_index"
        ),
        {"a": article_id, **({"ids": wanted} if wanted else {})},
    ).mappings().all()
    return [dict(r) for r in rows]
