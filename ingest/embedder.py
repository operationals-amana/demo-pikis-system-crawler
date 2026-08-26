"""
sentence-transformers wrapper.

Three things here are easy to get wrong and expensive to notice:

1. e5 PREFIXES ARE MANDATORY. These models are trained asymmetrically: passages must
   be embedded as "passage: ..." and queries as "query: ...". Omitting them costs
   roughly 8-10% nDCG and fails silently -- retrieval just gets quietly worse, with
   nothing in any log to explain why. embed_passages/embed_query are the only public
   entry points precisely so no caller can forget.

2. THREAD PINNING BEFORE IMPORT. torch reads OMP_NUM_THREADS at import time. On a
   shared-CPU container it otherwise grabs every visible core, all of them throttled,
   and the oversubscription makes it measurably slower than two threads.

3. The title is prepended at EMBED time only, never stored in chunks.content. A short
   chunk otherwise floats free of its document's topic ("...rose by 12% year on year"
   embeds against nothing useful). Keeping it out of `content` means the stored text
   still round-trips against char_start/char_end.
"""

import os

from app.config import EMBED_THREADS

# Must happen before torch is imported anywhere in the process.
os.environ.setdefault("OMP_NUM_THREADS", str(EMBED_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(EMBED_THREADS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from typing import Iterable  # noqa: E402

from sqlalchemy import text as sql  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.config import EMBEDDING_BATCH, EMBEDDING_MODEL  # noqa: E402
from app.logging_utils import _log  # noqa: E402
from ingest.store import _record_error  # noqa: E402

_model = None


def get_model():
    """Load once per process and cache. ~470 MB resident for e5-small."""
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(EMBED_THREADS)
        _log(f"embedder: loading {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        _log("embedder: model ready")
    return _model


def embed_passages(texts: Iterable[str], titles: Iterable[str | None] | None = None) -> list[list[float]]:
    """Embed chunk texts. Titles, when supplied, are prepended for context only."""
    body = list(texts)
    if not body:
        return []
    if titles is not None:
        heads = list(titles)
        prepared = [
            f"passage: {(heads[i] or '').strip()}\n\n{t}" if i < len(heads) else f"passage: {t}"
            for i, t in enumerate(body)
        ]
    else:
        prepared = [f"passage: {t}" for t in body]

    model = get_model()
    vectors = model.encode(
        prepared,
        batch_size=EMBEDDING_BATCH,
        # Unit vectors, so pgvector's <=> cosine distance is exact and cheap.
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


def embed_query(query: str) -> list[float]:
    """Embed a search query. NOTE the 'query: ' prefix -- see the module docstring."""
    model = get_model()
    vector = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    return vector.tolist()


_PENDING = """
SELECT c.id, c.content, a.title
FROM chunks c JOIN articles a ON a.id = c.article_id
WHERE (c.embedding IS NULL OR c.embedding_model IS DISTINCT FROM :model)
  AND a.processing_status <> 'archived'
ORDER BY c.id
"""


def embed_pending(db: Session, run_id: str | None = None, limit: int | None = None) -> dict[str, int]:
    rows = db.execute(sql(_PENDING), {"model": EMBEDDING_MODEL}).all()
    if limit:
        rows = rows[:limit]
    stats = {"embedded": 0, "failed": 0}
    if not rows:
        _log("embedder: nothing to embed")
        return stats

    _log(f"embedder: {len(rows)} chunk(s) to embed")
    batch_size = EMBEDDING_BATCH * 4
    for start in range(0, len(rows), batch_size):
        window = rows[start : start + batch_size]
        try:
            vectors = embed_passages([r[1] for r in window], [r[2] for r in window])
            for (chunk_id, _content, _title), vector in zip(window, vectors):
                db.execute(
                    sql(
                        "UPDATE chunks SET embedding = CAST(:v AS vector), "
                        "embedding_model = :m, embedded_at = now(), updated_at = now() "
                        "WHERE id = :id"
                    ),
                    {"v": str(vector), "m": EMBEDDING_MODEL, "id": chunk_id},
                )
            db.commit()
            stats["embedded"] += len(window)
            _log(f"embedder: {min(start + batch_size, len(rows))}/{len(rows)}")
        except Exception as exc:  # noqa: BLE001 -- batch boundary; record and continue
                                  # so one bad batch does not lose the whole corpus.
            db.rollback()
            stats["failed"] += len(window)
            _record_error(db, run_id, "embed", None, None, str(exc))

    # Roll the per-article counters up so the admin dashboard can show progress
    # without an expensive join.
    db.execute(
        sql(
            "UPDATE articles a SET embedded_chunk_count = sub.n, embedding_model = :m, "
            "embedded_at = now(), "
            "processing_status = CASE WHEN a.processing_status IN ('pending','processing') "
            "                         THEN 'ready' ELSE a.processing_status END "
            "FROM (SELECT article_id, count(*) AS n FROM chunks "
            "      WHERE embedding IS NOT NULL GROUP BY article_id) sub "
            "WHERE a.id = sub.article_id"
        ),
        {"m": EMBEDDING_MODEL},
    )
    db.commit()
    _log(f"embedder: {stats}")
    return stats
