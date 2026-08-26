"""
BM25 over the artifact built by ingest/index_builder.py.

The artifact is loaded once into process memory and hot-swapped when a newer version
appears, so a re-ingest reaches a running API without a redeploy.

Query lexing is the interesting part: the query is lexed under BOTH the english and
indonesian Postgres configs and the results unioned. That costs one cheap round trip
and means an Indonesian query can still reach English-stemmed chunks through shared
roots ("investasi"/"investment" do not share one, but "emisi"/"emission" nearly do,
and proper nouns and numerals match exactly).
"""

import pickle
import threading
import time

import numpy as np
from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.config import INDEX_POLL_SECONDS
from app.logging_utils import _log
from rag.stopwords import strip

_lock = threading.Lock()
_index: dict | None = None
_version: int = 0
_checked_at: float = 0.0


def _load(db: Session) -> None:
    global _index, _version, _checked_at
    row = db.execute(
        sql(
            "SELECT version, payload FROM search_index_artifacts "
            "WHERE kind='bm25' ORDER BY version DESC LIMIT 1"
        )
    ).first()
    if not row:
        _log("lexical: no BM25 artifact found; keyword channel disabled")
        _checked_at = time.time()
        return
    version, payload = row
    if version == _version:
        _checked_at = time.time()
        return
    with _lock:
        _index = pickle.loads(payload)
        _version = int(version)
        _checked_at = time.time()
    _log(f"lexical: loaded BM25 artifact v{_version} ({len(_index['chunk_ids'])} chunks)")


def ensure_loaded(db: Session, force: bool = False) -> None:
    """Load on first use, then re-check at most every INDEX_POLL_SECONDS."""
    if force or _index is None or (time.time() - _checked_at) > INDEX_POLL_SECONDS:
        _load(db)


def index_version() -> int:
    return _version


def lex_query(db: Session, query: str) -> list[str]:
    """
    Lex under both language configs and union.

    Done in Postgres rather than Python because Postgres owns the stemmers -- the
    chunk lexemes were produced by exactly these configs, so query terms must be
    produced the same way or they will not match.
    """
    row = db.execute(
        sql(
            "SELECT tsvector_to_array(to_tsvector('english', :q)) "
            "    || tsvector_to_array(to_tsvector('indonesian', :q)) AS lex"
        ),
        {"q": query},
    ).first()
    return strip(list(dict.fromkeys(row[0]))) if row and row[0] else []


def search(
    db: Session, query: str, limit: int = 60, allowed: set[int] | None = None
) -> list[tuple[int, float]]:
    """Return [(chunk_id, bm25_score)] best first."""
    ensure_loaded(db)
    if _index is None:
        return []

    terms = lex_query(db, query)
    if not terms:
        return []

    vocab = _index["vocab"]
    cols = [vocab[t] for t in terms if t in vocab]
    if not cols:
        return []

    tf = _index["tf"]
    idf = _index["idf"]
    denom_len = _index["denom_len"]
    k1 = _index["k1"]
    chunk_ids = _index["chunk_ids"]

    # Sparse slice of just the query's columns, then BM25 per (doc, term):
    #   idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len/avglen))
    sub = tf[:, cols].toarray().astype(np.float32)          # (n_docs, n_query_terms)
    denom = sub + denom_len[:, None]
    scored = np.where(sub > 0, (sub * (k1 + 1.0)) / np.maximum(denom, 1e-9), 0.0)
    scores = scored @ idf[cols]

    if allowed is not None:
        mask = np.fromiter((cid in allowed for cid in chunk_ids), dtype=bool, count=len(chunk_ids))
        scores = np.where(mask, scores, 0.0)

    hits = np.flatnonzero(scores > 0)
    if hits.size == 0:
        return []
    # argpartition beats a full sort: we only need the top `limit` of ~15k.
    top = hits[np.argsort(scores[hits])[::-1][:limit]]
    return [(int(chunk_ids[i]), float(scores[i])) for i in top]
