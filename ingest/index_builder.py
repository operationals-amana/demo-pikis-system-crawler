"""
Build the BM25 artifact and persist it to the database.

Why BM25 in Python at all, when the lexemes already live in Postgres: `ts_rank` and
`ts_rank_cd` score term frequency and proximity and have NO IDF term. In a corpus
where every document is about energy, that means "energi" is treated as informative.
BM25's IDF is exactly what fixes that, and its k1 saturation matters here because
chunk lengths range from a 20-token podcast title to a 450-token paper section.

Why a scipy CSR matrix rather than rank_bm25: rank_bm25 keeps a Python dict of term
frequencies per document -- hundreds of MB at this corpus size. The same data as a
sparse matrix is ~8 MB. And scikit-learn/scipy/numpy are already hard dependencies of
sentence-transformers, so this channel costs no extra disk at all.

Why the artifact lives in Postgres rather than on disk: Railway's filesystem is
ephemeral and replicas must agree on which index version they serve. The API polls
MAX(version) and hot-swaps, which is how a re-ingest reaches a running API with no
redeploy.
"""

import hashlib
import pickle

import numpy as np
from scipy import sparse
from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.config import BM25_B, BM25_K1
from app.logging_utils import _log
from rag.stopwords import strip

# duplicate_of_id IS NULL is the "show each document once" filter; 'ready' excludes
# anything still processing or failed.
_CHUNKS = """
SELECT c.id, c.lexemes
FROM chunks c JOIN articles a ON a.id = c.article_id
WHERE a.processing_status = 'ready' AND a.duplicate_of_id IS NULL
  AND c.lexemes IS NOT NULL
ORDER BY c.id
"""


def build_index(db: Session) -> dict[str, int]:
    rows = db.execute(sql(_CHUNKS)).all()
    if not rows:
        _log("index: no ready chunks; nothing to build")
        return {"chunks": 0, "vocab": 0}

    chunk_ids = np.array([r[0] for r in rows], dtype=np.int64)
    docs = [strip(r[1]) for r in rows]

    # Vocabulary: term -> column index.
    vocab: dict[str, int] = {}
    indptr = [0]
    indices: list[int] = []
    data: list[int] = []
    for lexemes in docs:
        counts: dict[int, int] = {}
        for lexeme in lexemes:
            col = vocab.get(lexeme)
            if col is None:
                col = len(vocab)
                vocab[lexeme] = col
            counts[col] = counts.get(col, 0) + 1
        indices.extend(counts.keys())
        data.extend(counts.values())
        indptr.append(len(indices))

    n_docs = len(docs)
    n_terms = len(vocab)
    tf = sparse.csr_matrix(
        (np.array(data, dtype=np.float32), np.array(indices, dtype=np.int32),
         np.array(indptr, dtype=np.int64)),
        shape=(n_docs, max(n_terms, 1)),
    )

    doc_len = np.asarray(tf.sum(axis=1)).ravel()
    avg_len = float(doc_len.mean()) if n_docs else 0.0

    # Document frequency per term, then Robertson/Sparck-Jones IDF with the +0.5
    # smoothing, floored at a small positive value so a term appearing in every
    # document contributes ~0 rather than a negative score.
    df = np.diff(tf.tocsc().indptr).astype(np.float32)
    idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

    # Precompute the length-normalisation denominator: k1 * (1 - b + b * len/avglen).
    denom_len = (BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len / (avg_len or 1.0)))).astype(np.float32)

    payload = pickle.dumps(
        {
            "chunk_ids": chunk_ids,
            "vocab": vocab,
            "tf": tf,
            "idf": idf,
            "denom_len": denom_len,
            "k1": BM25_K1,
            "b": BM25_B,
        },
        protocol=5,
    )
    checksum = hashlib.sha256(payload).hexdigest()

    version = int(
        db.execute(sql("SELECT COALESCE(MAX(version), 0) + 1 FROM search_index_artifacts WHERE kind='bm25'")).scalar()
    )
    db.execute(
        sql(
            "INSERT INTO search_index_artifacts (kind, version, payload, chunk_count, "
            "vocab_size, checksum) VALUES ('bm25', :v, :p, :c, :s, :k)"
        ),
        {"v": version, "p": payload, "c": n_docs, "s": n_terms, "k": checksum},
    )
    # Keep the last 3 versions: enough to roll back, not enough to bloat the table
    # with 15 MB blobs forever.
    db.execute(
        sql(
            "DELETE FROM search_index_artifacts WHERE kind='bm25' AND version <= :cut"
        ),
        {"cut": version - 3},
    )
    db.commit()

    _log(
        f"index: built v{version} -- {n_docs} chunks, {n_terms} terms, "
        f"{len(payload) / 1e6:.1f} MB, avg_len {avg_len:.0f}"
    )
    return {"chunks": n_docs, "vocab": n_terms, "version": version, "bytes": len(payload)}
