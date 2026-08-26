"""
Weighted Reciprocal Rank Fusion.

RRF rather than a normalised weighted sum, for a concrete reason: BM25 scores are
unbounded and their scale shifts with corpus statistics after every re-ingest, while
cosine sits in [-1, 1]. Min-max normalising both would make every blended score depend
on the WORST candidate in each list -- one outlier silently re-ranks everything.

RRF reads only ranks. It needs no calibration, survives a re-index unchanged, and
degrades gracefully when one channel returns nothing useful (that channel then
contributes a near-flat ~1/k to everything, which is the correct behaviour).
"""

from app.config import RRF_K, W_LEXICAL, W_SEMANTIC


def reciprocal_rank_fusion(
    semantic: list[tuple[int, float]],
    lexical: list[tuple[int, float]],
    k: int = RRF_K,
    w_semantic: float = W_SEMANTIC,
    w_lexical: float = W_LEXICAL,
) -> list[dict]:
    """
    Fuse two ranked lists.

    Returns [{chunk_id, rrf, semantic, lexical, rank_semantic, rank_lexical}] sorted
    by fused score. The per-channel scores are carried through because the reranker
    and the API's debug output both want to show which channel found what.
    """
    scores: dict[int, float] = {}
    detail: dict[int, dict] = {}

    for rank, (chunk_id, raw) in enumerate(semantic, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + w_semantic / (k + rank)
        detail.setdefault(chunk_id, {})["semantic"] = raw
        detail[chunk_id]["rank_semantic"] = rank

    for rank, (chunk_id, raw) in enumerate(lexical, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + w_lexical / (k + rank)
        detail.setdefault(chunk_id, {})["lexical"] = raw
        detail[chunk_id]["rank_lexical"] = rank

    fused = [
        {
            "chunk_id": chunk_id,
            "rrf": score,
            "semantic": detail[chunk_id].get("semantic"),
            "lexical": detail[chunk_id].get("lexical"),
            "rank_semantic": detail[chunk_id].get("rank_semantic"),
            "rank_lexical": detail[chunk_id].get("rank_lexical"),
        }
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda row: row["rrf"], reverse=True)
    return fused
