"""
Deterministic reranking, plus an MMR-style diversity cap.

No neural cross-encoder. bge-reranker-base is 278M params / ~1.1 GB and scoring 30
pairs on a throttled shared CPU is 2-4 seconds -- spent BEFORE the first streamed
token, which is precisely the latency the streaming UI exists to hide. This is stated
plainly as a demo-grade reranker; a cross-encoder is the obvious upgrade if the
deployment ever gets real CPU.

The diversity cap is not a nicety. Without it one long paper wins every slot and the
answer shows "6 sources" that are all the same document -- visible in the wireframe's
source count, and exactly the kind of thing a researcher notices immediately.
"""

import re
from typing import Any

from app.config import MAX_CHUNKS_PER_ARTICLE

# A journal article is a stronger citation than a podcast blurb. This is the lever
# that stops a 40-character media-coverage title outranking a peer-reviewed finding --
# see the corpus-composition risk: ~500 PYC documents are link-outs with no body.
DOC_TYPE_PRIOR = {
    "journal-article": 1.00,
    "brief-report": 0.92,
    "book": 0.90,
    "opinion": 0.82,
    "archive": 0.78,
    "post": 0.75,
    "works-archive": 0.70,
    "e-newsletter": 0.62,
    "media-coverage": 0.58,
    "podcast": 0.55,
    "infographic": 0.55,
}

W_RRF = 1.00
W_COVERAGE = 0.25
W_PHRASE = 0.15
W_TITLE = 0.10
W_RECENCY = 0.05
# Exponent on the doc_type prior. 1.0 = the prior applies as written; raise it to
# punish low-value document types harder, lower it toward 0 to disable.
DOC_PRIOR_STRENGTH = 2.0

RECENCY_WINDOW_YEARS = 8

_QUOTED = re.compile(r'"([^"]{3,})"')


def _coverage(query_terms: set[str], chunk_lexemes: list[str] | None) -> float:
    """Fraction of query terms present in the chunk. RRF loses this: it knows the
    chunk ranked highly, not how MUCH of the question it actually answers."""
    if not query_terms or not chunk_lexemes:
        return 0.0
    present = query_terms & set(chunk_lexemes)
    return len(present) / len(query_terms)


def _recency(published_year: int | None, newest_year: int | None) -> float:
    if not published_year or not newest_year:
        return 0.0
    age = newest_year - published_year
    if age <= 0:
        return 1.0
    if age >= RECENCY_WINDOW_YEARS:
        return 0.0
    return 1.0 - (age / RECENCY_WINDOW_YEARS)


def rerank(
    candidates: list[dict[str, Any]],
    query: str,
    query_terms: set[str],
    top_k: int,
    has_date_filter: bool = False,
    max_per_article: int = MAX_CHUNKS_PER_ARTICLE,
) -> list[dict[str, Any]]:
    """
    Score, sort, then apply the per-article cap.

    `candidates` must already carry the article metadata the signals need -- the
    retriever hydrates them in one query rather than N.
    """
    if not candidates:
        return []

    quoted = [q.lower() for q in _QUOTED.findall(query)]
    newest = max((c.get("published_year") or 0) for c in candidates) or None

    for c in candidates:
        content_lower = (c.get("content") or "").lower()
        title_lower = (c.get("title") or "").lower()

        coverage = _coverage(query_terms, c.get("lexemes"))
        phrase = 1.0 if quoted and any(q in content_lower for q in quoted) else 0.0
        title_overlap = _coverage(query_terms, (title_lower + " " + " ".join(c.get("keywords") or [])).split())
        doc_prior = DOC_TYPE_PRIOR.get(c.get("doc_type") or "", 0.7)
        # Recency only when the user did NOT ask for a date range -- if they did, they
        # have already expressed the time preference and we must not fight it.
        recency = 0.0 if has_date_filter else _recency(c.get("published_year"), newest)

        # Query-match signals are ADDITIVE bonuses on the fused rank...
        base = c.get("rrf", 0.0) * (
            W_RRF
            + W_COVERAGE * coverage
            + W_PHRASE * phrase
            + W_TITLE * title_overlap
            + W_RECENCY * recency
        )
        # ...but the document-type prior is MULTIPLICATIVE, and that distinction is
        # load-bearing rather than cosmetic.
        #
        # RRF scores across a candidate set sit in a narrow band (~0.016-0.026 here),
        # so an additive prior weighted at 0.10 moves a score by ~0.002 and can never
        # reorder anything. Measured on the wireframe's own demo question -- "Apa
        # tantangan utama transisi energi Indonesia?" -- the top eight were mostly
        # event announcements (a workshop, a roundtable, a handover ceremony) while
        # the substantive briefs sat just below, and the model correctly refused to
        # answer from them.
        #
        # This matters here specifically because ~500 of the ~1,325 PYC documents are
        # event notices and media links: without a real penalty they crowd out the
        # analysis. As a multiplier a `post` keeps 75% of its score and a
        # journal-article keeps 100%, which is enough to reorder a near-tied field
        # without letting an off-topic paper beat an on-topic brief.
        c["rerank"] = base * (doc_prior ** DOC_PRIOR_STRENGTH)
        c["signals"] = {
            "coverage": round(coverage, 3),
            "phrase": phrase,
            "title_overlap": round(title_overlap, 3),
            "doc_prior": doc_prior,
            "recency": round(recency, 3),
        }

    ranked = sorted(candidates, key=lambda r: r["rerank"], reverse=True)

    selected: list[dict[str, Any]] = []
    per_article: dict[str, int] = {}
    overflow: list[dict[str, Any]] = []
    for c in ranked:
        article_id = c.get("article_id")
        used = per_article.get(article_id, 0)
        if used >= max_per_article:
            overflow.append(c)
            continue
        per_article[article_id] = used + 1
        selected.append(c)
        if len(selected) >= top_k:
            break

    # If the cap starved the result set (a narrow filter leaving few articles), top up
    # from the overflow rather than returning three sources when eight were asked for.
    if len(selected) < top_k:
        selected.extend(overflow[: top_k - len(selected)])

    return selected[:top_k]
