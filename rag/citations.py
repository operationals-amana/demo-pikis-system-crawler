"""
Evidence snippet selection and citation validation.

Two jobs, both of which the wireframe depends on:

1. pick_snippet -- a 450-token chunk is far too long for the evidence panel's
   blockquote. This scores every 1-3 sentence window inside the chunk by query-term
   overlap and returns the best one, expanded to sentence boundaries. Deterministic
   and free: an LLM call per citation would add seconds to every answer for a job
   that is fundamentally "find the sentence that matches".

2. validate -- the third grounding gate. The model can only cite indices it was
   given, but nothing stops it emitting [9] when it was given six sources. An
   out-of-range index is stripped from the prose and recorded, and an answer with no
   valid citations at all is flagged ungrounded.
"""

import re
from typing import Any

CITATION = re.compile(r"\[(\d{1,3})\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

MAX_SNIPPET_CHARS = 320
WINDOW_SENTENCES = 3


def pick_snippet(content: str, query_terms: set[str], max_chars: int = MAX_SNIPPET_CHARS) -> str:
    """Best 1-3 sentence window of `content` for the query, capped at max_chars."""
    body = (content or "").strip()
    if not body:
        return ""
    if len(body) <= max_chars:
        return body

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(body) if s.strip()]
    if not sentences:
        return body[:max_chars].rsplit(" ", 1)[0] + "…"

    lowered = [s.lower() for s in sentences]
    best_score, best_window = -1.0, (0, 1)
    for start in range(len(sentences)):
        for size in range(1, WINDOW_SENTENCES + 1):
            end = start + size
            if end > len(sentences):
                break
            window = " ".join(lowered[start:end])
            if len(" ".join(sentences[start:end])) > max_chars:
                break
            # Distinct terms present, not raw frequency: a sentence repeating one
            # query word is not better evidence than one covering three.
            hits = sum(1 for term in query_terms if term and term in window)
            # Mild preference for longer windows at equal coverage -- more context
            # reads better in the evidence panel.
            score = hits + (end - start) * 0.01
            if score > best_score:
                best_score, best_window = score, (start, end)

    start, end = best_window
    snippet = " ".join(sentences[start:end]).strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0] + "…"
    # Signal that the quote is an extract, not the start of the document.
    if start > 0:
        snippet = "…" + snippet
    if end < len(sentences):
        snippet = snippet.rstrip(".") + "…"
    return snippet


def build_citations(candidates: list[dict[str, Any]], query_terms: set[str]) -> list[dict[str, Any]]:
    """
    Turn reranked candidates into the citation objects the API returns.

    Indices are assigned by FINAL RERANK ORDER, 1-based and contiguous, and are frozen
    before generation starts -- the model is told to cite these numbers and cannot
    influence which document a number points at.

    Note indices restart at 1 for every message. Turn 1's [3] and turn 2's [3] are
    different documents, so the frontend must scope lookups by message_id.
    """
    citations = []
    for index, c in enumerate(candidates, start=1):
        published = c.get("published_at")
        citations.append(
            {
                "index": index,
                "article_id": c.get("article_id"),
                "chunk_id": c.get("id"),
                "title": c.get("title"),
                "publisher": c.get("publisher"),
                "source": c.get("journal") or c.get("source_name"),
                "source_slug": c.get("source_slug"),
                "published_at": published.isoformat() if published else None,
                "url": c.get("url"),
                "doc_type": c.get("doc_type"),
                "language": c.get("language"),
                "section": c.get("section"),
                # `content` is the FULL chunk and is what goes into the <sources>
                # block the model reads. `snippet` is the short evidence quote the
                # UI shows. Sending only the snippet starves the model: measured, the
                # wireframe's own demo question was refused because eight ~230-char
                # fragments ending in an ellipsis genuinely do not contain an answer.
                # strip_content() removes `content` before the object is returned to
                # the client, so the wire payload stays small.
                "content": c.get("content", ""),
                "snippet": pick_snippet(c.get("content", ""), query_terms),
                "char_start": c.get("char_start"),
                "char_end": c.get("char_end"),
                "topic": c.get("topic_slug"),
                "score": round(float(c.get("rerank") or 0.0), 5),
            }
        )
    return citations


def validate(answer: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Strip out-of-range markers, report which sources were actually used, and decide
    whether the answer is grounded.
    """
    valid_indices = {c["index"] for c in citations}
    used: list[int] = []
    invalid: list[int] = []

    def _replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if index in valid_indices:
            if index not in used:
                used.append(index)
            return match.group(0)
        invalid.append(index)
        return ""  # a citation pointing nowhere is worse than no citation

    cleaned = CITATION.sub(_replace, answer or "")
    # Tidy the punctuation left behind by a removed marker.
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    substantive = len(cleaned) > 200
    grounded = bool(used) or not substantive

    return {
        "answer": cleaned,
        "used": sorted(used),
        "invalid": invalid,
        # Only the sources the answer actually cited reach the evidence panel, so it
        # never shows a document the prose did not use.
        "citations": [c for c in citations if c["index"] in used],
        "grounded": grounded,
    }


def strip_content(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Drop the full chunk text before sending citations to the client.

    The model needs the whole passage; the browser needs the title, the snippet and
    the URL. Shipping ~450 tokens x 8 of duplicate text to the UI would roughly
    quadruple the SSE payload for no visible benefit.
    """
    return [{k: v for k, v in c.items() if k != "content"} for c in citations]
