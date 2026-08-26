"""
IJE galley PDF -> clean full text.

Four post-processing steps, none of them optional and none of them obvious:

1. Running header/footer removal. Every page of an OJS galley carries the journal
   name and issue. Left in, every chunk repeats "Indonesian Journal of Energy Vol. 7
   No. 2", which both wastes ~15% of each chunk's budget and teaches BM25 that the
   journal name is a meaningless token.

2. De-hyphenation. Words broken across line ends ("renew-\\nable") otherwise tokenise
   as two non-words, so the chunk becomes unfindable by the term it is actually about.

3. References truncation -- the single highest-value line in this file. A bibliography
   is ~20% of an academic PDF and is pure citation noise. Left in, a query about
   Indonesian coal exports retrieves a BIBLIOGRAPHY ENTRY that mentions coal, and the
   model then cites a chunk containing no claim at all.

4. A quality gate. No OCR: the galleys are born-digital with a real text layer
   (verified -- 110 embedded fonts and text-drawing operators). A scanned outlier
   degrades to its abstract and is VISIBLE as full_text_status='poor' rather than
   silently becoming a page of ligature soup.

The PDF bytes are streamed, parsed and discarded. Nothing is ever written to disk:
95 x ~700 KB is nothing, but "we store PDFs" becomes permanent the moment it is true.
"""

import io
import re
from collections import Counter
from typing import Any

from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.logging_utils import _log
from ingest.http import client, get
from ingest.store import _record_error

# Matches a references heading on its own line, in English or Indonesian.
_REFS = re.compile(
    r"^[\s\d.]*(REFERENCES?|BIBLIOGRAPHY|DAFTAR\s+PUSTAKA|REFERENSI)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_HYPHEN_BREAK = re.compile(r"(\w+)-\s*\n\s*(\w+)")
_MULTISPACE = re.compile(r"[ \t]+")
_MULTINEWLINE = re.compile(r"\n{3,}")

MIN_CHARS = 500          # below this, extraction did not really work
MIN_ALPHA_RATIO = 0.60   # ligature soup and scan artefacts score far below this
BOILERPLATE_PAGE_RATIO = 0.6
EDGE_LINES = 2           # how many lines at each page edge count as header/footer


def _strip_running_headers(pages: list[str]) -> list[str]:
    """
    Drop lines that repeat at the top or bottom of most pages.

    Only the first and last EDGE_LINES of each page are considered, so a sentence that
    happens to recur in the body is never removed.
    """
    if len(pages) < 3:
        return pages

    counts: Counter[str] = Counter()
    for page in pages:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        edges = lines[:EDGE_LINES] + lines[-EDGE_LINES:]
        # set() so a line appearing twice on one page still counts once.
        for line in set(edges):
            counts[line] += 1

    threshold = max(2, int(len(pages) * BOILERPLATE_PAGE_RATIO))
    boilerplate = {
        line for line, n in counts.items()
        # Long lines are prose that happens to repeat, not a running header.
        if n >= threshold and len(line) < 120
    }
    if not boilerplate:
        return pages

    cleaned = []
    for page in pages:
        kept = [ln for ln in page.splitlines() if ln.strip() not in boilerplate]
        cleaned.append("\n".join(kept))
    return cleaned


def _drop_page_numbers(text_body: str) -> str:
    """Remove lines that are just a number -- what is left of a footer after step 1."""
    return "\n".join(
        ln for ln in text_body.splitlines() if not re.fullmatch(r"\s*\d{1,4}\s*", ln)
    )


def _truncate_references(text_body: str) -> tuple[str, bool]:
    """Cut from the LAST references heading, so an in-body mention does not truncate."""
    matches = list(_REFS.finditer(text_body))
    if not matches:
        return text_body, False
    match = matches[-1]
    # Guard: if the heading is in the first 40% of the document it is a table of
    # contents entry, not the bibliography.
    if match.start() < len(text_body) * 0.4:
        return text_body, False
    return text_body[: match.start()].rstrip(), True


def extract(pdf_bytes: bytes) -> dict[str, Any]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 -- one unreadable page must not lose the rest
            pages.append("")

    pages = _strip_running_headers(pages)
    body = "\n\n".join(pages)
    body = _HYPHEN_BREAK.sub(r"\1\2", body)
    body = _drop_page_numbers(body)
    body, refs_cut = _truncate_references(body)
    body = _MULTISPACE.sub(" ", body)
    body = _MULTINEWLINE.sub("\n\n", body).strip()

    alpha = sum(c.isalpha() or c.isspace() for c in body)
    ratio = alpha / len(body) if body else 0.0
    ok = len(body) >= MIN_CHARS and ratio >= MIN_ALPHA_RATIO

    return {
        "text": body,
        "pages": len(reader.pages),
        "chars": len(body),
        "alpha_ratio": round(ratio, 3),
        "references_truncated": refs_cut,
        "status": "extracted" if ok else "poor",
    }


_PENDING = """
SELECT id::text, pdf_url, title
FROM articles
WHERE pdf_url IS NOT NULL
  AND doc_type = 'journal-article'
  AND processing_status <> 'archived'
  AND full_text_status NOT IN ('extracted', 'poor')
  -- A galley that failed three times is gone upstream (the one live example 404s).
  -- Retrying it on every daily run just adds ~10s of backoff noise to the cron log;
  -- the admin can clear processing_attempts to force another try after a fix.
  AND NOT (full_text_status = 'failed' AND processing_attempts >= 3)
ORDER BY published_at DESC NULLS LAST
"""


def extract_pending(db: Session, run_id: str | None = None, limit: int | None = None) -> dict[str, int]:
    rows = db.execute(sql(_PENDING)).all()
    if limit:
        rows = rows[:limit]
    stats = {"extracted": 0, "poor": 0, "failed": 0, "refs_truncated": 0}
    if not rows:
        _log("pdf_text: nothing to extract")
        return stats

    _log(f"pdf_text: {len(rows)} galley PDF(s) to fetch")
    with client() as c:
        for i, (article_id, pdf_url, title) in enumerate(rows, 1):
            try:
                response = get(c, pdf_url)
                content_type = response.headers.get("content-type", "")
                if "pdf" not in content_type.lower():
                    raise RuntimeError(f"expected a PDF, got {content_type!r}")

                result = extract(response.content)
                db.execute(
                    sql(
                        "UPDATE articles SET full_text = :t, full_text_status = :s, "
                        "word_count = :w, char_count = :c, updated_at = now() "
                        "WHERE id = CAST(:id AS uuid)"
                    ),
                    {
                        "t": result["text"] or None,
                        "s": result["status"],
                        "w": len(result["text"].split()),
                        "c": result["chars"],
                        "id": article_id,
                    },
                )
                db.commit()
                stats[result["status"]] += 1
                stats["refs_truncated"] += int(result["references_truncated"])
                if i % 20 == 0:
                    _log(f"pdf_text: {i}/{len(rows)}")
            except Exception as exc:  # noqa: BLE001 -- per-article: one dead galley
                                      # must not cost the other 92.
                db.rollback()
                stats["failed"] += 1
                db.execute(
                    sql(
                        "UPDATE articles SET full_text_status = 'failed', "
                        "processing_error = :e, processing_attempts = processing_attempts + 1 "
                        "WHERE id = CAST(:id AS uuid)"
                    ),
                    {"e": str(exc)[:2000], "id": article_id},
                )
                db.commit()
                _record_error(db, run_id, "pdf_text", None, article_id, str(exc))

    _log(f"pdf_text: {stats}")
    return stats
