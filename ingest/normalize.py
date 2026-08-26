"""
Raw upstream records -> canonical article dicts.

Everything fiddly about IJE's Dublin Core lives here, each piece commented, because
none of it is guessable from the field names alone.
"""

import hashlib
import re
from datetime import date, datetime
from typing import Any

from ingest.html_text import html_to_text, normalize_title
from ingest.language import resolve as resolve_language

# WordPress post type -> our doc_type vocabulary.
WP_DOC_TYPES = {
    "posts": "post",
    "media-coverage": "media-coverage",
    "opinion": "opinion",
    "archive": "archive",
    "podcast": "podcast",
    "e-newsletter": "e-newsletter",
    "book": "book",
    "brief-report": "brief-report",
    "works-archive": "works-archive",
    "infographics": "infographic",
}

# IJE gives "Purnomo Yusgiantoro Center"; WordPress gives nothing. Normalising here
# keeps the admin's publisher count meaningful instead of counting spelling variants.
PUBLISHER_ALIASES = {
    "pyc": "Purnomo Yusgiantoro Center",
    "purnomo yusgiantoro center": "Purnomo Yusgiantoro Center",
    "purnomo yusgiantoro centre": "Purnomo Yusgiantoro Center",
}

_VOL = re.compile(r"Vol\.?\s*(\d+)", re.I)
_NO = re.compile(r"No\.?\s*(\d+)", re.I)
_YEAR = re.compile(r"\((\d{4})\)")
_PAGES = re.compile(r"(\d+)\s*-\s*(\d+)\s*$")
_ISSN = re.compile(r"^\d{4}-\d{3}[\dxX]$")


def _norm_publisher(value: str | None) -> str | None:
    if not value:
        return None
    return PUBLISHER_ALIASES.get(value.strip().lower(), value.strip())


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    raw = value.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y/%m/%d", "%Y"):
        try:
            parsed = datetime.strptime(raw[: len(fmt) + 6] if "%z" in fmt else raw, fmt)
            # A 1970 or 0000 date is WordPress's "unset", not a real publication date.
            # NULL keeps it out of the by-year chart instead of inventing a 1970 bar.
            return parsed.date() if parsed.year > 1990 else None
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _content_hash(title: str, body: str) -> str:
    """Whitespace-normalised, case-folded fingerprint of the actual content."""
    joined = re.sub(r"\s+", " ", f"{title}\n{body}").strip().casefold()
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _dedup_key(title: str, published: date | None) -> str:
    """
    Fingerprint of the *opportunity*, not the record: significant title words
    (stopword-stripped, sorted) plus the year. Sorted so word-order variants collide;
    year-scoped so an annual report series does not collapse into one row.
    """
    stop = {
        "the", "a", "an", "of", "and", "in", "on", "for", "to", "dan", "di", "yang",
        "untuk", "pada", "dengan", "dari", "ke", "as", "at", "by",
    }
    words = sorted({w for w in normalize_title(title).split() if w not in stop and len(w) > 2})
    year = published.year if published else 0
    return hashlib.sha256((" ".join(words) + f"|{year}").encode("utf-8")).hexdigest()


def from_ije(record: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise one OAI record. Returns None for deletions."""
    if record.get("deleted"):
        return None

    external_id = record["external_id"]

    if record.get("empty"):
        # In the feed but carrying no oai_dc metadata. Stored as a visible failure so
        # the admin's Failed count is real rather than these vanishing silently.
        return {
            "external_id": external_id,
            "doc_type": "journal-article",
            "title": f"(no metadata) {external_id}",
            "processing_status": "failed",
            "processing_error": "OAI record contains no oai_dc metadata",
            "full_text_status": "none",
            "raw": record,
        }

    title = record.get("title") or "(untitled)"
    abstract = "\n\n".join(record.get("descriptions") or []).strip() or None

    # dc:identifier repeats: one article URL and one DOI. Classify by shape, never by
    # position -- the order is not guaranteed.
    url, doi = None, None
    for value in record.get("identifiers") or []:
        if value.startswith("10.") or "doi.org" in value:
            doi = value.split("doi.org/")[-1]
        elif value.startswith("http") and url is None:
            url = value

    # dc:relation holds the galley URL as .../article/view/{articleId}/{galleyId}.
    # /view/ returns an HTML viewer wrapper; /download/ returns the actual PDF. Getting
    # this backwards yields 95 "PDFs" that are all HTML.
    pdf_url, galley_id = None, None
    for value in record.get("relations") or []:
        match = re.search(r"/article/view/(\d+)/(\d+)", value)
        if match:
            galley_id = match.group(2)
            pdf_url = value.replace("/article/view/", "/article/download/")
            break

    # dc:source is "Journal; Vol. 1 No. 1 (2018): Title; 5-21" plus ISSNs plus an
    # issue DOI, all as separate repeated values.
    journal = volume = issue = pages = None
    issns: list[str] = []
    for value in record.get("sources") or []:
        if _ISSN.match(value.strip()):
            issns.append(value.strip())
            continue
        if value.startswith("10."):
            continue
        parts = [p.strip() for p in value.split(";")]
        if parts and not journal:
            journal = parts[0]
        for part in parts:
            if (m := _VOL.search(part)):
                volume = m.group(1)
            if (m := _NO.search(part)):
                issue = m.group(1)
            if (m := _PAGES.search(part)):
                pages = f"{m.group(1)}-{m.group(2)}"

    published = _parse_date((record.get("dates") or [None])[0])
    declared = (record.get("languages") or [None])[0]
    language, confidence, source = resolve_language(declared, abstract, url, title)

    return {
        "external_id": external_id,
        "doc_type": "journal-article",
        "title": title,
        "title_normalized": normalize_title(title),
        "abstract": abstract,
        "full_text": None,           # filled by the PDF stage
        "full_text_status": "abstract" if abstract else "none",
        "authors": record.get("creators") or [],
        "keywords": record.get("subjects") or [],
        "publisher": _norm_publisher((record.get("publishers") or [None])[0]),
        "journal": journal,
        "volume": volume,
        "issue": issue,
        "pages": pages,
        "doi": doi,
        "issn": issns,
        "published_at": published,
        "url": url,
        "pdf_url": pdf_url,
        "galley_id": galley_id,
        "language": language,
        "language_confidence": confidence,
        "language_source": source,
        "word_count": len((abstract or "").split()),
        "char_count": len(abstract or ""),
        "content_hash": _content_hash(title, abstract or ""),
        "dedup_key": _dedup_key(title, published),
        "processing_status": "pending",
        "raw": record,
    }


def from_wp(post: dict[str, Any]) -> dict[str, Any] | None:
    post_type = post.get("_post_type", "posts")
    external_id = str(post.get("id"))
    if not external_id or external_id == "None":
        return None

    title = html_to_text((post.get("title") or {}).get("rendered")) or "(untitled)"
    body = html_to_text((post.get("content") or {}).get("rendered"))
    excerpt = html_to_text((post.get("excerpt") or {}).get("rendered")) or None
    url = post.get("link")
    published = _parse_date(post.get("date"))
    language, confidence, source = resolve_language(None, body or excerpt, url, title)

    # 517 PYC documents carry no content.rendered at all -- media-coverage (374),
    # podcasts (72), works-archive (26) and infographics (14) are link-outs or embeds,
    # not articles. Most of them DO have an excerpt (media-coverage 297/374,
    # works-archive 26/26, averaging ~314 and ~522 characters), so the excerpt becomes
    # the retrievable body rather than the document being unsearchable.
    #
    # full_text_status distinguishes the two cases so the admin table shows the truth
    # and the reranker's doc_type prior can down-weight them:
    #   extracted -> a real body
    #   abstract  -> excerpt standing in for a body
    #   none      -> title only (podcasts, infographics); still findable by title
    retrievable = body or excerpt or None
    if body:
        status = "extracted"
    elif excerpt:
        status = "abstract"
    else:
        status = "none"

    return {
        "external_id": external_id,
        "doc_type": WP_DOC_TYPES.get(post_type, "post"),
        "title": title,
        "title_normalized": normalize_title(title),
        "abstract": excerpt,
        # WordPress needs no PDF stage: the body (or the excerpt) IS the full text.
        "full_text": retrievable,
        "full_text_status": status,
        "authors": [],
        "keywords": [],
        "publisher": "Purnomo Yusgiantoro Center",
        "journal": None,
        "volume": None,
        "issue": None,
        "pages": None,
        "doi": None,
        "issn": [],
        "published_at": published,
        "url": url,
        "pdf_url": None,
        "galley_id": None,
        "language": language,
        "language_confidence": confidence,
        "language_source": source,
        "word_count": len(retrievable.split()) if retrievable else 0,
        "char_count": len(retrievable) if retrievable else 0,
        "content_hash": _content_hash(title, retrievable or ""),
        "dedup_key": _dedup_key(title, published),
        "processing_status": "pending",
        "raw": post,
    }
