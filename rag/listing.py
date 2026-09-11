"""
Deterministic handling of metadata LISTING queries -- "list artikel di 2018",
"daftar publikasi tentang solar", "show all journal articles since 2020".

These questions are about the CATALOGUE, not the content. No chunk of prose is
semantically similar to "list the 2018 articles", so the semantic evidence gate
(rag/retriever.has_evidence) refuses them -- correctly by its own contract, uselessly
for the user, because the database can answer them exactly. This module is the routing
branch that lets it: detected listings are answered from article metadata, with no LLM
call at all, so there is nothing to hallucinate and nothing to gate.

Detection is deliberately deterministic and CONSERVATIVE:

* A regex, not an LLM intent classifier. A false positive hijacks a real research
  question into a catalogue dump -- the worst direction to fail in -- so the pattern
  requires an explicit listing verb (list/daftar/sebutkan/...) at most two words
  before a document noun (artikel/publikasi/paper/...), or the Indonesian
  "<noun> apa saja" form. "Sebutkan tantangan utama dalam laporan 2023" does NOT
  match (three words between verb and noun) and stays a research question; that
  asymmetry is intended.
* Year handling understands single years, ranges ("2018-2020", "antara 2018 dan
  2020"), and open intervals ("sejak 2018", "before 2020").

A listing with a topical remainder ("list artikel tentang solar di 2018") still goes
through hybrid retrieval -- the topic decides WHICH articles, and retrieval is what
knows that -- but the result is collapsed to unique articles and formatted as a
catalogue, and the same MIN_SEMANTIC_SIMILARITY floor applies so an off-corpus topic
lists nothing rather than the least-bad matches.

The answer text is markdown-lite on purpose: the frontend answer renderer understands
exactly **bold**, [n] citation pills, and blank-line paragraph breaks (see the
frontend's lib/citations.ts) -- so each article is one paragraph carrying its own [n],
and the pill wires it to the evidence panel like any cited source.
"""

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.config import (
    ENABLE_QUERY_TRANSLATION,
    MIN_SEMANTIC_SIMILARITY,
    TRANSLATE_FROM_LANGUAGES,
)
from rag.filters import BASE_PREDICATE, Filters, build

# Catalogue answers are bounded: 50 paragraphs is already a long scroll, and the
# header always states the true total so a cap never masquerades as completeness.
LIST_LIMIT = 50

_VERB = r"(?:list|daftar|sebutkan|tampilkan|tunjukkan|show|enumerate)"
_NOUN = (
    r"(?:artikel|articles?|publikasi|publications?|papers?|jurnal|journals?"
    r"|dokumen|documents?|laporan|reports?|terbitan)"
)

# Verb, at most two intervening words, then a document noun. The {0,2} window is the
# false-positive guard discussed in the module docstring.
_VERB_NOUN = re.compile(rf"\b{_VERB}\b(?:\s+\S+){{0,2}}?\s+({_NOUN})\b", re.IGNORECASE)
# "artikel apa saja (yang ...)" -- the Indonesian listing idiom with no verb.
_APA_SAJA = re.compile(rf"\b({_NOUN})\b\s+(?:\S+\s+)?apa\s+saja\b", re.IGNORECASE)

_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
_FROM_MARKERS = {"sejak", "since", "setelah", "after", "mulai", "from"}
_TO_MARKERS = {"sebelum", "before", "hingga", "sampai", "until"}

# Words that carry no topical meaning once the listing frame is removed. Includes the
# verbs and nouns themselves, because either may occur again outside the matched span.
_FILLER = {
    # id
    "di", "pada", "tahun", "dalam", "yang", "dan", "atau", "tentang", "mengenai",
    "terkait", "semua", "seluruh", "saja", "apa", "ada", "dari", "ke", "untuk",
    "oleh", "dengan", "koleksi", "terbit", "diterbitkan", "dipublikasikan",
    "ditulis", "dirilis", "membahas", "dibahas", "tolong", "mohon", "bisa",
    "berikan", "buatkan", "daftar", "sebutkan", "tampilkan", "tunjukkan",
    "artikel", "publikasi", "jurnal", "dokumen", "laporan", "terbitan",
    "sejak", "setelah", "mulai", "sebelum", "hingga", "sampai", "antara",
    # en
    "in", "on", "of", "the", "a", "an", "all", "any", "from", "to", "about",
    "since", "after", "before", "until",
    "regarding", "published", "written", "released", "during", "between", "and",
    "or", "collection", "me", "please", "can", "you", "have", "we", "do", "list",
    "show", "enumerate", "discuss", "discussing", "covering", "covers", "cover",
    "article", "articles", "publication", "publications", "paper", "papers",
    "journal", "journals", "document", "documents", "report", "reports",
    # product names
    "pikis", "pyc",
}

# Indonesian surface markers. `ingest.language.detect` is tuned for prose and is
# unreliable on a 5-word query, but a listing query that uses the Indonesian verb or
# noun forms has already told us its language.
_ID_MARKERS = re.compile(
    r"\b(daftar|sebutkan|tampilkan|tunjukkan|artikel|publikasi|dokumen|laporan"
    r"|terbitan|tahun|tentang|mengenai|yang|pada|di|apa\s+saja)\b",
    re.IGNORECASE,
)


@dataclass
class ListingIntent:
    year_from: int | None = None
    year_to: int | None = None
    topic_text: str = ""
    language: str = "en"


def _years(question: str) -> tuple[int | None, int | None]:
    matches = list(_YEAR.finditer(question))
    if not matches:
        return None, None
    years = sorted(int(m.group(0)) for m in matches)
    if len(years) >= 2:
        return years[0], years[-1]
    year = years[0]
    # Only the two words immediately before the year can direct the interval:
    # "sejak tahun 2018", "before 2020".
    preceding = question[: matches[0].start()].lower().split()[-2:]
    if any(w in _FROM_MARKERS for w in preceding):
        return year, None
    if any(w in _TO_MARKERS for w in preceding):
        return None, year
    return year, year


def _topic_remainder(question: str) -> str:
    text = question.lower()
    text = _VERB_NOUN.sub(" ", text)
    text = _APA_SAJA.sub(" ", text)
    text = _YEAR.sub(" ", text)
    words = [w for w in re.findall(r"[a-zà-ɏ0-9][a-zà-ɏ0-9-]*", text) if w not in _FILLER]
    # A remainder of only glue fragments ("me", "yg") is noise, not a topic.
    if not any(len(w) >= 4 for w in words):
        return ""
    return " ".join(words)


def _language(question: str) -> str:
    if _ID_MARKERS.search(question):
        return "id"
    from ingest.language import detect

    language, _ = detect(question)
    return language or "en"


def detect_listing(question: str) -> ListingIntent | None:
    """Return a ListingIntent when the question is a catalogue ask, else None."""
    q = (question or "").strip()
    # A long paragraph is a research question that happens to contain "list ...";
    # catalogue asks are short.
    if not q or len(q) > 300:
        return None
    if not (_VERB_NOUN.search(q) or _APA_SAJA.search(q)):
        return None
    year_from, year_to = _years(q)
    return ListingIntent(
        year_from=year_from,
        year_to=year_to,
        topic_text=_topic_remainder(q),
        language=_language(q),
    )


# --- execution ---------------------------------------------------------------

_LIST_SQL = (
    "SELECT a.id::text AS article_id, a.title, a.publisher, a.published_at, a.url, "
    "a.doc_type, a.language, a.journal, a.abstract, "
    "s.slug AS source_slug, s.name AS source_name "
    "FROM articles a JOIN sources s ON s.id = a.source_id "
    f"WHERE {BASE_PREDICATE}{{fragment}} "
    "ORDER BY a.published_at DESC NULLS LAST, a.title LIMIT :lim"
)
_COUNT_SQL = (
    "SELECT count(*) FROM articles a JOIN sources s ON s.id = a.source_id "
    f"WHERE {BASE_PREDICATE}{{fragment}}"
)


def _merge_filters(intent: ListingIntent, filters: Filters | None) -> Filters:
    """UI filters carry over; years stated IN the question override the date range,
    because the question is the more recent expression of intent."""
    f = filters or Filters()
    merged = Filters(
        source_slugs=list(f.source_slugs),
        authors=list(f.authors),
        doc_types=list(f.doc_types),
        topics=list(f.topics),
        languages=list(f.languages),
        date_from=f.date_from,
        date_to=f.date_to,
    )
    if intent.year_from:
        merged.date_from = date(intent.year_from, 1, 1)
    if intent.year_to:
        merged.date_to = date(intent.year_to, 12, 31)
    return merged


def _catalogue_rows(db: Session, merged: Filters) -> tuple[list[dict[str, Any]], int]:
    fragment, params = build(merged)
    total = int(db.execute(sql(_COUNT_SQL.format(fragment=fragment)), params).scalar() or 0)
    rows = db.execute(
        sql(_LIST_SQL.format(fragment=fragment)), {**params, "lim": LIST_LIMIT}
    ).mappings().all()
    return [dict(r) for r in rows], total


def _topical_rows(db: Session, intent: ListingIntent, merged: Filters) -> tuple[list[dict[str, Any]], int]:
    """Hybrid retrieval decides WHICH articles match the topic; the result is then
    collapsed to one row per article, best chunk first."""
    from rag.generator import rewrite_query
    from rag.retriever import retrieve

    extra: list[str] = []
    if ENABLE_QUERY_TRANSLATION:
        rewritten = rewrite_query(intent.topic_text)
        if rewritten.query_translated and rewritten.language in TRANSLATE_FROM_LANGUAGES:
            extra.append(rewritten.query_translated)

    result = retrieve(db, intent.topic_text, filters=merged, top_k=LIST_LIMIT, extra_queries=extra or None)
    # The same absolute floor as chat's Gate 1: an off-corpus topic must list
    # nothing, not the least-bad matches.
    if result.best_similarity() < MIN_SEMANTIC_SIMILARITY:
        return [], 0

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for c in result.candidates:
        article_id = c.get("article_id")
        if article_id in seen:
            continue
        seen.add(article_id)
        rows.append(c)
    return rows, len(rows)


def _scope_phrase(intent: ListingIntent) -> str:
    indonesian = intent.language == "id"
    parts: list[str] = []
    a, b = intent.year_from, intent.year_to
    if a and b and a == b:
        parts.append(f"yang terbit pada tahun {a}" if indonesian else f"published in {a}")
    elif a and b:
        parts.append(
            f"yang terbit antara {a} dan {b}" if indonesian else f"published between {a} and {b}"
        )
    elif a:
        parts.append(f"yang terbit sejak {a}" if indonesian else f"published since {a}")
    elif b:
        parts.append(f"yang terbit sampai {b}" if indonesian else f"published up to {b}")
    if intent.topic_text:
        parts.append(
            f'tentang "{intent.topic_text}"' if indonesian else f'about "{intent.topic_text}"'
        )
    return (" " + " ".join(parts)) if parts else ""


def _format(intent: ListingIntent, rows: list[dict[str, Any]], total: int) -> tuple[str, list[dict[str, Any]]]:
    indonesian = intent.language == "id"
    scope = _scope_phrase(intent)

    if not rows:
        answer = (
            f"Tidak ditemukan artikel{scope} dalam koleksi PIKIS."
            if indonesian
            else f"No articles{scope} were found in the PIKIS collection."
        )
        return answer, []

    if indonesian:
        header = f"Terdapat {total} artikel{scope} dalam koleksi PIKIS:"
        if total > len(rows):
            header += f" (menampilkan {len(rows)} pertama)"
    else:
        header = f"There are {total} articles{scope} in the PIKIS collection:"
        if total > len(rows):
            header += f" (showing the first {len(rows)})"

    paragraphs = [header]
    citations: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        published = row.get("published_at")
        published_str = published.isoformat() if published else None
        # '*' inside a title would open a bold span the renderer never closes.
        title = (row.get("title") or "(untitled)").replace("*", "")
        publisher = row.get("publisher") or row.get("source_name") or ""
        line = f"**{index}. {title}**"
        details = ", ".join(p for p in [publisher, published_str] if p)
        if details:
            line += f" — {details}"
        if row.get("doc_type"):
            line += f" · {row['doc_type'].replace('-', ' ')}"
        line += f" [{index}]"
        paragraphs.append(line)

        citations.append(
            {
                "index": index,
                "article_id": row.get("article_id"),
                "chunk_id": row.get("id"),  # present on topical rows, None otherwise
                "title": row.get("title"),
                "publisher": row.get("publisher"),
                "source": row.get("journal") or row.get("source_name"),
                "source_slug": row.get("source_slug"),
                "published_at": published_str,
                "url": row.get("url"),
                "doc_type": row.get("doc_type"),
                "language": row.get("language"),
                "section": None,
                "snippet": (row.get("abstract") or row.get("content") or "")[:300],
                "char_start": None,
                "char_end": None,
                "score": round(float(row.get("rerank") or 0.0), 5),
            }
        )

    return "\n\n".join(paragraphs), citations


def run_listing(db: Session, intent: ListingIntent, filters: Filters | None) -> dict[str, Any]:
    """Execute a detected listing and return {answer, citations, total, shown, filters}."""
    merged = _merge_filters(intent, filters)
    if intent.topic_text:
        rows, total = _topical_rows(db, intent, merged)
    else:
        rows, total = _catalogue_rows(db, merged)
    answer, citations = _format(intent, rows, total)
    return {
        "answer": answer,
        "citations": citations,
        "total": total,
        "shown": len(rows),
        "filters": merged.describe(),
        "topic": intent.topic_text or None,
    }
