"""Conservative extraction of explicit metadata constraints from retrieval queries.

This is deliberately deterministic. Metadata constraints change which records are
eligible, so an LLM guess is not safe enough. The extractor recognizes publication
years/ranges plus explicit source, author, topic/category, document-type and language
phrasing. Unmarked subject words remain part of semantic retrieval.
"""

from __future__ import annotations

import re
from datetime import date

from sqlalchemy import text as sql

from rag.filters import Filters

_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
_DATE_SIGNAL = re.compile(
    r"\b(published|publication|publication_year|year|released|dated|terbit|diterbitkan|dipublikasikan|"
    r"tahun|sejak|since|before|after|between|until|hingga|sampai)\b",
    re.IGNORECASE,
)
_DOCUMENT = re.compile(
    r"\b(articles?|papers?|publications?|documents?|reports?|journals?|"
    r"artikel|publikasi|dokumen|laporan|jurnal)\b",
    re.IGNORECASE,
)
_FROM = {"since", "after", "from", "sejak", "setelah", "mulai"}
_TO = {"before", "until", "to", "sebelum", "hingga", "sampai"}
_YEAR_FIELD = re.compile(r"\b(?:publication_year|year|tahun)\s*[:=]\s*(19[89]\d|20[0-4]\d)\b", re.I)
_DATE_FIELD = re.compile(
    r"\b(date_from|from_date|date_to|to_date)\s*[:=]\s*(\d{4}-\d{2}-\d{2})\b",
    re.I,
)


def _copy(filters: Filters | None) -> Filters:
    source = filters or Filters()
    return Filters(
        source_slugs=list(source.source_slugs),
        authors=list(source.authors),
        doc_types=list(source.doc_types),
        topics=list(source.topics),
        languages=list(source.languages),
        date_from=source.date_from,
        date_to=source.date_to,
    )


def _publication_years(query: str) -> tuple[int | None, int | None]:
    field = _YEAR_FIELD.search(query)
    if field:
        year = int(field.group(1))
        return year, year
    matches = list(_YEAR.finditer(query))
    # A bare year is metadata only in an article-listing/search construction. This
    # avoids treating subjects such as "Indonesia 2045 strategy" as publication dates.
    if not matches or not (
        _DATE_SIGNAL.search(query) or _DOCUMENT.search(query)
    ):
        return None, None
    years = sorted(int(match.group()) for match in matches)
    if len(years) > 1:
        return years[0], years[-1]
    year = years[0]
    preceding = query[: matches[0].start()].lower().split()[-3:]
    if any(word in _FROM for word in preceding):
        return year, None
    if any(word in _TO for word in preceding):
        return None, year
    return year, year


def _field_values(query: str, names: str) -> list[str]:
    """Read `field:value` / `field:"multi word value"` without guessing boundaries."""
    pattern = re.compile(
        rf"\b(?:{names})\s*:\s*(?:\"([^\"]+)\"|'([^']+)'|([\w.-]+))",
        re.IGNORECASE,
    )
    return [next(group for group in match.groups() if group).strip() for match in pattern.finditer(query)]


def _known_metadata(db, query: str, marker: str, statement: str) -> list[str]:
    """Resolve human labels only when introduced by an explicit metadata marker."""
    lowered = query.casefold()
    if not re.search(marker, lowered, re.IGNORECASE):
        return []
    rows = db.execute(sql(statement)).all()
    matches: list[tuple[int, str]] = []
    for row in rows:
        slug, *labels = row
        for label in (slug, *labels):
            label_text = str(label or "").strip()
            acronym = "".join(word[0] for word in label_text.split() if word).casefold()
            variants = {label_text.casefold()}
            if len(acronym) >= 2:
                variants.add(acronym)
            if label_text and any(
                re.search(rf"(?<!\w){re.escape(value)}(?!\w)", lowered)
                for value in variants
            ):
                matches.append((len(label_text), str(slug)))
                break
    # Prefer the longest name if a short slug is contained in it.
    return [max(matches)[1]] if matches else []


def _resolve_values(db, values: list[str], statement: str) -> list[str]:
    """Turn explicit human labels/acronyms into canonical metadata slugs."""
    rows = db.execute(sql(statement)).all()
    aliases: dict[str, str] = {}
    for row in rows:
        slug, *labels = row
        for label in (slug, *labels):
            text = str(label or "").strip()
            if not text:
                continue
            aliases[text.casefold()] = str(slug)
            acronym = "".join(word[0] for word in text.split() if word)
            if len(acronym) >= 2:
                aliases[acronym.casefold()] = str(slug)
    return [aliases.get(value.casefold(), value) for value in values]


def extract_explicit_filters(db, query: str, filters: Filters | None = None) -> Filters:
    """Merge query-stated constraints with filter-bar constraints.

    A query-stated value replaces the same UI dimension (the newest explicit user
    instruction), while all unrelated UI constraints remain active.
    """
    merged = _copy(filters)
    year_from, year_to = _publication_years(query)
    if year_from is not None or year_to is not None:
        merged.date_from = date(year_from, 1, 1) if year_from else None
        merged.date_to = date(year_to, 12, 31) if year_to else None
    for field, value in _DATE_FIELD.findall(query):
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            continue
        if field.casefold() in {"date_from", "from_date"}:
            merged.date_from = parsed
        else:
            merged.date_to = parsed

    sources = _field_values(query, r"source|sumber")
    if sources:
        merged.source_slugs = _resolve_values(db, sources, "SELECT slug, name FROM sources")
    else:
        known_sources = _known_metadata(
            db,
            query,
            r"\b(?:from|dari|source|sumber)\b",
            "SELECT slug, name FROM sources",
        )
        if known_sources:
            merged.source_slugs = known_sources

    authors = _field_values(query, r"author|authors|penulis")
    if not authors:
        match = re.search(
            r"\b(?:written\s+by|authored\s+by|by|ditulis\s+oleh|oleh)\s+"
            r"[\"']([^\"']+)[\"']",
            query,
            re.IGNORECASE,
        )
        authors = [match.group(1).strip()] if match else []
    if authors:
        merged.authors = authors

    topics = _field_values(query, r"topic|category|topik|kategori")
    if topics:
        merged.topics = _resolve_values(
            db,
            topics,
            "SELECT slug, label_en, label_id FROM topics WHERE is_active",
        )
    else:
        known_topics = _known_metadata(
            db,
            query,
            r"\b(?:topic|category|topik|kategori)\b",
            "SELECT slug, label_en, label_id FROM topics WHERE is_active",
        )
        if known_topics:
            merged.topics = known_topics

    doc_types = _field_values(query, r"type|doc_type|document_type|jenis")
    if doc_types:
        merged.doc_types = doc_types
    languages = _field_values(query, r"language|lang|bahasa")
    if languages:
        aliases = {"english": "en", "inggris": "en", "indonesian": "id", "indonesia": "id"}
        merged.languages = [aliases.get(value.casefold(), value.casefold()) for value in languages]
    return merged
