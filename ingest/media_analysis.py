"""
Media analysis: relevance + narrative + actors per (tracked issue, article).

Same shape as ingest/topics.py, for the same reasons: a cheap deterministic
prefilter, batched Haiku calls with a json_schema built from LIVE rows, DB writes
on the calling thread only, and degrade-not-raise everywhere.

The prefilter is what keeps the nightly bill flat. A crawl lands ~1,500 items of
which most are ordinary economy-desk noise; only articles that either matched an
issue keyword (title + summary) or arrived via that issue's Google News search go
to the model. Everything else gets a rule row (is_relevant = false) so the pending
query -- "no analysis row yet" -- never re-reads it.

Narratives are OPEN-WORLD, unlike topics: the model classifies against the enum of
existing media_narratives rows but may answer "new" with a proposed English label.
Proposals are capped per issue (MAX_LLM_NARRATIVES) because an unbounded taxonomy
converges on one narrative per article, which is no taxonomy at all.
"""

import json
import re
from typing import Any

from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    LLM_CONCURRENCY,
    MEDIA_LOOKBACK_DAYS,
    MEDIA_RELEVANCE_THRESHOLD,
)
from app.logging_utils import _log

BATCH_SIZE = 8
MAX_LLM_NARRATIVES = 12

_SYSTEM = """\
You are a media analyst for an Indonesian energy think tank (PYC). You monitor
Indonesian news coverage of one tracked policy issue. For each numbered news item
(title, optional summary, outlet) you judge:

1. relevance: 0-1. How directly is this item about the tracked issue described in
   <issue>? 1.0 = squarely about it; 0.5 = discusses it as one of several topics;
   below 0.3 = passing mention or unrelated. Judge by MEANING; headlines are
   usually Indonesian.
2. narrative: which framing from <narratives> the item advances. Use "none" when
   the item is irrelevant or fits no listed framing. Use "new" ONLY for a framing
   that is clearly recurring and distinct from every listed one -- then give a
   short English new_narrative_label (3-5 words, title case).
3. actors: institutions and named public figures the item quotes or discusses,
   as canonical short names, e.g. "Kementerian ESDM", "Kementerian Keuangan",
   "Pertamina", "PLN", "DPR RI", "BPH Migas", "Presiden Prabowo". Use the
   institution name, not job titles; at most 6; empty when irrelevant."""


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return slug[:60] or "unnamed"


def _issues(db: Session) -> list[dict[str, Any]]:
    rows = db.execute(
        sql(
            "SELECT id, slug, name, description, keywords FROM tracked_issues "
            "WHERE is_active ORDER BY id"
        )
    ).mappings().all()
    return [dict(r) for r in rows]


def _narratives(db: Session, issue_id: int) -> list[dict[str, Any]]:
    rows = db.execute(
        sql(
            "SELECT id, slug, label, description, created_by FROM media_narratives "
            "WHERE issue_id = :i ORDER BY id"
        ),
        {"i": issue_id},
    ).mappings().all()
    return [dict(r) for r in rows]


# Candidates: inside the issue's monitored outlet set, inside the lookback window,
# not yet analysed for this issue. The keyword/google prefilter happens in Python
# (arrays of Indonesian phrases do not make a sane SQL predicate).
_PENDING = """
SELECT a.id::text, a.title, a.summary, a.raw->>'keyword' AS via_keyword,
       o.name AS outlet_name
FROM media_articles a
JOIN media_outlets o ON o.id = a.outlet_id
JOIN tracked_issue_outlets tio ON tio.outlet_id = a.outlet_id AND tio.issue_id = :issue
LEFT JOIN media_article_analysis an
       ON an.article_id = a.id AND an.issue_id = :issue
WHERE an.article_id IS NULL
  AND a.published_at > now() - make_interval(days => :lookback)
ORDER BY a.published_at DESC
"""


def _keyword_hits(article: dict[str, Any], keywords: list[str]) -> int:
    haystack = f"{article.get('title') or ''} {article.get('summary') or ''}".lower()
    return sum(1 for kw in keywords if kw and kw.lower() in haystack)


def _schema(slugs: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "integer"},
                        "relevance": {"type": "number"},
                        "narrative": {"type": "string", "enum": [*slugs, "none", "new"]},
                        "new_narrative_label": {"type": "string"},
                        "actors": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["item", "relevance", "narrative", "actors"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def _batch_message(issue: dict, narratives: list[dict], batch: list[dict]) -> str:
    parts = [
        "<issue>",
        f"{issue['name']}: {issue['description'] or ''}",
        f"keywords: {', '.join(issue['keywords'] or [])}",
        "</issue>",
        "<narratives>",
        *[f"{n['slug']}: {n['label']}. {n['description'] or ''}" for n in narratives],
        "</narratives>",
        "",
    ]
    for i, article in enumerate(batch):
        parts.append(f"--- item {i} ---")
        parts.append(f"title: {(article['title'] or '')[:300]}")
        if article.get("summary"):
            parts.append(f"summary: {article['summary'][:600]}")
        parts.append(f"outlet: {article.get('outlet_name') or '?'}")
        parts.append("")
    return "\n".join(parts)


def _analyze_batch(issue: dict, narratives: list[dict], batch: list[dict]) -> list[dict] | None:
    """One Haiku call for up to BATCH_SIZE articles. None on failure."""
    from rag.generator import client

    slugs = [n["slug"] for n in narratives]
    try:
        response = client().messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1500,
            system=_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": _schema(slugs)}},
            messages=[{"role": "user", "content": _batch_message(issue, narratives, batch)}],
        )
        if response.stop_reason == "refusal":
            return None
        payload = json.loads(next(b.text for b in response.content if b.type == "text"))
        return payload.get("items") or []
    except Exception as exc:  # noqa: BLE001 -- one failed batch degrades to the
                              # rule row; it must not stop the other batches.
        _log(f"media-analysis: batch failed ({exc})")
        return None


_UPSERT_ANALYSIS = sql(
    """
    INSERT INTO media_article_analysis
        (issue_id, article_id, relevance, is_relevant, narrative_id, actors, analysis_source)
    VALUES
        (:issue, CAST(:article AS uuid), :rel, :is_rel, :narrative, :actors, :src)
    ON CONFLICT (issue_id, article_id) DO UPDATE SET
        relevance = EXCLUDED.relevance, is_relevant = EXCLUDED.is_relevant,
        narrative_id = EXCLUDED.narrative_id, actors = EXCLUDED.actors,
        analysis_source = EXCLUDED.analysis_source, analyzed_at = now()
    """
)


def _write(db: Session, issue_id: int, article_id: str, relevance: float,
           narrative_id: int | None, actors: list[str], source: str) -> None:
    relevance = max(0.0, min(1.0, relevance))
    db.execute(
        _UPSERT_ANALYSIS,
        {
            "issue": issue_id, "article": article_id, "rel": round(relevance, 3),
            "is_rel": relevance >= MEDIA_RELEVANCE_THRESHOLD,
            "narrative": narrative_id,
            "actors": [a.strip()[:120] for a in actors if a and a.strip()][:6],
            "src": source,
        },
    )


def _ensure_narrative(db: Session, issue_id: int, label: str,
                      by_slug: dict[str, dict]) -> int | None:
    """Create (or fetch) an LLM-proposed narrative, honouring the cap."""
    slug = _slugify(label)
    if slug in by_slug:
        return by_slug[slug]["id"]
    llm_count = sum(1 for n in by_slug.values() if n["created_by"] == "llm")
    if llm_count >= MAX_LLM_NARRATIVES:
        return None
    narrative_id = db.execute(
        sql(
            "INSERT INTO media_narratives (issue_id, slug, label, created_by) "
            "VALUES (:i, :s, :l, 'llm') "
            "ON CONFLICT (issue_id, slug) DO UPDATE SET label = media_narratives.label "
            "RETURNING id"
        ),
        {"i": issue_id, "s": slug, "l": label.strip()[:120]},
    ).scalar()
    by_slug[slug] = {"id": narrative_id, "slug": slug, "label": label,
                     "description": None, "created_by": "llm"}
    _log(f"media-analysis: new narrative '{label}' ({slug}) for issue {issue_id}")
    return narrative_id


def analyze_pending(db: Session, limit: int | None = None) -> dict[str, int]:
    """Analyse every un-analysed (issue, article) pair. Safe to re-run."""
    from concurrent.futures import ThreadPoolExecutor

    stats = {"llm": 0, "rule_skipped": 0, "rule_fallback": 0, "batches_failed": 0,
             "narratives_created": 0}

    for issue in _issues(db):
        rows = [
            dict(r)
            for r in db.execute(
                sql(_PENDING), {"issue": issue["id"], "lookback": MEDIA_LOOKBACK_DAYS}
            ).mappings().all()
        ]
        if limit:
            rows = rows[:limit]
        if not rows:
            continue

        keywords = issue["keywords"] or []
        issue_kw = {k.lower() for k in keywords}
        candidates, noise = [], []
        for article in rows:
            via = (article.get("via_keyword") or "").lower()
            if via in issue_kw or _keyword_hits(article, keywords) > 0:
                candidates.append(article)
            else:
                noise.append(article)

        # Noise gets a rule row so it is never fetched again -- relevance 0, no
        # model involved. This is the bulk of every feed crawl.
        for article in noise:
            _write(db, issue["id"], article["id"], 0.0, None, [], "rule")
            stats["rule_skipped"] += 1
        db.commit()

        narratives = _narratives(db, issue["id"])
        by_slug = {n["slug"]: n for n in narratives}

        def _fallback(batch: list[dict]) -> None:
            """Keyword-count relevance, no narrative, no actors."""
            for article in batch:
                hits = _keyword_hits(article, keywords) + (1 if article.get("via_keyword") else 0)
                _write(db, issue["id"], article["id"],
                       0.55 if hits >= 2 else 0.35, None, [], "rule")
                stats["rule_fallback"] += 1

        if not ANTHROPIC_API_KEY:
            _log("media-analysis: no ANTHROPIC_API_KEY; keyword fallback for everything")
            _fallback(candidates)
            db.commit()
            continue

        batches = [candidates[i : i + BATCH_SIZE] for i in range(0, len(candidates), BATCH_SIZE)]
        _log(
            f"media-analysis: issue '{issue['slug']}' -> {len(candidates)} candidate(s) "
            f"in {len(batches)} batch(es), {len(noise)} filtered by rule"
        )

        # Model calls run concurrently; ALL DB writes stay on this thread (the
        # session is not thread-safe -- same bargain as ingest/topics.py).
        with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
            for batch, result in zip(
                batches, pool.map(lambda b: _analyze_batch(issue, list(by_slug.values()), b), batches)
            ):
                if result is None:
                    stats["batches_failed"] += 1
                    _fallback(batch)
                    db.commit()
                    continue
                by_item = {r.get("item"): r for r in result}
                for i, article in enumerate(batch):
                    verdict = by_item.get(i)
                    if not verdict:
                        _fallback([article])
                        continue
                    relevance = float(verdict.get("relevance") or 0)
                    narrative_id = None
                    choice = verdict.get("narrative") or "none"
                    if relevance >= MEDIA_RELEVANCE_THRESHOLD:
                        if choice == "new" and verdict.get("new_narrative_label"):
                            before = len(by_slug)
                            narrative_id = _ensure_narrative(
                                db, issue["id"], verdict["new_narrative_label"], by_slug
                            )
                            stats["narratives_created"] += len(by_slug) - before
                        elif choice in by_slug:
                            narrative_id = by_slug[choice]["id"]
                    _write(db, issue["id"], article["id"], relevance, narrative_id,
                           verdict.get("actors") or [], "llm")
                    stats["llm"] += 1
                db.commit()

    db.commit()
    _log(f"media-analysis: {stats}")
    return stats
