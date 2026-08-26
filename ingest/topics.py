"""
Topic classification: keyword prior + Haiku decision.

Pure rules fail on this corpus for a measurable reason -- it is mono-topical. Every
document is about energy, so keyword lists fire everywhere and the highest-count
match is noise. The sibling crawler hit the same wall with tender relevance and
landed on the same shape used here: compute a cheap deterministic prior, hand it to
the model EXPLICITLY LABELLED AS OVERRULABLE, and let the model decide.

The taxonomy is read from the `topics` TABLE at call time, never hardcoded: the
json_schema enum is built from `SELECT slug FROM topics WHERE is_active`, so an admin
adding a topic changes classification behaviour without a code release.

Articles are classified in BATCHES of 8 per call rather than one call each. At
~1,380 articles that is ~170 calls instead of ~1,380 -- the taxonomy block is resent
per call either way (it is ~1,300 tokens, under Haiku's 4,096-token caching floor,
so cache_control would be silently ignored; batching is what actually cuts the
repeated-prefix cost). Total run cost lands around $1.

Failures degrade, never raise: a batch that errors falls back to the rule prior with
topic_source='rule' and low confidence, which the admin table surfaces.
"""

import json
from typing import Any

from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, LLM_CONCURRENCY
from app.logging_utils import _log

BATCH_SIZE = 8

_SYSTEM = """\
You classify publications from an Indonesian energy think tank (PYC) into an energy
topic taxonomy. For each numbered item you receive a title, an optional abstract,
optional author keywords, and a crude keyword-match prior.

The prior is a keyword count, not a judgement -- overrule it freely. Titles are often
in Indonesian; classify by MEANING, not by which language the words are in.

Boundary rules that matter in this corpus:
- A paper on coal-plant EMISSIONS or capture belongs in energy-transition, not
  coal-mining. coal-mining is about producing and trading the fuel.
- Financing, investment, tariffs and project economics go to energy-economics even
  when the project being financed is renewable.
- Grid integration, electrification and EV charging go to power-grid even when the
  electricity is renewable.
- Institutional news (event recaps, visits, anniversaries, competitions) with no
  substantive energy content is `other`.

Pick exactly one primary topic per item, up to two secondaries when genuinely
warranted, and a 0-1 confidence in the primary."""


def taxonomy(db: Session) -> list[dict[str, Any]]:
    rows = db.execute(
        sql(
            "SELECT id, slug, label_en, description, keywords FROM topics "
            "WHERE is_active ORDER BY sort_order"
        )
    ).mappings().all()
    return [dict(r) for r in rows]


def rule_prior(title: str, abstract: str | None, keywords: list[str], topics: list[dict]) -> list[tuple[str, int]]:
    """Cheap keyword-count prior over title + abstract + author keywords."""
    haystack = " ".join([title or "", abstract or "", " ".join(keywords or [])]).lower()
    scores = []
    for topic in topics:
        hits = sum(1 for kw in (topic["keywords"] or []) if kw and kw.lower() in haystack)
        if hits:
            scores.append((topic["slug"], hits))
    scores.sort(key=lambda pair: -pair[1])
    return scores[:3]


def _schema(slugs: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "integer"},
                        "primary": {"type": "string", "enum": slugs},
                        "secondary": {
                            "type": "array",
                            "items": {"type": "string", "enum": slugs},
                        },
                        "confidence": {"type": "number"},
                    },
                    "required": ["item", "primary", "secondary", "confidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["classifications"],
        "additionalProperties": False,
    }


def _taxonomy_block(topics: list[dict]) -> str:
    lines = ["<taxonomy>"]
    for t in topics:
        lines.append(f"{t['slug']}: {t['label_en']}. {t['description'] or ''}")
    lines.append("</taxonomy>")
    return "\n".join(lines)


def _batch_message(batch: list[dict], topics: list[dict]) -> str:
    parts = [_taxonomy_block(topics), ""]
    for i, article in enumerate(batch):
        prior = rule_prior(article["title"], article.get("abstract"), article.get("keywords") or [], topics)
        parts.append(f"--- item {i} ---")
        parts.append(f"title: {article['title'][:300]}")
        if article.get("abstract"):
            parts.append(f"abstract: {article['abstract'][:900]}")
        if article.get("keywords"):
            parts.append(f"author_keywords: {', '.join(article['keywords'][:10])}")
        parts.append(f"doc_type: {article.get('doc_type')}")
        if prior:
            parts.append(
                "keyword_prior (crude signal, not a judgement -- overrule freely): "
                + ", ".join(f"{slug}({hits})" for slug, hits in prior)
            )
        parts.append("")
    return "\n".join(parts)


def _classify_batch(batch: list[dict], topics: list[dict]) -> list[dict[str, Any]] | None:
    """One Haiku call for up to BATCH_SIZE articles. None on failure."""
    from rag.generator import client

    slugs = [t["slug"] for t in topics]
    try:
        response = client().messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1200,
            system=_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": _schema(slugs)}},
            messages=[{"role": "user", "content": _batch_message(batch, topics)}],
        )
        if response.stop_reason == "refusal":
            return None
        payload = json.loads(next(b.text for b in response.content if b.type == "text"))
        return payload.get("classifications") or []
    except Exception as exc:  # noqa: BLE001 -- one failed batch degrades to the rule
                              # prior; it must not stop the other ~170 batches.
        _log(f"topics: batch failed ({exc})")
        return None


def _apply(db: Session, article: dict, primary_slug: str, secondary: list[str],
           confidence: float, source: str, topic_ids: dict[str, int]) -> None:
    primary_id = topic_ids.get(primary_slug) or topic_ids.get("other")
    db.execute(
        sql(
            "UPDATE articles SET topic_primary_id = :t, topic_confidence = :c, "
            "topic_source = :s WHERE id = CAST(:id AS uuid)"
        ),
        {"t": primary_id, "c": round(confidence, 3), "s": source, "id": article["id"]},
    )
    db.execute(
        sql("DELETE FROM article_topics WHERE article_id = CAST(:id AS uuid)"),
        {"id": article["id"]},
    )
    seen = set()
    for rank, slug in enumerate([primary_slug, *secondary], start=1):
        topic_id = topic_ids.get(slug)
        if not topic_id or topic_id in seen:
            continue
        seen.add(topic_id)
        db.execute(
            sql(
                "INSERT INTO article_topics (article_id, topic_id, rank, confidence) "
                "VALUES (CAST(:a AS uuid), :t, :r, :c) ON CONFLICT DO NOTHING"
            ),
            {"a": article["id"], "t": topic_id, "r": rank, "c": round(confidence, 3)},
        )


_PENDING = """
SELECT id::text, title, abstract, keywords, doc_type
FROM articles
WHERE processing_status <> 'archived' AND duplicate_of_id IS NULL
  AND topic_source IS DISTINCT FROM 'manual'
  AND (topic_primary_id IS NULL OR :reclassify)
ORDER BY published_at DESC NULLS LAST
"""


def classify_pending(db: Session, reclassify: bool = False, limit: int | None = None) -> dict[str, int]:
    from concurrent.futures import ThreadPoolExecutor

    topics = taxonomy(db)
    topic_ids = {t["slug"]: t["id"] for t in topics}
    rows = [dict(r) for r in db.execute(sql(_PENDING), {"reclassify": reclassify}).mappings().all()]
    if limit:
        rows = rows[:limit]
    stats = {"llm": 0, "rule": 0, "batches_failed": 0}
    if not rows:
        _log("topics: nothing to classify")
        return stats

    batches = [rows[i : i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    _log(f"topics: {len(rows)} article(s) in {len(batches)} batch(es)")

    def _rule_only(batch: list[dict]) -> None:
        for article in batch:
            prior = rule_prior(article["title"], article.get("abstract"), article.get("keywords") or [], topics)
            slug = prior[0][0] if prior else "other"
            _apply(db, article, slug, [], 0.2 if prior else 0.0, "rule", topic_ids)
            stats["rule"] += 1

    if not ANTHROPIC_API_KEY:
        _log("topics: no ANTHROPIC_API_KEY; falling back to the rule prior for everything")
        for batch in batches:
            _rule_only(batch)
        db.commit()
        return stats

    # The classification calls run concurrently; the DB writes stay on this thread.
    # Sharing one sync Session across a pool is exactly the kind of subtle corruption
    # the sync-SQLAlchemy choice is supposed to avoid.
    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
        for batch, result in zip(batches, pool.map(lambda b: _classify_batch(b, topics), batches)):
            if result is None:
                stats["batches_failed"] += 1
                _rule_only(batch)
                continue
            by_item = {c.get("item"): c for c in result}
            for i, article in enumerate(batch):
                verdict = by_item.get(i)
                if not verdict:
                    _rule_only([article])
                    continue
                _apply(
                    db, article, verdict["primary"], verdict.get("secondary") or [],
                    float(verdict.get("confidence") or 0.5), "llm", topic_ids,
                )
                stats["llm"] += 1
            db.commit()
            done = stats["llm"] + stats["rule"]
            if done % 200 < BATCH_SIZE:
                _log(f"topics: {done}/{len(rows)}")

    db.commit()
    _log(f"topics: {stats}")
    return stats
