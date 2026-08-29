"""
Early-warning detection over the analysed media stream.

Deliberately DETERMINISTIC -- SQL aggregates and thresholds, no model. An alert is
an interruption of an analyst's attention; the analyst must be able to ask "why
did this fire" and get an arithmetic answer, not a vibe. The LLM's judgement is
already priced into the inputs (relevance, narrative, actors); the rules here are
just comparisons over sliding windows.

Idempotence: detection runs nightly over overlapping windows, so every rule builds
a dedup_key that names the CONDITION (the week of a spike, the slug of a new
narrative) rather than the run date, and inserts are ON CONFLICT DO NOTHING. A
condition that persists therefore fires once, and an analyst's dismiss is never
overwritten by the next night's run.

Thresholds are module constants, not env knobs: they were tuned against the live
window and there is one deployment; promote them to app/config.py the day a second
issue needs different sensitivity.
"""

import json
from typing import Any

from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.logging_utils import _log

# volume_spike: mean daily relevant coverage in the last RECENT_DAYS vs the
# BASELINE_DAYS before them.
RECENT_DAYS = 3
BASELINE_DAYS = 14
SPIKE_RATIO = 2.0
SPIKE_HIGH_RATIO = 3.0
SPIKE_MIN_RECENT = 6          # articles in the recent window, else tiny bases alert

# new_narrative / new_actor
NARRATIVE_FRESH_DAYS = 7
NARRATIVE_MIN_ARTICLES = 3
ACTOR_FRESH_DAYS = 7
ACTOR_PRIOR_DAYS = 30
ACTOR_MIN_MENTIONS = 4

# regional_spread
REGIONAL_FRESH_DAYS = 7
REGIONAL_PRIOR_DAYS = 30
REGIONAL_MIN_ARTICLES = 5
REGIONAL_MIN_SHARE = 0.20
REGIONAL_GROWTH = 1.8


def _insert(db: Session, issue_id: int, kind: str, severity: str, title: str,
            detail: str, confidence: float, evidence: list[str], dedup_key: str) -> int:
    """1 when a new alert row landed, 0 when the dedup key already existed."""
    row = db.execute(
        sql(
            "INSERT INTO media_alerts "
            "  (issue_id, kind, severity, title, detail, confidence, evidence, dedup_key) "
            "VALUES (:i, :k, :sev, :t, :d, :c, CAST(:e AS jsonb), :key) "
            "ON CONFLICT (issue_id, dedup_key) DO NOTHING RETURNING id"
        ),
        {"i": issue_id, "k": kind, "sev": severity, "t": title[:200], "d": detail,
         "c": round(min(confidence, 0.99), 2), "e": json.dumps(evidence), "key": dedup_key},
    ).scalar()
    return 1 if row else 0


def _evidence(db: Session, issue_id: int, days: int, limit: int = 6,
              narrative_id: int | None = None, actor: str | None = None,
              regional_only: bool = False) -> list[str]:
    """Top supporting article ids for an alert, best relevance first."""
    conditions = ["an.issue_id = :i", "an.is_relevant",
                  "a.published_at > now() - make_interval(days => :d)"]
    params: dict[str, Any] = {"i": issue_id, "d": days, "n": limit}
    if narrative_id is not None:
        conditions.append("an.narrative_id = :nar")
        params["nar"] = narrative_id
    if actor is not None:
        conditions.append(":actor = ANY(an.actors)")
        params["actor"] = actor
    if regional_only:
        conditions.append("o.media_type = 'regional'")
    rows = db.execute(
        sql(
            "SELECT a.id::text FROM media_article_analysis an "
            "JOIN media_articles a ON a.id = an.article_id "
            "JOIN media_outlets o ON o.id = a.outlet_id "
            f"WHERE {' AND '.join(conditions)} "
            "ORDER BY an.relevance DESC, a.published_at DESC LIMIT :n"
        ),
        params,
    ).scalars().all()
    return list(rows)


def _detect_volume_spike(db: Session, issue: dict) -> int:
    row = db.execute(
        sql(
            "SELECT "
            "  count(*) FILTER (WHERE a.published_at > now() - make_interval(days => :r)) "
            "    AS recent, "
            "  count(*) FILTER (WHERE a.published_at <= now() - make_interval(days => :r) "
            "    AND a.published_at > now() - make_interval(days => :r + :b)) AS baseline "
            "FROM media_article_analysis an "
            "JOIN media_articles a ON a.id = an.article_id "
            "WHERE an.issue_id = :i AND an.is_relevant"
        ),
        {"i": issue["id"], "r": RECENT_DAYS, "b": BASELINE_DAYS},
    ).mappings().one()
    recent, baseline = int(row["recent"]), int(row["baseline"])
    recent_rate = recent / RECENT_DAYS
    # max(_, 0.5): a silent baseline must read as "low base", not divide-by-zero.
    baseline_rate = max(baseline / BASELINE_DAYS, 0.5)
    ratio = recent_rate / baseline_rate
    if recent < SPIKE_MIN_RECENT or ratio < SPIKE_RATIO:
        return 0

    week = db.execute(sql("SELECT to_char(now(), 'IYYY\"-W\"IW')")).scalar()
    severity = "high" if ratio >= SPIKE_HIGH_RATIO else "medium"
    return _insert(
        db, issue["id"], "volume_spike", severity,
        "Coverage volume rising sharply",
        f"Relevant coverage is running at {recent_rate:.1f} items/day over the last "
        f"{RECENT_DAYS} days, {ratio:.1f}x the prior {BASELINE_DAYS}-day baseline of "
        f"{baseline_rate:.1f}/day ({recent} recent items).",
        0.5 + min(ratio / 10, 0.45),
        _evidence(db, issue["id"], RECENT_DAYS),
        f"volume_spike:{week}",
    )


def _detect_new_narratives(db: Session, issue: dict) -> int:
    rows = db.execute(
        sql(
            "SELECT n.id, n.slug, n.label, count(an.article_id) AS articles "
            "FROM media_narratives n "
            "JOIN media_article_analysis an ON an.narrative_id = n.id AND an.is_relevant "
            "WHERE n.issue_id = :i AND n.created_by = 'llm' "
            "  AND n.first_seen_at > now() - make_interval(days => :d) "
            "GROUP BY n.id HAVING count(an.article_id) >= :min"
        ),
        {"i": issue["id"], "d": NARRATIVE_FRESH_DAYS, "min": NARRATIVE_MIN_ARTICLES},
    ).mappings().all()
    created = 0
    for row in rows:
        created += _insert(
            db, issue["id"], "new_narrative", "medium",
            f"New narrative: {row['label']}",
            f"The analysis stage identified a framing not in the curated set -- "
            f"\"{row['label']}\" -- now supported by {row['articles']} relevant "
            f"article(s) within {NARRATIVE_FRESH_DAYS} days of first detection.",
            0.55 + min(int(row["articles"]) / 20, 0.35),
            _evidence(db, issue["id"], NARRATIVE_FRESH_DAYS * 2, narrative_id=row["id"]),
            f"new_narrative:{row['slug']}",
        )
    return created


def _detect_new_actors(db: Session, issue: dict) -> int:
    rows = db.execute(
        sql(
            "WITH mentions AS ("
            "  SELECT unnest(an.actors) AS actor, a.published_at "
            "  FROM media_article_analysis an "
            "  JOIN media_articles a ON a.id = an.article_id "
            "  WHERE an.issue_id = :i AND an.is_relevant "
            ") "
            "SELECT actor, "
            "  count(*) FILTER (WHERE published_at > now() - make_interval(days => :f)) AS fresh, "
            "  count(*) FILTER (WHERE published_at <= now() - make_interval(days => :f) "
            "    AND published_at > now() - make_interval(days => :f + :p)) AS prior "
            "FROM mentions GROUP BY actor "
            "HAVING count(*) FILTER (WHERE published_at > now() - make_interval(days => :f)) >= :min "
            "   AND count(*) FILTER (WHERE published_at <= now() - make_interval(days => :f) "
            "    AND published_at > now() - make_interval(days => :f + :p)) = 0"
        ),
        {"i": issue["id"], "f": ACTOR_FRESH_DAYS, "p": ACTOR_PRIOR_DAYS,
         "min": ACTOR_MIN_MENTIONS},
    ).mappings().all()
    created = 0
    for row in rows:
        actor = row["actor"]
        created += _insert(
            db, issue["id"], "new_actor", "medium",
            f"New actor gaining visibility: {actor}",
            f"{actor} appears in {row['fresh']} relevant article(s) in the last "
            f"{ACTOR_FRESH_DAYS} days after zero mentions in the prior "
            f"{ACTOR_PRIOR_DAYS} days.",
            0.5 + min(int(row["fresh"]) / 15, 0.4),
            _evidence(db, issue["id"], ACTOR_FRESH_DAYS, actor=actor),
            f"new_actor:{actor.lower().replace(' ', '-')[:60]}",
        )
    return created


def _detect_regional_spread(db: Session, issue: dict) -> int:
    row = db.execute(
        sql(
            "SELECT "
            "  count(*) FILTER (WHERE o.media_type = 'regional' "
            "    AND a.published_at > now() - make_interval(days => :f)) AS fresh_regional, "
            "  count(*) FILTER (WHERE a.published_at > now() - make_interval(days => :f)) AS fresh_total, "
            "  count(*) FILTER (WHERE o.media_type = 'regional' "
            "    AND a.published_at <= now() - make_interval(days => :f) "
            "    AND a.published_at > now() - make_interval(days => :f + :p)) AS prior_regional, "
            "  count(*) FILTER (WHERE a.published_at <= now() - make_interval(days => :f) "
            "    AND a.published_at > now() - make_interval(days => :f + :p)) AS prior_total "
            "FROM media_article_analysis an "
            "JOIN media_articles a ON a.id = an.article_id "
            "JOIN media_outlets o ON o.id = a.outlet_id "
            "WHERE an.issue_id = :i AND an.is_relevant"
        ),
        {"i": issue["id"], "f": REGIONAL_FRESH_DAYS, "p": REGIONAL_PRIOR_DAYS},
    ).mappings().one()
    fresh_r, fresh_t = int(row["fresh_regional"]), int(row["fresh_total"])
    prior_r, prior_t = int(row["prior_regional"]), int(row["prior_total"])
    if fresh_r < REGIONAL_MIN_ARTICLES or fresh_t == 0:
        return 0
    fresh_share = fresh_r / fresh_t
    prior_share = (prior_r / prior_t) if prior_t else 0.0
    # Both gates: absolute share (regional voice is material) and growth (it is
    # NEW) -- a corpus that is always 30% regional should never alert.
    if fresh_share < REGIONAL_MIN_SHARE or fresh_share < REGIONAL_GROWTH * max(prior_share, 0.02):
        return 0

    week = db.execute(sql("SELECT to_char(now(), 'IYYY\"-W\"IW')")).scalar()
    return _insert(
        db, issue["id"], "regional_spread", "medium",
        "Issue spreading into regional media",
        f"Regional outlets account for {fresh_share:.0%} of relevant coverage over the "
        f"last {REGIONAL_FRESH_DAYS} days ({fresh_r} of {fresh_t} items), up from "
        f"{prior_share:.0%} across the prior {REGIONAL_PRIOR_DAYS} days.",
        0.5 + min(fresh_share, 0.4),
        _evidence(db, issue["id"], REGIONAL_FRESH_DAYS, regional_only=True),
        f"regional_spread:{week}",
    )


def detect(db: Session) -> dict[str, int]:
    """Run every rule for every active issue. Safe to re-run nightly."""
    stats = {"volume_spike": 0, "new_narrative": 0, "new_actor": 0, "regional_spread": 0}
    issues = [
        dict(r)
        for r in db.execute(
            sql("SELECT id, slug FROM tracked_issues WHERE is_active")
        ).mappings().all()
    ]
    for issue in issues:
        stats["volume_spike"] += _detect_volume_spike(db, issue)
        stats["new_narrative"] += _detect_new_narratives(db, issue)
        stats["new_actor"] += _detect_new_actors(db, issue)
        stats["regional_spread"] += _detect_regional_spread(db, issue)
    db.commit()
    _log(f"media-alerts: {stats}")
    return stats
