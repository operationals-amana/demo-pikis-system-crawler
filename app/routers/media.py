"""
Media Intelligence: tracked issues, the dashboard aggregate, evidence feed, alerts.

The dashboard is ONE aggregated endpoint rather than six small ones, the same
bargain as /api/admin/stats: the page renders from a single server-side fetch, and
the queries share the (issue, period) scoping so splitting them would just repeat
the joins. The evidence feed is separate because it pages and filters.

Every read is scoped through tracked_issue_outlets: an outlet an analyst removed
from the issue disappears from counts, chart, feed and alerts alike, because they
all pass through the same join.

Any signed-in user (all accounts are internal PYC) may read AND configure --
the brief says analysts, not admins, tune keywords and confirm alerts.
"""

from datetime import date, datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.deps import current_user, get_db
from app.errors import bad_request, not_found
from app.schemas import AlertResolve, MediaIssueUpdate

router = APIRouter(prefix="/api/media", tags=["media"])

PERIODS = (7, 30, 90)


def _iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return None


def _issue_row(db: Session, issue_id: int) -> dict[str, Any]:
    row = db.execute(
        sql(
            "SELECT i.id, i.slug, i.name, i.description, i.keywords, "
            "       i.default_period_days, i.is_active, "
            "       coalesce(array_agg(tio.outlet_id) FILTER (WHERE tio.outlet_id IS NOT NULL), '{}') AS outlet_ids "
            "FROM tracked_issues i "
            "LEFT JOIN tracked_issue_outlets tio ON tio.issue_id = i.id "
            "WHERE i.id = :i GROUP BY i.id"
        ),
        {"i": issue_id},
    ).mappings().first()
    if not row:
        raise not_found("Tracked issue")
    out = dict(row)
    out["keywords"] = list(out["keywords"] or [])
    out["outlet_ids"] = sorted(out["outlet_ids"] or [])
    return out


@router.get("/issues")
def list_issues(db: Session = Depends(get_db), user: dict = Depends(current_user)) -> list[dict[str, Any]]:
    ids = db.execute(
        sql("SELECT id FROM tracked_issues WHERE is_active ORDER BY id")
    ).scalars().all()
    return [_issue_row(db, int(i)) for i in ids]


@router.get("/outlets")
def list_outlets(db: Session = Depends(get_db), user: dict = Depends(current_user)) -> list[dict[str, Any]]:
    rows = db.execute(
        sql(
            "SELECT o.id, o.slug, o.name, o.media_type, o.region, o.base_url, "
            "       o.discovered, o.is_active, count(a.id) AS article_count "
            "FROM media_outlets o LEFT JOIN media_articles a ON a.outlet_id = o.id "
            "GROUP BY o.id "
            "ORDER BY o.discovered, o.media_type, lower(o.name)"
        )
    ).mappings().all()
    return [dict(r) for r in rows]


@router.patch("/issues/{issue_id}")
def update_issue(
    issue_id: int,
    body: MediaIssueUpdate,
    db: Session = Depends(get_db),
    user: dict = Depends(current_user),
) -> dict[str, Any]:
    _issue_row(db, issue_id)  # 404 before any write

    fields, params = [], {"i": issue_id}
    if body.name is not None:
        fields.append("name = :name"); params["name"] = body.name.strip()
    if body.description is not None:
        fields.append("description = :desc"); params["desc"] = body.description.strip()
    if body.keywords is not None:
        cleaned = [k.strip() for k in body.keywords if k and k.strip()]
        if not cleaned:
            raise bad_request("At least one keyword is required")
        fields.append("keywords = :kw"); params["kw"] = cleaned[:50]
    if body.default_period_days is not None:
        fields.append("default_period_days = :period")
        params["period"] = body.default_period_days
    if fields:
        db.execute(sql(f"UPDATE tracked_issues SET {', '.join(fields)} WHERE id = :i"), params)

    if body.outlet_ids is not None:
        valid = set(
            db.execute(
                sql("SELECT id FROM media_outlets WHERE id = ANY(:ids)"),
                {"ids": list(body.outlet_ids)},
            ).scalars().all()
        )
        if not valid:
            raise bad_request("At least one monitored outlet is required")
        db.execute(sql("DELETE FROM tracked_issue_outlets WHERE issue_id = :i"), {"i": issue_id})
        # One set-based statement, not a loop: the database is a network hop away,
        # and hundreds of single-row inserts at ~170ms RTT each blow straight
        # through the frontend proxy's timeout.
        db.execute(
            sql(
                "INSERT INTO tracked_issue_outlets (issue_id, outlet_id) "
                "SELECT :i, unnest(CAST(:ids AS int[])) ON CONFLICT DO NOTHING"
            ),
            {"i": issue_id, "ids": sorted(int(o) for o in valid)},
        )
    db.commit()
    return _issue_row(db, issue_id)


# Scoping fragment shared by every dashboard query: relevant analysis rows for
# the issue, restricted to its monitored outlets. :p / :p2 bound per query.
_SCOPE = """
FROM media_article_analysis an
JOIN media_articles a ON a.id = an.article_id
JOIN media_outlets o ON o.id = a.outlet_id
JOIN tracked_issue_outlets tio ON tio.outlet_id = o.id AND tio.issue_id = an.issue_id
WHERE an.issue_id = :i AND an.is_relevant
"""

_WINDOW = "AND a.published_at > now() - make_interval(days => :p)"
_PREV_WINDOW = (
    "AND a.published_at <= now() - make_interval(days => :p) "
    "AND a.published_at > now() - make_interval(days => :p2)"
)


def _delta_pct(current: int, previous: int) -> int | None:
    if previous <= 0:
        return None
    return round((current - previous) * 100 / previous)


def _alert_payload(db: Session, issue_id: int) -> list[dict[str, Any]]:
    rows = db.execute(
        sql(
            "SELECT al.id::text, al.kind, al.severity, al.title, al.detail, "
            "       al.confidence, al.evidence, al.status, al.triggered_at, "
            "       al.resolved_at, u.full_name AS resolved_by_name "
            "FROM media_alerts al LEFT JOIN users u ON u.id = al.resolved_by "
            "WHERE al.issue_id = :i "
            "ORDER BY (al.status = 'pending') DESC, al.triggered_at DESC LIMIT 30"
        ),
        {"i": issue_id},
    ).mappings().all()
    alerts = []
    for row in rows:
        alert = dict(row)
        alert["triggered_at"] = _iso(alert["triggered_at"])
        alert["resolved_at"] = _iso(alert["resolved_at"])
        ids = [str(x) for x in (alert.pop("evidence") or [])][:6]
        evidence = []
        if ids:
            evidence = [
                dict(r)
                for r in db.execute(
                    sql(
                        "SELECT a.id::text, a.title, a.url, a.published_at, "
                        "       o.name AS outlet_name, o.media_type "
                        "FROM media_articles a JOIN media_outlets o ON o.id = a.outlet_id "
                        "WHERE a.id = ANY(CAST(:ids AS uuid[])) "
                        "ORDER BY a.published_at DESC"
                    ),
                    {"ids": ids},
                ).mappings().all()
            ]
            for item in evidence:
                item["published_at"] = _iso(item["published_at"])
        alert["evidence"] = evidence
        alerts.append(alert)
    return alerts


@router.get("/issues/{issue_id}/dashboard")
def dashboard(
    issue_id: int,
    period: int = Query(default=0),
    db: Session = Depends(get_db),
    user: dict = Depends(current_user),
) -> dict[str, Any]:
    issue = _issue_row(db, issue_id)
    if period not in PERIODS:
        period = int(issue["default_period_days"])
    params = {"i": issue_id, "p": period, "p2": period * 2}

    totals = db.execute(
        sql(
            "SELECT count(*) FILTER (WHERE true " + _WINDOW + ") AS current, "
            "       count(*) FILTER (WHERE true " + _PREV_WINDOW + ") AS previous "
            + _SCOPE
        ),
        params,
    ).mappings().one()
    current_total, previous_total = int(totals["current"]), int(totals["previous"])

    split = {
        r["media_type"]: int(r["n"])
        for r in db.execute(
            sql("SELECT o.media_type, count(*) AS n " + _SCOPE + _WINDOW + " GROUP BY 1"),
            params,
        ).mappings().all()
    }

    # Daily series, gaps filled in Python: recharts needs a row per day, and
    # generate_series-with-three-left-joins is harder to read than a dict loop.
    by_day: dict[str, dict[str, int]] = {}
    for r in db.execute(
        sql(
            "SELECT a.published_at::date AS day, o.media_type, count(*) AS n "
            + _SCOPE + _WINDOW + " GROUP BY 1, 2"
        ),
        params,
    ).mappings().all():
        day = r["day"].isoformat()
        by_day.setdefault(day, {})[r["media_type"]] = int(r["n"])

    series = []
    today = date.today()
    for offset in range(period - 1, -1, -1):
        day = (today - timedelta(days=offset)).isoformat()
        counts = by_day.get(day, {})
        series.append(
            {
                "day": day,
                "national": counts.get("national", 0),
                "regional": counts.get("regional", 0),
                "print": counts.get("print", 0),
            }
        )
    peak = max(series, key=lambda s: s["national"] + s["regional"] + s["print"], default=None)

    narratives = []
    for r in db.execute(
        sql(
            "SELECT n.id, n.slug, n.label, n.description, n.created_by, n.first_seen_at, "
            "       count(*) FILTER (WHERE true " + _WINDOW + ") AS current, "
            "       count(*) FILTER (WHERE true " + _PREV_WINDOW + ") AS previous "
            "FROM media_narratives n "
            "JOIN media_article_analysis an ON an.narrative_id = n.id AND an.is_relevant "
            "JOIN media_articles a ON a.id = an.article_id "
            "JOIN tracked_issue_outlets tio ON tio.outlet_id = a.outlet_id "
            "  AND tio.issue_id = n.issue_id "
            "WHERE n.issue_id = :i GROUP BY n.id "
            "HAVING count(*) FILTER (WHERE true " + _WINDOW + ") > 0 "
            "ORDER BY 7 DESC"
        ),
        params,
    ).mappings().all():
        n = dict(r)
        count = int(n.pop("current"))
        previous = int(n.pop("previous"))
        first_seen = n.pop("first_seen_at")
        narratives.append(
            {
                **n,
                "count": count,
                "delta_pct": _delta_pct(count, previous),
                "share_pct": round(count * 100 / current_total) if current_total else 0,
                "is_new": n["created_by"] == "llm"
                and first_seen > datetime.now(timezone.utc) - timedelta(days=period),
            }
        )

    # Actor mentions: unnest inside the scoped subquery, window arithmetic on the
    # flattened rows (t.published_at) outside it.
    _T_WINDOW = "t.published_at > now() - make_interval(days => :p)"
    _T_PREV = (
        "t.published_at <= now() - make_interval(days => :p) "
        "AND t.published_at > now() - make_interval(days => :p2)"
    )
    # "New" is only a signal when there WAS a previous period to be absent from;
    # with an empty baseline (fresh deployment) every actor would be "new" and
    # the badge would mean nothing.
    has_baseline = previous_total > 0
    actors = [
        {
            "name": r["actor"],
            "mentions": int(r["current"]),
            "delta_pct": _delta_pct(int(r["current"]), int(r["previous"])),
            "is_new": has_baseline and int(r["previous"]) == 0,
        }
        for r in db.execute(
            sql(
                "SELECT t.actor, "
                f"       count(*) FILTER (WHERE {_T_WINDOW}) AS current, "
                f"       count(*) FILTER (WHERE {_T_PREV}) AS previous "
                "FROM (SELECT unnest(an.actors) AS actor, a.published_at " + _SCOPE + ") t "
                "GROUP BY t.actor "
                f"HAVING count(*) FILTER (WHERE {_T_WINDOW}) > 0 "
                "ORDER BY 2 DESC LIMIT 12"
            ),
            params,
        ).mappings().all()
    ]

    stats_row = db.execute(
        sql(
            "SELECT count(DISTINCT an.narrative_id) AS active_narratives, "
            "       count(DISTINCT actor) AS active_actors "
            "FROM (SELECT an.narrative_id, unnest(an.actors) AS actor, a.published_at "
            + _SCOPE + _WINDOW + ") AS an"
        ),
        params,
    ).mappings().one()

    last_updated = db.execute(
        sql(
            "SELECT max(a.fetched_at) FROM media_articles a "
            "JOIN tracked_issue_outlets tio ON tio.outlet_id = a.outlet_id "
            "WHERE tio.issue_id = :i"
        ),
        {"i": issue_id},
    ).scalar()

    alerts = _alert_payload(db, issue_id)

    return {
        "issue": issue,
        "period_days": period,
        "last_updated": _iso(last_updated),
        "stats": {
            "relevant_total": current_total,
            "previous_total": previous_total,
            "delta_pct": _delta_pct(current_total, previous_total),
            "media_split": {
                "national": split.get("national", 0),
                "regional": split.get("regional", 0),
                "print": split.get("print", 0),
            },
            "active_narratives": int(stats_row["active_narratives"] or 0),
            "new_narratives": sum(1 for n in narratives if n["is_new"]),
            "active_actors": int(stats_row["active_actors"] or 0),
            "new_actors": sum(1 for a in actors if a["is_new"]),
            "peak_day": peak["day"] if peak else None,
            "peak_count": (peak["national"] + peak["regional"] + peak["print"]) if peak else 0,
            "pending_alerts": sum(1 for a in alerts if a["status"] == "pending"),
        },
        "series": series,
        "narratives": narratives,
        "actors": actors,
        "alerts": alerts,
    }


@router.get("/issues/{issue_id}/articles")
def evidence_feed(
    issue_id: int,
    period: int = Query(default=30),
    media_type: str | None = Query(default=None, pattern="^(national|regional|print)$"),
    region: str | None = Query(default=None, max_length=80),
    narrative: str | None = Query(default=None, max_length=80),
    q: str | None = Query(default=None, max_length=200),
    date_from: str | None = Query(default=None, max_length=10),
    date_to: str | None = Query(default=None, max_length=10),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    user: dict = Depends(current_user),
) -> dict[str, Any]:
    _issue_row(db, issue_id)
    if period not in PERIODS:
        period = 30

    # Explicit dates REPLACE the rolling window -- an analyst reaching for "that
    # Tuesday in June" must not be silently clipped by the period toggle.
    conditions = [] if (date_from or date_to) else [_WINDOW]
    params: dict[str, Any] = {"i": issue_id, "p": period,
                              "limit": page_size, "offset": (page - 1) * page_size}
    if media_type:
        conditions.append("AND o.media_type = :mt"); params["mt"] = media_type
    if region:
        conditions.append("AND o.region = :region"); params["region"] = region
    if narrative:
        conditions.append(
            "AND an.narrative_id = (SELECT id FROM media_narratives "
            "WHERE issue_id = :i AND slug = :nar)"
        )
        params["nar"] = narrative
    if q:
        conditions.append("AND a.title ILIKE :q"); params["q"] = f"%{q}%"
    if date_from:
        conditions.append("AND a.published_at >= CAST(:df AS date)"); params["df"] = date_from
    if date_to:
        conditions.append("AND a.published_at < CAST(:dt AS date) + 1"); params["dt"] = date_to
    where = " ".join(conditions)

    total = db.execute(sql("SELECT count(*) " + _SCOPE + where), params).scalar()

    rows = db.execute(
        sql(
            "SELECT a.id::text, a.title, a.url, a.summary, a.published_at, a.found_via, "
            "       an.relevance, an.actors, o.name AS outlet_name, o.media_type, "
            "       o.region, n.slug AS narrative, n.label AS narrative_label "
            + _SCOPE.replace(
                "WHERE an.issue_id",
                "LEFT JOIN media_narratives n ON n.id = an.narrative_id WHERE an.issue_id",
            )
            + where
            + " ORDER BY a.published_at DESC NULLS LAST LIMIT :limit OFFSET :offset"
        ),
        params,
    ).mappings().all()
    items = []
    for row in rows:
        item = dict(row)
        item["published_at"] = _iso(item["published_at"])
        item["actors"] = list(item["actors"] or [])
        item["relevance"] = float(item["relevance"] or 0)
        items.append(item)

    regions = db.execute(
        sql(
            "SELECT DISTINCT o.region " + _SCOPE
            + " AND o.region IS NOT NULL ORDER BY 1"
        ),
        {"i": issue_id},
    ).scalars().all()

    return {
        "items": items,
        "total": int(total or 0),
        "page": page,
        "page_size": page_size,
        "regions": list(regions),
    }


@router.post("/alerts/{alert_id}/resolve")
def resolve_alert(
    alert_id: str,
    body: AlertResolve,
    db: Session = Depends(get_db),
    user: dict = Depends(current_user),
) -> dict[str, Any]:
    status = "confirmed" if body.action == "confirm" else "dismissed"
    row = db.execute(
        sql(
            "UPDATE media_alerts SET status = :s, resolved_by = CAST(:u AS uuid), "
            "resolved_at = now() WHERE id = CAST(:id AS uuid) "
            "RETURNING id::text, status, resolved_at"
        ),
        {"s": status, "u": str(user["id"]), "id": alert_id},
    ).mappings().first()
    if not row:
        db.rollback()
        raise not_found("Alert")
    db.commit()
    out = dict(row)
    out["resolved_at"] = _iso(out["resolved_at"])
    return out
