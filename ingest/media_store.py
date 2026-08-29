"""
Persistence for media-monitoring articles, and outlet attribution.

Mirrors ingest/store.py's bargain -- batch upsert, then row-by-row replay when a
batch fails, so one malformed feed item costs one item. The upsert key is
(outlet_id, external_id), same shape as articles(source_id, external_id).

Attribution: items from an outlet's own RSS carry their outlet_id from the spider.
Items found via Google News carry only a source name + homepage URL; `resolve_outlet`
maps the domain onto a curated outlet row when one exists, and otherwise creates a
`discovered = true` row so no article is ever dropped for want of attribution. The
media_type of a discovered outlet is guessed from the domain (the Tribun and ANTARA
provincial networks are recognisably regional); everything else defaults to
national, which an admin can correct in place -- the row, not the guess, is what
queries read.
"""

from typing import Any
from urllib.parse import urlparse

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.logging_utils import _log
from ingest.store import _record_error

BATCH_SIZE = 50

# Province names for the two big regional networks' subdomains. Deliberately not
# exhaustive: an unlisted bureau still resolves as regional, just without a
# province label.
_REGION_BY_SUBDOMAIN = {
    "aceh": "Aceh", "sumut": "Sumatera Utara", "medan": "Sumatera Utara",
    "sumbar": "Sumatera Barat", "padang": "Sumatera Barat", "riau": "Riau",
    "pekanbaru": "Riau", "jambi": "Jambi", "sumsel": "Sumatera Selatan",
    "palembang": "Sumatera Selatan", "babel": "Bangka Belitung",
    "lampung": "Lampung", "banten": "Banten", "jabar": "Jawa Barat",
    "bandung": "Jawa Barat", "jateng": "Jawa Tengah", "jogja": "DI Yogyakarta",
    "yogya": "DI Yogyakarta", "jatim": "Jawa Timur", "surabaya": "Jawa Timur",
    "bali": "Bali", "mataram": "Nusa Tenggara Barat", "kupang": "Nusa Tenggara Timur",
    "kalbar": "Kalimantan Barat", "pontianak": "Kalimantan Barat",
    "kalteng": "Kalimantan Tengah", "kalsel": "Kalimantan Selatan",
    "banjarmasin": "Kalimantan Selatan", "kaltim": "Kalimantan Timur",
    "kaltara": "Kalimantan Utara", "sulut": "Sulawesi Utara",
    "manado": "Sulawesi Utara", "sulteng": "Sulawesi Tengah", "palu": "Sulawesi Tengah",
    "sultra": "Sulawesi Tenggara", "kendari": "Sulawesi Tenggara",
    "makassar": "Sulawesi Selatan", "gorontalo": "Gorontalo",
    "ambon": "Maluku", "maluku": "Maluku", "ternate": "Maluku Utara",
    "papua": "Papua", "jayapura": "Papua", "papuabarat": "Papua Barat",
}

_REGIONAL_PARENTS = ("tribunnews.com", "antaranews.com")


def _domain(url: str) -> str:
    netloc = urlparse(url if "//" in url else f"https://{url}").netloc.lower()
    return netloc.removeprefix("www.")


def _guess_kind(domain: str) -> tuple[str, str | None]:
    """(media_type, region) for a domain we have no curated row for."""
    for parent in _REGIONAL_PARENTS:
        if domain.endswith("." + parent):
            sub = domain[: -(len(parent) + 1)].split(".")[-1]
            return "regional", _REGION_BY_SUBDOMAIN.get(sub)
    # Beyond the two big networks, regional outlets advertise their region in the
    # domain itself (balipost, eksposkaltim, tribunmaluku). Substring matching
    # over the province/city tokens is fuzzy but errs the right way: a miss just
    # leaves the outlet national, which an admin can correct on the row.
    host = domain.split(".")[0]
    for token, region in _REGION_BY_SUBDOMAIN.items():
        if len(token) >= 4 and token in host:
            return "regional", region
    return "national", None


def resolve_outlet(
    db: Session, name: str, homepage: str, issue_id: int | None = None,
    cache: dict[str, int] | None = None,
) -> int:
    """Outlet id for a Google News source, creating a discovered row if needed."""
    domain = _domain(homepage)
    if cache is not None and domain in cache:
        return cache[domain]

    row = db.execute(
        text(
            "SELECT id FROM media_outlets "
            "WHERE lower(regexp_replace(coalesce(base_url, ''), '^https?://(www\\.)?', '')) "
            "      LIKE :d || '%' LIMIT 1"
        ),
        {"d": domain},
    ).scalar()

    if row is None:
        media_type, region = _guess_kind(domain)
        slug = domain.replace(".", "-")
        row = db.execute(
            text(
                "INSERT INTO media_outlets (slug, name, media_type, region, base_url, discovered) "
                "VALUES (:slug, :name, :mt, :region, :url, true) "
                "ON CONFLICT (slug) DO UPDATE SET name = media_outlets.name "
                "RETURNING id"
            ),
            {"slug": slug, "name": (name or domain)[:120], "mt": media_type,
             "region": region, "url": f"https://{domain}"},
        ).scalar()
        db.commit()

    # A discovered outlet surfaced through THIS issue's keywords, so it belongs in
    # the issue's monitored set -- without this row its articles would be invisible
    # to every dashboard query.
    if issue_id:
        db.execute(
            text(
                "INSERT INTO tracked_issue_outlets (issue_id, outlet_id) "
                "VALUES (:i, :o) ON CONFLICT DO NOTHING"
            ),
            {"i": issue_id, "o": row},
        )
        db.commit()

    if cache is not None:
        cache[domain] = int(row)
    return int(row)


_UPSERT = text(
    """
    INSERT INTO media_articles
        (outlet_id, external_id, url, title, summary, published_at, language,
         found_via, raw)
    VALUES
        (:outlet_id, :external_id, :url, :title, :summary,
         CAST(:published_at AS timestamptz), :language, :found_via, CAST(:raw AS jsonb))
    ON CONFLICT (outlet_id, external_id) DO UPDATE SET
        url = EXCLUDED.url,
        title = EXCLUDED.title,
        summary = coalesce(EXCLUDED.summary, media_articles.summary),
        published_at = coalesce(EXCLUDED.published_at, media_articles.published_at)
    RETURNING (xmax = 0) AS inserted
    """
)


def _params(item: dict[str, Any]) -> dict[str, Any]:
    import json

    return {
        "outlet_id": item["outlet_id"],
        "external_id": str(item["external_id"])[:500],
        "url": item.get("url") or "",
        "title": (item.get("title") or "")[:500],
        "summary": (item.get("summary") or None),
        "published_at": item.get("published_at"),
        "language": item.get("language") or "id",
        "found_via": item.get("found_via") or "feed",
        "raw": json.dumps(item.get("raw") or {}, default=str),
    }


def _existing_titles(db: Session, items: list[dict[str, Any]]) -> set[tuple[int, str]]:
    """(outlet_id, lower(title)) pairs already stored, for redirect-dedupe."""
    pairs = {(it["outlet_id"], (it.get("title") or "").lower()) for it in items}
    titles = [t for _, t in pairs if t]
    if not titles:
        return set()
    rows = db.execute(
        text(
            "SELECT outlet_id, lower(title) FROM media_articles "
            "WHERE lower(title) = ANY(:titles)"
        ),
        {"titles": titles},
    ).fetchall()
    return {(int(o), t) for o, t in rows}


def upsert_media_articles(
    db: Session, items: list[dict[str, Any]], run_id: str | None = None
) -> dict[str, int]:
    stats = {"inserted": 0, "updated": 0, "skipped": 0, "failed": 0}
    if not items:
        return stats

    # A Google News hit for a story we already hold from the outlet's own feed has
    # a different external_id (Google's guid) and a redirector URL -- the feed copy
    # is strictly better, so the Google copy is skipped on (outlet, title).
    seen = _existing_titles(db, [it for it in items if it.get("found_via") == "google-news"])

    def flush(batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        try:
            for params in batch:
                inserted = db.execute(_UPSERT, params).scalar()
                stats["inserted" if inserted else "updated"] += 1
            db.commit()
        except Exception:  # noqa: BLE001 -- replay row-by-row to isolate the bad item
            db.rollback()
            for params in batch:
                try:
                    inserted = db.execute(_UPSERT, params).scalar()
                    db.commit()
                    stats["inserted" if inserted else "updated"] += 1
                except Exception as exc:  # noqa: BLE001
                    db.rollback()
                    stats["failed"] += 1
                    _record_error(db, run_id, "media-store", None,
                                  params.get("external_id"), str(exc))

    batch: list[dict[str, Any]] = []
    for item in items:
        if item.get("found_via") == "google-news":
            key = (item["outlet_id"], (item.get("title") or "").lower())
            if key in seen:
                stats["skipped"] += 1
                continue
            seen.add(key)
        batch.append(_params(item))
        if len(batch) >= BATCH_SIZE:
            flush(batch)
            batch = []
    flush(batch)

    if stats["failed"]:
        _log(f"media-store: {stats['failed']} item(s) failed")
    return stats
