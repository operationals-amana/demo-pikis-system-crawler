"""CLI entry point for a harvest/index cycle."""

import argparse
import sys

from app.logging_utils import configure_logging


def _last_successful_run_date() -> str | None:
    """
    The `since` value for a cron-driven incremental run.

    Two deliberate safety margins:
    * the date of the last OK/PARTIAL run minus TWO days, because both upstreams
      timestamp by their own clocks and OAI-PMH `from=` is inclusive by day -- a
      one-day overlap re-harvests a handful of records (the upsert is idempotent),
      whereas a gap silently loses them forever;
    * when no successful run exists (first deploy, or every prior run failed), return
      None so the cycle degrades to a FULL harvest rather than an empty one.
    """
    from sqlalchemy import text as sql

    from db.engine import SessionLocal

    db = SessionLocal()
    try:
        last = db.execute(
            sql(
                "SELECT max(started_at) FROM ingest_runs "
                "WHERE status IN ('ok', 'partial') AND kind IN ('full', 'incremental')"
            )
        ).scalar()
    finally:
        db.close()
    if not last:
        return None
    from datetime import timedelta

    return (last - timedelta(days=2)).strftime("%Y-%m-%d")


def main() -> int:
    parser = argparse.ArgumentParser(description="Harvest and index the PIKIS corpus")
    parser.add_argument("--source", default="all", choices=["all", "ije", "pyc-wp"])
    parser.add_argument("--since", help="ISO date for an incremental run")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="derive --since from the last successful run (what the daily cron uses)",
    )
    parser.add_argument("--limit", type=int, help="cap records per source (smoke tests)")
    parser.add_argument("--skip-pdf", action="store_true", help="skip PDF full-text extraction")
    parser.add_argument("--skip-embed", action="store_true", help="skip embedding + index build")
    parser.add_argument("--triggered-by", default="cli")
    args = parser.parse_args()

    configure_logging()
    from ingest.runner import run

    since = args.since
    if args.incremental and not since:
        since = _last_successful_run_date()

    sources = None if args.source == "all" else [args.source]
    stats = run(
        sources=sources,
        since=since,
        triggered_by=args.triggered_by,
        skip_pdf=args.skip_pdf,
        skip_embed=args.skip_embed,
        limit=args.limit,
    )
    return 1 if stats.get("failed_stages") else 0


if __name__ == "__main__":
    sys.exit(main())
