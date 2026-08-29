"""
Stage orchestration for a harvest run.

Every stage is wrapped at its boundary in a broad except that records an ingest_errors
row and continues. A run finishing 'partial' is a normal, reported outcome; a run that
dies is a bug.

Single-slot via a Postgres advisory lock rather than an in-process flag. The sibling
used a threading.Lock, which is correct for one container -- but two Railway replicas
cannot see each other's flags, and two concurrent embed runs over the same articles
would interleave chunk deletes and inserts.
"""

import json
from typing import Any, Callable

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.logging_utils import _log
from db.engine import SessionLocal
from ingest import dedupe
from ingest.store import _record_error

# hashtext('pikis:ingest') -- any stable int works; this one is self-documenting.
LOCK_KEY = 0x504B4953  # "PKIS"


def _acquire(db: Session) -> bool:
    return bool(db.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": LOCK_KEY}).scalar())


def _release(db: Session) -> None:
    db.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": LOCK_KEY})
    db.commit()


def _source_id(db: Session, slug: str) -> int:
    value = db.execute(text("SELECT id FROM sources WHERE slug = :s"), {"s": slug}).scalar()
    if value is None:
        raise RuntimeError(f"source {slug!r} missing -- run migrations (0006_seed.sql)")
    return int(value)


def _reclaim_stale_runs(db: Session) -> None:
    """
    Mark orphaned 'running' rows as failed.

    Called AFTER acquiring the advisory lock: ingest is single-slot, so while we hold
    the lock any other row still claiming to run belongs to a process that died
    mid-run (observed: a killed smoke test left a 'running' row that made the admin
    trigger 409 for two hours). The row keeps its stats; only the status is corrected.
    """
    swept = db.execute(
        text(
            "UPDATE ingest_runs SET status = 'failed', finished_at = now(), "
            "error = 'reclaimed: the owning process died mid-run' "
            "WHERE status = 'running'"
        )
    ).rowcount
    db.commit()
    if swept:
        _log(f"ingest: reclaimed {swept} stale run(s) left by dead processes")


def _start_run(db: Session, kind: str, triggered_by: str) -> str:
    run_id = db.execute(
        text(
            "INSERT INTO ingest_runs (kind, triggered_by) VALUES (:k, :t) RETURNING id::text"
        ),
        {"k": kind, "t": triggered_by},
    ).scalar()
    db.commit()
    return str(run_id)


def _finish_run(db: Session, run_id: str, status: str, stats: dict[str, Any], error: str | None = None) -> None:
    db.execute(
        text(
            "UPDATE ingest_runs SET status = :s, finished_at = now(), "
            "stats = CAST(:st AS jsonb), error = :e WHERE id = CAST(:id AS uuid)"
        ),
        {"s": status, "st": json.dumps(stats, default=str), "e": error, "id": run_id},
    )
    db.commit()


def _stage(db: Session, run_id: str, name: str, fn: Callable[[], Any], stats: dict[str, Any]) -> bool:
    """Run one stage. Returns False on failure but never raises -- degrade, record, continue."""
    _log(f"stage: {name} starting")
    try:
        result = fn()
        if result is not None:
            stats[name] = result
        _log(f"stage: {name} ok {result if result is not None else ''}")
        return True
    except Exception as exc:  # noqa: BLE001 -- a stage boundary is exactly where a
                              # broad except belongs: the next stage may still be able
                              # to do useful work, and the failure is recorded.
        db.rollback()
        _log(f"stage: {name} FAILED -- {exc}")
        _record_error(db, run_id, name, None, None, str(exc))
        stats.setdefault("failed_stages", []).append(name)
        return False


def _run_crawl_subprocess(
    run_id: str, sources: set[str], since: str | None
) -> dict[str, Any]:
    """
    Run the Scrapy spiders in a SUBPROCESS -- never in-process.

    Scrapy runs on Twisted, whose reactor cannot be restarted within one process; a
    second in-process crawl raises ReactorNotRestartable. The subprocess boundary is
    also what lets the admin "ingest now" endpoint (a thread inside the long-lived
    API process) trigger a crawl at all.

    scripts/run_crawl.py prints a JSON stats object as its LAST stdout line; that
    contract is parsed here. stdout/stderr are captured so a spider's log ends up in
    the ingest error record rather than interleaved with the API's own log.
    """
    import json as _json
    import subprocess
    import sys as _sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    command = [_sys.executable, "-m", "scripts.run_crawl", "--run-id", run_id]
    if sources != {"ije", "pyc-wp", "news"}:
        command += ["--source", next(iter(sources))]
    if since:
        command += ["--since", since]

    import os as _os

    result = subprocess.run(
        command,
        cwd=repo_root,
        env={**_os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        timeout=1800,
    )

    payload: dict[str, Any] = {}
    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = _json.loads(line)
            except ValueError:
                pass
            break

    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip()[-1500:]
        raise RuntimeError(
            f"crawl subprocess exited {result.returncode} "
            f"(failed spiders: {payload.get('failed') or 'unknown'}): {tail}"
        )
    return payload.get("spiders", payload)


def run(
    sources: list[str] | None = None,
    since: str | None = None,
    triggered_by: str = "cli",
    skip_pdf: bool = False,
    skip_embed: bool = False,
    limit: int | None = None,  # kept for CLI compatibility; the spiders crawl fully
) -> dict[str, Any]:
    wanted = set(sources or ["ije", "pyc-wp", "news"])
    stats: dict[str, Any] = {}
    db = SessionLocal()

    if not _acquire(db):
        db.close()
        raise RuntimeError("another ingest run holds the lock")

    _reclaim_stale_runs(db)
    run_id = _start_run(db, "full" if not since else "incremental", triggered_by)
    _log(f"ingest: run {run_id} started (sources={sorted(wanted)})")

    try:
        _stage(
            db,
            run_id,
            "crawl",
            lambda: _run_crawl_subprocess(run_id, wanted, since),
            stats,
        )

        # Media stages run right after the crawl, before the RAG-heavy ones: a
        # failure in pdf/chunk/embed must not cost the morning's media dashboard.
        from ingest.media_alerts import detect as detect_media_alerts
        from ingest.media_analysis import analyze_pending

        _stage(db, run_id, "media_analyze", lambda: analyze_pending(db), stats)
        _stage(db, run_id, "media_alerts", lambda: detect_media_alerts(db), stats)

        _stage(db, run_id, "dedupe", lambda: dedupe.relink(db), stats)

        from ingest.topics import classify_pending

        _stage(db, run_id, "classify_topics", lambda: classify_pending(db), stats)

        if not skip_pdf:
            from ingest.pdf_text import extract_pending

            _stage(db, run_id, "pdf_text", lambda: extract_pending(db, run_id), stats)

        from ingest.chunker import chunk_pending

        _stage(db, run_id, "chunk", lambda: chunk_pending(db, run_id), stats)

        if not skip_embed:
            from ingest.embedder import embed_pending
            from ingest.index_builder import build_index

            _stage(db, run_id, "embed", lambda: embed_pending(db, run_id), stats)
            _stage(db, run_id, "index", lambda: build_index(db), stats)

        failed = stats.get("failed_stages") or []
        status = "partial" if failed else "ok"
        _finish_run(db, run_id, status, stats)
        _log(f"ingest: run {run_id} {status} {stats}")
        return stats
    except Exception as exc:  # noqa: BLE001 -- record the fatal case too
        db.rollback()
        _finish_run(db, run_id, "failed", stats, str(exc))
        raise
    finally:
        _release(db)
        db.close()
