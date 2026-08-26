"""
Single-slot ingest starts, shared by the admin button and the daily schedule.

`ingest.runner.run` already guards concurrency with a Postgres advisory lock, so a
second start is *safe* no matter who asks. What lives here is the answer above that
lock: a caller deserves to be told "one is already running, here is how long it has
been at it" rather than have a thread spawn, take the lock's refusal and die into
the log. Both callers ask the same question, so they ask it in one place.

Two checks, because they fail differently:

  * the `ingest_runs` row, which outlives this process and so also catches a run
    started by a previous deploy or by the batch `ingest` role against the same DB;
  * the thread handle, which covers the window between "thread started" and "the
    runner wrote its row".

The row check is bounded by INGEST_STALE_MINUTES. A process killed mid-run leaves
`status = 'running'` behind forever, and an unbounded check would refuse every start
for the rest of time after one crash; the runner sweeps those corpses itself once it
holds the advisory lock.
"""

from __future__ import annotations

import threading
from typing import Any

from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.logging_utils import _log

# Guards the read-then-launch below. Without it the admin button and the scheduler
# firing in the same second could both see an idle slot and both spawn -- the
# advisory lock would still save the corpus, but one thread would die on a
# RuntimeError that reads like a fault when it is really a race.
_slot = threading.Lock()
_thread: threading.Thread | None = None


def running_run(db: Session) -> dict[str, Any] | None:
    """The live ingest row, or None. Live means 'running' and younger than the
    stale window -- see the module docstring for why the window is there."""
    from app.config import INGEST_STALE_MINUTES

    row = db.execute(
        sql(
            "SELECT id::text, started_at FROM ingest_runs WHERE status = 'running' "
            "AND started_at > now() - make_interval(mins => :stale) "
            "ORDER BY started_at DESC LIMIT 1"
        ),
        {"stale": INGEST_STALE_MINUTES},
    ).mappings().first()
    return dict(row) if row else None


def is_running(db: Session) -> bool:
    with _slot:
        if _thread is not None and _thread.is_alive():
            return True
    return running_run(db) is not None


def start(
    triggered_by: str,
    *,
    incremental: bool = True,
    db: Session | None = None,
) -> tuple[bool, dict[str, Any] | None]:
    """
    Launch a cycle in a background thread unless one is already going.

    Returns `(started, blocker)`. `blocker` describes the run that refused this
    start when there is a row for it, and is None both when the start succeeded and
    when the blocker is a thread too young to have written its row yet -- the
    caller phrases the 409 either way.

    Pass `db` when the caller already holds a session (the request handlers do);
    the scheduler thread has none and gets a short-lived one of its own.
    """
    global _thread

    from db.engine import SessionLocal

    session = db or SessionLocal()
    try:
        with _slot:
            if _thread is not None and _thread.is_alive():
                return False, running_run(session)
            blocker = running_run(session)
            if blocker:
                return False, blocker
            # daemon=True: on SIGTERM the platform gives us seconds, not the twenty
            # minutes an embed pass needs, so a deploy mid-ingest kills the thread
            # and leaves a 'running' row for the stale sweeper. That is the existing
            # bargain for the admin button; the schedule takes the same one.
            _thread = threading.Thread(
                target=_work,
                args=(triggered_by, incremental),
                daemon=True,
                name="ingest",
            )
            _thread.start()
            return True, None
    finally:
        if db is None:
            session.close()


def _work(triggered_by: str, incremental: bool) -> None:
    try:
        from ingest.runner import run

        since = None
        if incremental:
            from scripts.run_ingest import _last_successful_run_date

            since = _last_successful_run_date()
        run(since=since, triggered_by=triggered_by)
    except Exception as exc:  # noqa: BLE001 -- the run records its own failure in
                              # ingest_runs; this thread must simply not die loudly.
        _log(f"ingest[{triggered_by}]: {exc}")
