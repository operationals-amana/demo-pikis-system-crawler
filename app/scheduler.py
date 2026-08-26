"""
The daily crawl, scheduled inside the API process.

There are three ways to get a cycle to run once a day, and this is the one the
Railway deployment uses:

  * `docker-entrypoint.sh ingest` -- one cycle, then exit, for a platform cron;
  * `docker-entrypoint.sh worker` -- the same cycle on a loop, no HTTP, for
    docker-compose and anything without a scheduler;
  * this module -- the schedule living in the `serve` process, the sibling
    tender-intelligence crawler's arrangement.

The API is already a long-running process on an always-on service, so the schedule
rides along for free: no second Railway service, no second image pull, no second
resident copy of torch and the e5 weights just to wake up once a day. The batch role
above is still the better shape for a platform that bills per running minute; both
remain supported, and CRAWL_SCHEDULE_ENABLED=false turns this one off if a cron is
ever attached to a separate service (otherwise the corpus gets harvested twice).

It is a bare thread rather than APScheduler because the entire requirement is "fire
once a day at a fixed local hour": one wait, one call, no job store, and no new
dependency in an image that already carries torch.

Concurrency is not this module's problem. `app.ingest_control.start` is single-slot
and `ingest.runner.run` holds a Postgres advisory lock underneath it, so a scheduled
cycle landing on top of someone pressing "Ingest now" -- or on top of a second API
replica's scheduler -- is refused, not duplicated.

The next wake-up is recomputed from the wall clock in CRAWL_TIMEZONE on every
iteration rather than by adding 24h to the last one, so neither a slow ingest nor a
zone with DST can make the schedule drift.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.config import CRAWL_HOUR_LOCAL, CRAWL_SCHEDULE_ENABLED, CRAWL_TIMEZONE
from app.logging_utils import _log

_stop = threading.Event()
_thread: threading.Thread | None = None
_next_run: datetime | None = None


def next_run_at() -> datetime | None:
    """When the schedule fires next, or None if it is not running.

    Read from the loop rather than derived from the last ingest: a manual run does
    not move the schedule, so the two disagree, and the loop is the one telling the
    truth. Surfaced on /api/admin/stats so the dashboard can say when the next crawl
    is due instead of only when the last one finished.
    """
    return _next_run


def _next_after(now: datetime) -> datetime:
    target = now.replace(hour=CRAWL_HOUR_LOCAL, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def _arm() -> datetime:
    """Publish the next wake-up, and return it.

    Called by `start` before the thread exists as well as by the loop, so
    `next_run_at` is never momentarily None while the schedule is in fact armed --
    a null there means "not scheduled here", and a deploy racing the first
    iteration must not be able to say that.
    """
    global _next_run

    _next_run = _next_after(datetime.now(ZoneInfo(CRAWL_TIMEZONE)))
    return _next_run


def _loop() -> None:
    global _next_run

    while not _stop.is_set():
        now = datetime.now(ZoneInfo(CRAWL_TIMEZONE))
        target = _arm()
        wait = (target - now).total_seconds()
        _log(f"schedule: next ingest {target.isoformat()} (in {wait / 3600:.1f}h)")

        # Event.wait, not sleep: shutdown then lands in milliseconds instead of
        # after however many hours are left on the clock.
        if _stop.wait(wait):
            break

        from app import ingest_control

        started, blocker = ingest_control.start("cron", incremental=True)
        if started:
            _log("schedule: ingest started")
        else:
            detail = f" (run {blocker['id']}, started {blocker['started_at']})" if blocker else ""
            # Not an error: an ingest that overran a day, or an admin run in flight.
            # The corpus is being harvested either way and tomorrow gets a fresh go.
            _log(f"schedule: skipped, an ingest is already in progress{detail}")

    _next_run = None


def start() -> bool:
    """Start the scheduler thread. Idempotent, and a no-op when disabled."""
    global _thread

    if not CRAWL_SCHEDULE_ENABLED:
        _log("schedule: disabled (CRAWL_SCHEDULE_ENABLED=false); no daily ingest here")
        return False
    if _thread is not None and _thread.is_alive():
        return True

    _stop.clear()
    _arm()
    # Logged before the thread exists, or the loop's own first line beats it out.
    _log(f"schedule: daily ingest at {CRAWL_HOUR_LOCAL:02d}:00 {CRAWL_TIMEZONE}")
    _thread = threading.Thread(target=_loop, daemon=True, name="ingest-schedule")
    _thread.start()
    return True


def stop() -> None:
    _stop.set()
