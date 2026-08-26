"""
The daily crawl as a standalone process: once a day at 01:00 local time, forever.

Railway deployments do NOT use this -- the API's `serve` role carries the same
schedule in-process (app/scheduler.py), so the always-on service that is already
running is the one that wakes up. This loop is for docker-compose and anything that
wants the cadence in a process of its own: zero extra dependencies, wakes once a day.
Set CRAWL_SCHEDULE_ENABLED=false on the API if you run this alongside it.

Each cycle is incremental (scoped to the last successful run) and each crawl is a
subprocess, so a crashed cycle never takes the loop down with it.
"""

import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.config import CRAWL_HOUR_LOCAL, CRAWL_TIMEZONE
from app.logging_utils import _log, configure_logging


def _next_run(now: datetime) -> datetime:
    target = now.replace(hour=CRAWL_HOUR_LOCAL, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def main() -> int:
    configure_logging()
    zone = ZoneInfo(CRAWL_TIMEZONE)
    _log(f"worker: daily crawl at {CRAWL_HOUR_LOCAL:02d}:00 {CRAWL_TIMEZONE}")

    while True:
        now = datetime.now(zone)
        target = _next_run(now)
        wait = (target - now).total_seconds()
        _log(f"worker: next run {target.isoformat()} (sleeping {wait / 3600:.1f}h)")
        # Sleep in short slices so SIGTERM lands promptly during shutdown.
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            time.sleep(min(60.0, max(1.0, deadline - time.monotonic())))

        try:
            from ingest.runner import run
            from scripts.run_ingest import _last_successful_run_date

            run(since=_last_successful_run_date(), triggered_by="worker")
        except Exception as exc:  # noqa: BLE001 -- one failed cycle must not kill the
                                  # loop; the run recorded its own failure and the
                                  # next 01:00 gets a fresh chance.
            _log(f"worker: cycle failed ({exc}); will retry at the next scheduled run")


if __name__ == "__main__":
    sys.exit(main())
