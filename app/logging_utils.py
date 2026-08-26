"""
Logging that survives a container.

A hand-rolled `_log()` writing "[iso8601] message" with flush=True, matching the
tender-intelligence crawler. The flush matters: Railway captures stdout, and a
buffered writer means a crashed process loses the lines that would explain the crash.

Library loggers (httpx, sentence_transformers, urllib3) are turned down here because
at INFO they emit one line per HTTP request, which buries the ~1,400 lines of ingest
progress that are actually worth reading.
"""

import logging
import sys
from datetime import datetime, timezone

from app.config import LOG_LEVEL


def _log(message: str) -> None:
    """Timestamped stdout line. Used by scripts and stage boundaries."""
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "sentence_transformers",
        "transformers",
        "filelock",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
