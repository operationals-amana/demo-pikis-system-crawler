"""
Create the database if it does not exist, then migrate it.

`CREATE DATABASE` cannot run inside a transaction, so it needs an AUTOCOMMIT
connection to the maintenance database rather than the usual engine. This is the one
place that connects to something other than DATABASE_URL.

Safe and idempotent: run it on every container start (docker-entrypoint.sh does).
"""

import sys
import time
from urllib.parse import urlparse

import sqlalchemy
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.config import DATABASE_URL
from app.logging_utils import _log, configure_logging


def ensure_database() -> None:
    parsed = urlparse(DATABASE_URL)
    dbname = (parsed.path or "/").lstrip("/")
    if not dbname:
        raise RuntimeError(f"DATABASE_URL has no database name: {DATABASE_URL}")

    # Connect to `postgres` to ask whether our database exists.
    admin_url = DATABASE_URL.replace(f"/{dbname}", "/postgres", 1)
    admin_engine = sqlalchemy.create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": dbname}
            ).scalar()
            if exists:
                _log(f"init_db: database {dbname!r} already exists")
                return
            _log(f"init_db: creating database {dbname!r}")
            # Identifier cannot be parameterised; quote it instead.
            conn.execute(text(f'CREATE DATABASE "{dbname}"'))
    finally:
        admin_engine.dispose()


def main() -> int:
    configure_logging()
    try:
        ensure_database()
    except Exception as exc:  # noqa: BLE001 -- a managed Postgres may forbid CREATE
                              # DATABASE; that is fine if the database already exists,
                              # so report and let migrate decide.
        _log(f"init_db: could not ensure database ({exc}); trying migrations anyway")

    from db.migrate import run_migrations

    # The Supabase pooler occasionally drops a brand-new connection ("SSL SYSCALL
    # error: EOF detected"). This script runs on every container start, so crashing
    # on the first such drop fails the whole deploy over a blip -- retry instead.
    attempts = 5
    for attempt in range(1, attempts + 1):
        try:
            run_migrations()
            break
        except OperationalError as exc:
            if attempt == attempts:
                raise
            _log(f"init_db: transient DB error (attempt {attempt}/{attempts}): {exc}; retrying")
            time.sleep(2 * attempt)

    # Prove the extension actually landed -- a migration that silently no-ops on a
    # platform without pgvector would otherwise only surface at first query.
    from db.engine import engine

    with engine.connect() as conn:
        version = conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
    if not version:
        _log("init_db: FATAL -- pgvector is not installed in this database")
        return 1
    _log(f"init_db: pgvector {version} present; schema ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
