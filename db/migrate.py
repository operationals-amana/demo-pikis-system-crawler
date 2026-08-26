"""
Forward-only SQL migration runner.

Plain .sql files rather than Alembic, deliberately. The hard parts of this schema --
CREATE EXTENSION vector, a vector(384) column, a tsvector trigger, CHECK constraints,
partial unique indexes, seed data -- are things Alembic's autogenerate understands
none of, so every revision would be op.execute(\"\"\"...\"\"\") anyway. You would pay a
dependency and an env.py kept in lockstep with db/models.py, to receive a revision
graph. The sibling project has no migration tool at all; this is the smaller deviation.

Forward-only, no downgrades: the corpus is 100% re-derivable from two public APIs, so
a bad migration is fixed by a forward migration or a rebuild, not a rollback.

The checksum guard catches the classic "someone edited 0003 after it shipped" bug --
which Alembic does not catch either.
"""

import hashlib
import re
from pathlib import Path

from sqlalchemy import text

from app.logging_utils import _log
from db.engine import engine

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    text PRIMARY KEY,
    checksum   char(64) NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
);
"""


def _checksum(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _discover() -> list[tuple[str, Path]]:
    """.sql only -- `.sql.optional` files are documentation, never auto-applied."""
    found = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        version = re.split(r"[_.]", path.name, maxsplit=1)[0]
        found.append((version, path))
    return found


def run_migrations() -> int:
    applied_count = 0
    with engine.begin() as conn:
        conn.execute(text(_BOOTSTRAP))

    with engine.connect() as conn:
        rows = conn.execute(text("SELECT version, checksum FROM schema_migrations")).all()
    applied = {version: checksum for version, checksum in rows}

    for version, path in _discover():
        body = path.read_text(encoding="utf-8")
        digest = _checksum(body)

        if version in applied:
            if applied[version] != digest:
                raise RuntimeError(
                    f"Migration {path.name} changed after it was applied "
                    f"(recorded {applied[version][:12]}, file {digest[:12]}). "
                    "Applied migrations are immutable -- add a new migration instead."
                )
            continue

        _log(f"migrate: applying {path.name}")
        # One transaction per file: a failure leaves the database on the last good
        # migration rather than half-way through this one.
        with engine.begin() as conn:
            conn.execute(text(body))
            conn.execute(
                text(
                    "INSERT INTO schema_migrations (version, checksum) "
                    "VALUES (:v, :c)"
                ),
                {"v": version, "c": digest},
            )
        applied_count += 1

    if applied_count == 0:
        _log("migrate: already up to date")
    else:
        _log(f"migrate: applied {applied_count} migration(s)")
    return applied_count


if __name__ == "__main__":
    run_migrations()
