"""Bootstrap or reset an administrator account."""

import argparse
import sys

from sqlalchemy import text as sql

from app.logging_utils import _log, configure_logging


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    configure_logging()
    from auth.passwords import hash_password
    from db.engine import SessionLocal

    db = SessionLocal()
    try:
        digest = hash_password(args.password)
        existing = db.execute(
            sql("SELECT id::text FROM users WHERE lower(email) = lower(:e)"), {"e": args.email}
        ).scalar()
        if existing:
            db.execute(
                sql(
                    "UPDATE users SET password_hash = :p, role = 'admin', is_active = true, "
                    "full_name = COALESCE(:n, full_name) WHERE id = CAST(:i AS uuid)"
                ),
                {"p": digest, "n": args.name, "i": existing},
            )
            _log(f"create_admin: updated {args.email} (now admin, password reset)")
        else:
            db.execute(
                sql(
                    "INSERT INTO users (email, password_hash, full_name, role) "
                    "VALUES (:e, :p, :n, 'admin')"
                ),
                {"e": args.email, "p": digest, "n": args.name},
            )
            _log(f"create_admin: created admin {args.email}")
        db.commit()
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
