"""(Re)run topic classification over the corpus."""

import argparse
import sys

from app.logging_utils import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reclassify", action="store_true",
                        help="re-run even for articles that already have a topic")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    configure_logging()
    from db.engine import SessionLocal
    from ingest.topics import classify_pending

    db = SessionLocal()
    try:
        stats = classify_pending(db, reclassify=args.reclassify, limit=args.limit)
        return 1 if stats.get("batches_failed") else 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
