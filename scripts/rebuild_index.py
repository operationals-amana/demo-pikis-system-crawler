"""Rebuild the BM25 artifact only. The API hot-swaps it within INDEX_POLL_SECONDS."""

import sys

from app.logging_utils import configure_logging


def main() -> int:
    configure_logging()
    from db.engine import SessionLocal
    from ingest.index_builder import build_index

    db = SessionLocal()
    try:
        build_index(db)
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
