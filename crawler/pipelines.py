"""
Validation + Postgres persistence.

Dedup-by-upsert and persistence live in the SAME pipeline stage on purpose (the
sibling's reasoning, inherited): splitting them means two round trips and two
sessions per record, and leaves a race where the check and the write disagree.

The writer reuses ingest/store.upsert_articles verbatim -- batch-then-row-by-row on
failure, metadata_locked_fields honoured -- so switching the fetch layer to Scrapy
changed nothing about how records land in the database.
"""

from typing import Any

from scrapy import Spider
from scrapy.exceptions import DropItem

from app.logging_utils import _log
from db.engine import SessionLocal
from ingest.store import upsert_articles

BATCH_SIZE = 50


class ValidationPipeline:
    """Drop records that cannot become an article row. Runs first (priority 100)."""

    def process_item(self, item: dict[str, Any], spider: Spider) -> dict[str, Any]:
        if not item.get("external_id"):
            raise DropItem("record has no external_id")
        if not item.get("title"):
            raise DropItem(f"record {item.get('external_id')!r} has no title")
        return item


class PostgresPipeline:
    """Batched upserts, flushed every BATCH_SIZE items and at spider close."""

    def open_spider(self, spider: Spider) -> None:
        self.db = SessionLocal()
        self.batch: list[dict[str, Any]] = []
        self.stats = {"inserted": 0, "updated": 0, "failed": 0}

    def process_item(self, item: dict[str, Any], spider: Spider) -> dict[str, Any]:
        self.batch.append(item)
        if len(self.batch) >= BATCH_SIZE:
            self._flush(spider)
        return item

    def _flush(self, spider: Spider) -> None:
        if not self.batch:
            return
        result = upsert_articles(
            self.db,
            self.batch,
            source_id=spider.source_id,  # set in each spider's __init__
            run_id=getattr(spider, "run_id", None),
        )
        for key in self.stats:
            self.stats[key] += result.get(key, 0)
        self.batch = []

    def close_spider(self, spider: Spider) -> None:
        try:
            self._flush(spider)
            _log(f"{spider.name}: upserts {self.stats}")
            # Surface the write stats through Scrapy's stats collector so
            # scripts/run_crawl.py can report them without a second channel.
            for key, value in self.stats.items():
                spider.crawler.stats.set_value(f"pikis/{key}", value)
        finally:
            self.db.close()
