"""
One crawl cycle: both spiders on one CrawlerProcess, then exit.

ALWAYS INVOKED AS A SUBPROCESS by ingest/runner.py, never in-process. Scrapy runs on
Twisted, whose reactor cannot be restarted within one process -- a second in-process
crawl raises ReactorNotRestartable. Being a subprocess is also what lets the admin
"ingest now" endpoint trigger a crawl from inside the long-lived API process.

Spider outcomes are captured via the spider_closed signal, because Scrapy swallows
spider errors inside the reactor: process.start() returns normally even when a
spider aborted, so `reason != "finished"` is the only reliable failure signal (the
sibling crawler's hard-won lesson, inherited unchanged).

The final line of stdout is a JSON stats object -- the runner parses exactly that
line, so nothing else may print after it.
"""

import argparse
import json
import sys

from app.logging_utils import _log, configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the PIKIS Scrapy spiders once")
    parser.add_argument("--source", default="all", choices=["all", "ije", "pyc-wp", "news"])
    parser.add_argument("--since", help="ISO date lower bound for an incremental crawl")
    parser.add_argument("--run-id", help="ingest_runs id, threaded into ingest_errors")
    args = parser.parse_args()

    configure_logging()

    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings
    from scrapy import signals

    from crawler.spiders.ije_spider import IjeSpider
    from crawler.spiders.news_spider import NewsSpider
    from crawler.spiders.pyc_spider import PycSpider

    settings = get_project_settings()
    process = CrawlerProcess(settings, install_root_handler=False)

    outcomes: dict[str, str] = {}
    stats: dict[str, dict] = {}
    # Scrapy's SignalManager (pydispatch) holds receivers by WEAK reference. A
    # closure defined in the loop below is garbage-collected the moment the loop
    # variable moves on -- observed: with two spiders queued, only the last-defined
    # handler survived and the first spider was reported as never_closed even though
    # it crawled 95 records. This list is the strong reference that keeps every
    # handler alive until process.start() returns.
    handlers: list = []

    wanted = {"ije": IjeSpider, "pyc-wp": PycSpider, "news": NewsSpider}
    if args.source != "all":
        wanted = {args.source: wanted[args.source]}

    for name, spider_cls in wanted.items():
        crawler = process.create_crawler(spider_cls)

        def on_closed(spider, reason, _name=name, _crawler=crawler):  # noqa: ANN001
            outcomes[_name] = reason
            s = _crawler.stats.get_stats() or {}
            stats[_name] = {
                "inserted": s.get("pikis/inserted", 0),
                "updated": s.get("pikis/updated", 0),
                "failed": s.get("pikis/failed", 0),
                "items": s.get("item_scraped_count", 0),
                "requests": s.get("downloader/request_count", 0),
                "reason": reason,
            }

        handlers.append(on_closed)
        crawler.signals.connect(on_closed, signal=signals.spider_closed)
        process.crawl(crawler, since=args.since, run_id=args.run_id)

    process.start()  # blocks until every queued spider finishes

    failed = [name for name, reason in outcomes.items() if reason != "finished"]
    missing = [name for name in wanted if name not in outcomes]
    for name in missing:
        stats[name] = {"reason": "never_closed"}

    _log(f"crawl: outcomes {outcomes or 'none'}")
    # The runner parses this exact line -- keep it last on stdout.
    print(json.dumps({"spiders": stats, "failed": failed + missing}, default=str), flush=True)
    return 1 if (failed or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
