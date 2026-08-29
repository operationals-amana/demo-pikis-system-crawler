"""
News monitoring: outlet RSS feeds + Google News keyword search, in one spider.

Two fetch paths because they answer different questions. An outlet's own feed says
"here is everything this desk published today" -- full summaries, canonical URLs,
but only the last day or two of items. A Google News search per tracked-issue
keyword says "here is who covered this topic recently" -- weeks of depth across
outlets we do not poll, at the cost of redirector links and no summary. Together
they give the dashboard both breadth and history; media_store dedupes the overlap
on (outlet, title).

Feeds are RSS 2.0 or Atom; remove_namespaces() + a tolerant field-by-field read
handles both without a feed-parsing dependency (house rule: the stdlib and what is
already installed). Failure is per request: one dead feed records an ingest_errors
row and costs that outlet's items only.
"""

from email.utils import parsedate_to_datetime
from typing import Any, AsyncIterator, Iterator
from urllib.parse import quote_plus

import scrapy
from sqlalchemy import text as sql

from app.config import (
    MEDIA_GOOGLE_NEWS_ENABLED,
    MEDIA_GOOGLE_NEWS_WINDOW_DAYS,
    MEDIA_MAX_KEYWORDS_PER_ISSUE,
)
from app.logging_utils import _log
from db.engine import SessionLocal
from ingest.html_text import html_to_text
from ingest.store import _record_error


def _parse_date(value: str | None) -> str | None:
    """RFC 822 (RSS) or ISO 8601 (Atom) -> ISO string, None when unparseable."""
    if not value:
        return None
    value = value.strip()
    try:
        return parsedate_to_datetime(value).isoformat()
    except (TypeError, ValueError):
        pass
    try:
        from datetime import datetime

        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


class NewsSpider(scrapy.Spider):
    name = "news"

    custom_settings = {
        "ITEM_PIPELINES": {
            "crawler.pipelines.ValidationPipeline": 100,
            "crawler.pipelines.MediaPostgresPipeline": 300,
        },
        # ~18 feeds across as many hosts plus ~15 Google News queries against one:
        # global concurrency can be generous while per-domain stays polite.
        "CONCURRENT_REQUESTS": 8,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "DOWNLOAD_DELAY": 0.5,
        "DOWNLOAD_TIMEOUT": 60,
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 2,
        "RETRY_HTTP_CODES": [408, 429, 500, 502, 503, 504],
    }

    def __init__(self, since: str | None = None, run_id: str | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.since = since or None
        self.run_id = run_id or None
        with SessionLocal() as db:
            self.outlets = [
                dict(r)
                for r in db.execute(
                    sql(
                        "SELECT id, slug, name, feed_url FROM media_outlets "
                        "WHERE is_active AND feed_url IS NOT NULL ORDER BY id"
                    )
                ).mappings()
            ]
            self.issues = [
                dict(r)
                for r in db.execute(
                    sql("SELECT id, slug, keywords FROM tracked_issues WHERE is_active")
                ).mappings()
            ]
        if not self.outlets and not self.issues:
            raise RuntimeError("no media outlets or tracked issues -- run migrations (0010)")

    def _seeds(self) -> Iterator[scrapy.Request]:
        for outlet in self.outlets:
            yield scrapy.Request(
                outlet["feed_url"],
                callback=self.parse_feed,
                errback=self.on_error,
                cb_kwargs={"outlet_id": outlet["id"], "outlet_slug": outlet["slug"]},
            )
        if MEDIA_GOOGLE_NEWS_ENABLED:
            for issue in self.issues:
                for keyword in (issue["keywords"] or [])[:MEDIA_MAX_KEYWORDS_PER_ISSUE]:
                    query = quote_plus(f'"{keyword}"') + f"+when:{MEDIA_GOOGLE_NEWS_WINDOW_DAYS}d"
                    yield scrapy.Request(
                        f"https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id",
                        callback=self.parse_google,
                        errback=self.on_error,
                        cb_kwargs={"issue_id": issue["id"], "keyword": keyword},
                    )

    # Both spellings for Scrapy 2.13 compatibility -- see ije_spider.py.
    async def start(self) -> AsyncIterator[scrapy.Request]:
        for request in self._seeds():
            yield request

    def start_requests(self) -> Iterator[scrapy.Request]:
        yield from self._seeds()

    def parse_feed(self, response: scrapy.http.Response, outlet_id: int, outlet_slug: str) -> Iterator[dict]:
        selector = response.selector
        selector.remove_namespaces()
        nodes = selector.xpath("//item") or selector.xpath("//entry")
        count = 0
        for node in nodes:
            title = html_to_text(node.xpath("string(title)").get() or "").strip()
            link = (
                node.xpath("link/@href").get()  # Atom
                or (node.xpath("string(link)").get() or "").strip()  # RSS
            )
            guid = (node.xpath("string(guid)").get() or node.xpath("string(id)").get() or "").strip()
            summary = html_to_text(
                node.xpath("string(description)").get()
                or node.xpath("string(summary)").get()
                or node.xpath("string(content)").get()
                or ""
            ).strip()
            published = _parse_date(
                node.xpath("string(pubDate)").get()
                or node.xpath("string(published)").get()
                or node.xpath("string(updated)").get()
            )
            if not title or not (link or guid):
                continue
            count += 1
            yield {
                "outlet_id": outlet_id,
                "external_id": guid or link,
                "url": link or guid,
                "title": title,
                "summary": summary[:2000] or None,
                "published_at": published,
                "found_via": "feed",
                "raw": {"feed": outlet_slug},
            }
        _log(f"news: {outlet_slug} -> {count} item(s)")

    def parse_google(self, response: scrapy.http.Response, issue_id: int, keyword: str) -> Iterator[dict]:
        selector = response.selector
        selector.remove_namespaces()
        count = 0
        for node in selector.xpath("//item"):
            raw_title = (node.xpath("string(title)").get() or "").strip()
            link = (node.xpath("string(link)").get() or "").strip()
            guid = (node.xpath("string(guid)").get() or "").strip() or link
            source_name = (node.xpath("string(source)").get() or "").strip()
            source_url = node.xpath("source/@url").get() or ""
            published = _parse_date(node.xpath("string(pubDate)").get())
            # Google formats titles as "Headline - Outlet"; strip the suffix when
            # it matches the source name so the same story dedupes against the
            # outlet's own feed copy.
            title = raw_title
            if source_name and title.endswith(f" - {source_name}"):
                title = title[: -(len(source_name) + 3)].strip()
            if not title or not link or not source_url:
                continue
            count += 1
            yield {
                # outlet_id resolved in MediaPostgresPipeline (it owns the session)
                "outlet_id": None,
                "source_name": source_name,
                "source_url": source_url,
                "issue_id": issue_id,
                "external_id": guid,
                "url": link,
                "title": title,
                "summary": None,
                "published_at": published,
                "found_via": "google-news",
                "raw": {"keyword": keyword},
            }
        _log(f"news: google '{keyword}' -> {count} item(s)")

    def on_error(self, failure: Any) -> None:
        """Record the miss and keep crawling -- a gap beats a lost outlet."""
        request = failure.request
        label = (
            request.cb_kwargs.get("outlet_slug")
            or f"google:{request.cb_kwargs.get('keyword', '?')}"
        )
        message = f"{label}: {failure.value!r}"
        _log(f"news: {message}")
        with SessionLocal() as db:
            _record_error(db, self.run_id, "media-crawl", None, label, message)
