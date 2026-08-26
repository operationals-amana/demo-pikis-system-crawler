"""
purnomoyusgiantorocenter.org over the WordPress REST API.

Ten public post types, paged at 100 via X-WP-TotalPages. Failure is per page, not
per run: one post type 404ing (types can be unregistered by a plugin change at any
time) must not cost the other nine -- the errback records the miss and the crawl
continues.

Parsing hands off to ingest/normalize.from_wp; Scrapy owns fetching only.
"""

from typing import Any, AsyncIterator, Iterator
from urllib.parse import urlencode

import scrapy
from sqlalchemy import text as sql

from app.config import PYC_POST_TYPES, PYC_WP_URL
from app.logging_utils import _log
from db.engine import SessionLocal
from ingest.normalize import from_wp
from ingest.store import _record_error

# Ask for only what we use; the default response embeds rendered HTML for a dozen
# fields we throw away, tripling the transfer.
FIELDS = "id,date,modified,link,slug,title,excerpt,content,categories,tags,type"
PER_PAGE = 100


class PycSpider(scrapy.Spider):
    name = "pyc-wp"

    custom_settings = {
        "ITEM_PIPELINES": {
            "crawler.pipelines.ValidationPipeline": 100,
            "crawler.pipelines.PostgresPipeline": 300,
        },
        # Two concurrent requests against a production WordPress is polite and still
        # finishes the full harvest in ~15 requests.
        "CONCURRENT_REQUESTS": 2,
        "DOWNLOAD_DELAY": 0.5,
        "DOWNLOAD_TIMEOUT": 120,
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 3,
        "RETRY_HTTP_CODES": [408, 429, 500, 502, 503, 504],
    }

    def __init__(self, since: str | None = None, run_id: str | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.since = since or None
        self.run_id = run_id or None
        with SessionLocal() as db:
            self.source_id = int(
                db.execute(sql("SELECT id FROM sources WHERE slug = 'pyc-wp'")).scalar() or 0
            )
        if not self.source_id:
            raise RuntimeError("source 'pyc-wp' missing -- run migrations (0006_seed.sql)")

    def _url(self, post_type: str, page: int) -> str:
        params: dict[str, Any] = {"per_page": PER_PAGE, "page": page, "_fields": FIELDS}
        if self.since:
            # WordPress expects full ISO 8601 here, not a bare date.
            params["modified_after"] = f"{self.since}T00:00:00"
        return f"{PYC_WP_URL}/{post_type}?{urlencode(params)}"

    def _seeds(self) -> Iterator[scrapy.Request]:
        for post_type in PYC_POST_TYPES:
            yield scrapy.Request(
                self._url(post_type, 1),
                callback=self.parse,
                errback=self.on_error,
                cb_kwargs={"post_type": post_type, "page": 1},
            )

    # Both spellings for Scrapy 2.13 compatibility -- see ije_spider.py.
    async def start(self) -> AsyncIterator[scrapy.Request]:
        for request in self._seeds():
            yield request

    def start_requests(self) -> Iterator[scrapy.Request]:
        yield from self._seeds()

    def parse(self, response: scrapy.http.Response, post_type: str, page: int) -> Iterator[Any]:
        posts = response.json()
        if not isinstance(posts, list):
            _log(f"pyc-wp: {post_type} page {page} returned a non-list; skipping")
            return

        if page == 1:
            total = response.headers.get("X-WP-Total", b"0").decode()
            total_pages = int(response.headers.get("X-WP-TotalPages", b"0").decode() or 0)
            _log(f"pyc-wp: {post_type} -> {total} item(s) across {total_pages} page(s)")
            for next_page in range(2, total_pages + 1):
                yield scrapy.Request(
                    self._url(post_type, next_page),
                    callback=self.parse,
                    errback=self.on_error,
                    cb_kwargs={"post_type": post_type, "page": next_page},
                )

        for post in posts:
            post["_post_type"] = post_type
            article = from_wp(post)
            if article:
                yield article

    def on_error(self, failure: Any) -> None:
        """Record the miss and keep crawling -- a gap beats a lost post type."""
        request = failure.request
        post_type = request.cb_kwargs.get("post_type", "?")
        page = request.cb_kwargs.get("page", "?")
        message = f"{post_type} page {page}: {failure.value!r}"
        _log(f"pyc-wp: {message}")
        with SessionLocal() as db:
            _record_error(db, self.run_id, "crawl", self.source_id, str(post_type), message)
