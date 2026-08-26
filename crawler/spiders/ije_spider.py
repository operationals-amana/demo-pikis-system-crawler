"""
Indonesian Journal of Energy over OAI-PMH.

One request usually returns the whole set (95 records, no resumption token) -- but
the resumption-token loop is implemented anyway: OJS pages at 100 and the journal
grows ~12 articles a year, so the day the corpus crosses 100 is the day a
single-request harvester starts silently truncating.

Parsing hands off to ingest/normalize.from_ije, so the Scrapy layer owns fetching
and nothing else -- the field mapping stayed byte-identical through the httpx->Scrapy
switch.
"""

from typing import Any, AsyncIterator, Iterator
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

import scrapy
from sqlalchemy import text as sql

from app.config import IJE_OAI_SET, IJE_OAI_URL
from app.logging_utils import _log
from db.engine import SessionLocal
from ingest.normalize import from_ije

NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
}


def _values(node: ET.Element, tag: str) -> list[str]:
    return [
        (element.text or "").strip()
        for element in node.findall(f".//dc:{tag}", NS)
        if (element.text or "").strip()
    ]


class IjeSpider(scrapy.Spider):
    name = "ije"

    custom_settings = {
        "ITEM_PIPELINES": {
            "crawler.pipelines.ValidationPipeline": 100,
            "crawler.pipelines.PostgresPipeline": 300,
        },
        # One OAI request per page of 100 records: concurrency buys nothing and a
        # polite delay costs nothing.
        "CONCURRENT_REQUESTS": 1,
        "DOWNLOAD_DELAY": 1.0,
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
                db.execute(sql("SELECT id FROM sources WHERE slug = 'ije'")).scalar() or 0
            )
        if not self.source_id:
            raise RuntimeError("source 'ije' missing -- run migrations (0006_seed.sql)")

    def _first_url(self) -> str:
        params = {"verb": "ListRecords", "metadataPrefix": "oai_dc", "set": IJE_OAI_SET}
        if self.since:
            params["from"] = self.since
        return f"{IJE_OAI_URL}?{urlencode(params)}"

    # Scrapy 2.13 replaced start_requests() with async start(); the removal is
    # silent, so a spider defining only the old one crawls nothing without raising.
    # Both are defined, like the sibling crawler.
    async def start(self) -> AsyncIterator[scrapy.Request]:
        yield scrapy.Request(self._first_url(), callback=self.parse)

    def start_requests(self) -> Iterator[scrapy.Request]:
        yield scrapy.Request(self._first_url(), callback=self.parse)

    def parse(self, response: scrapy.http.Response) -> Iterator[Any]:
        root = ET.fromstring(response.body)

        error = root.find("oai:error", NS)
        if error is not None:
            code = error.get("code", "")
            if code == "noRecordsMatch":
                # A quiet incremental day is a normal outcome, not a failure.
                _log("ije: no records match the window")
                return
            raise RuntimeError(f"OAI error {code}: {error.text}")

        records = root.findall(".//oai:record", NS)
        _log(f"ije: page -> {len(records)} record(s)")

        for record in records:
            header = record.find("oai:header", NS)
            identifier = header.findtext("oai:identifier", "", NS).strip()
            if header.get("status") == "deleted":
                # Withdrawn upstream (two known cases). Skipped, not archived: the
                # row simply never lands, matching the pre-Scrapy behaviour.
                continue

            meta = record.find(".//oai_dc:dc", NS)
            if meta is None:
                raw = {"external_id": identifier, "deleted": False, "empty": True}
            else:
                raw = {
                    "external_id": identifier,
                    "datestamp": header.findtext("oai:datestamp", "", NS).strip(),
                    "deleted": False,
                    "title": (_values(meta, "title") or [""])[0],
                    "creators": _values(meta, "creator"),
                    "subjects": _values(meta, "subject"),
                    "descriptions": _values(meta, "description"),
                    "publishers": _values(meta, "publisher"),
                    "dates": _values(meta, "date"),
                    "types": _values(meta, "type"),
                    "formats": _values(meta, "format"),
                    "identifiers": _values(meta, "identifier"),
                    "sources": _values(meta, "source"),
                    "languages": _values(meta, "language"),
                    "relations": _values(meta, "relation"),
                    "rights": _values(meta, "rights"),
                }
            article = from_ije(raw)
            if article:
                yield article

        token_element = root.find(".//oai:resumptionToken", NS)
        token = (token_element.text or "").strip() if token_element is not None else ""
        if token:
            # A resumption token replaces every other argument -- sending
            # metadataPrefix alongside it is an OAI protocol error.
            yield scrapy.Request(
                f"{IJE_OAI_URL}?{urlencode({'verb': 'ListRecords', 'resumptionToken': token})}",
                callback=self.parse,
            )
