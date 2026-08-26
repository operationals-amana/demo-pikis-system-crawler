"""
Scrapy settings for the PIKIS crawler.

Both upstreams are structured APIs (OAI-PMH XML, WordPress REST JSON), not HTML to
spider -- what Scrapy contributes here is its retry middleware, polite throttling,
and the operational shape shared with the sibling tender-intelligence crawler.
Per-spider tuning lives in each spider's custom_settings, matching that convention.
"""

from app.config import USER_AGENT

BOT_NAME = "pikis"

SPIDER_MODULES = ["crawler.spiders"]
NEWSPIDER_MODULE = "crawler.spiders"

# APIs, not pages: robots.txt does not apply to the OAI endpoint and WordPress
# serves its REST API regardless; fetching robots.txt would just add a request.
ROBOTSTXT_OBEY = False

USER_AGENT = USER_AGENT

# Scrapy's own noise level. The pipelines and run_crawl print the lines that matter.
LOG_LEVEL = "INFO"

# Fail fast on schema drift instead of silently coercing.
FEED_EXPORT_ENCODING = "utf-8"

REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
