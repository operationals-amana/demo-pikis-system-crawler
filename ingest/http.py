"""One shared httpx client with retry/backoff, used by both harvesters."""

import time
from typing import Any

import httpx

from app.config import HTTP_RETRIES, HTTP_TIMEOUT, USER_AGENT
from app.logging_utils import _log

_RETRYABLE = {408, 429, 500, 502, 503, 504}


def client() -> httpx.Client:
    return httpx.Client(
        timeout=HTTP_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )


def get(c: httpx.Client, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
    """
    GET with bounded exponential backoff.

    Raises on final failure -- callers wrap this at the stage boundary and record an
    ingest_errors row, so one dead page never aborts a whole run.
    """
    last: Exception | None = None
    for attempt in range(HTTP_RETRIES):
        try:
            response = c.get(url, params=params)
            if response.status_code in _RETRYABLE:
                raise httpx.HTTPStatusError(
                    f"retryable {response.status_code}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            # A 4xx that is not 408/429 is a permanent answer: the galley genuinely
            # does not exist. Retrying it three times with backoff just makes a
            # failing harvest slower, and buries the real error under retry noise.
            status = exc.response.status_code if exc.response is not None else 0
            if status and status not in _RETRYABLE:
                raise RuntimeError(f"GET {url} -> {status} (not retryable)") from exc
            last = exc
            if attempt < HTTP_RETRIES - 1:
                delay = 2**attempt
                _log(f"http: {url} -> {status}; retry {attempt + 1} in {delay}s")
                time.sleep(delay)
        except Exception as exc:  # noqa: BLE001 -- transport errors are retryable
            last = exc
            if attempt < HTTP_RETRIES - 1:
                delay = 2**attempt
                _log(f"http: {url} failed ({exc}); retry {attempt + 1} in {delay}s")
                time.sleep(delay)
    raise RuntimeError(f"GET {url} failed after {HTTP_RETRIES} attempts: {last}")
