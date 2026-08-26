"""
Server-sent event framing.

Two rules that matter more than they look:

1. Every `data:` line must be SINGLE-LINE JSON. A raw newline inside a data payload
   terminates the frame early and the client sees a truncated event -- the single most
   common SSE bug. Putting the text INSIDE a JSON object (rather than sending raw
   text) is what makes that safe, because json.dumps escapes newlines for us.

2. Heartbeats from t=0. Retrieval plus rewriting can be two seconds of silence, and
   intermediaries (Railway's ingress, corporate proxies) idle-timeout quiet
   connections. A `:` comment line is valid SSE and is ignored by every client.
"""

import json
from typing import Any


def frame(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    # Belt and braces: json.dumps already escapes newlines, but a stray one here would
    # silently corrupt the stream, so make it impossible.
    payload = payload.replace("\r", "").replace("\n", " ")
    return f"event: {event}\ndata: {payload}\n\n"


def comment(text: str = "heartbeat") -> str:
    return f": {text}\n\n"


# Headers required for the stream to survive the Next.js proxy on Vercel and any
# nginx-family ingress in between. `no-transform` stops intermediaries re-encoding
# (which re-chunks and therefore buffers); X-Accel-Buffering is nginx's documented
# opt-out from response buffering.
SSE_HEADERS = {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-store, no-transform, must-revalidate",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
