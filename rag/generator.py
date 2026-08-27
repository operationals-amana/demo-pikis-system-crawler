"""
Anthropic calls: query rewriting, answer generation, session titling.

Two deliberate deviations from the sibling crawler's json_schema-everywhere convention,
both argued in the plan:

* The STREAMING answer uses XML tags, not output_config/json_schema. Half a JSON object
  ('{"answer": "Publikasi dalam tiga tahu') is unrenderable without a partial-JSON
  parser, and progressive prose is the entire point of the wireframe. Claude is also
  more reliable with XML tags than with brace-balanced JSON under a token cap.
* The model is never asked to emit a citations array. It can only emit [n] markers
  against a source list it was handed, which makes fabricating a title or URL
  structurally impossible rather than merely discouraged.

json_schema IS used everywhere it fits: rewriting, titling, and non-streaming answers.

Every call handles stop_reason 'refusal' and 'max_tokens' explicitly, and failures are
returned as values rather than raised -- a rewrite failure must degrade to the raw
question, not 500 the request.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from app.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    LLM_MAX_TOKENS,
    PRICE_CACHE_READ,
    PRICE_CACHE_WRITE,
    PRICE_IN,
    PRICE_OUT,
)
from app.logging_utils import _log
from rag.prompts import (
    ANSWER_SCHEMA,
    CHAT_SYSTEM,
    REWRITE_SCHEMA,
    REWRITE_SYSTEM,
    TITLE_SYSTEM,
)

_client = None
_async_client = None


def client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def async_client():
    global _async_client
    if _async_client is None:
        import anthropic

        _async_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return _async_client


def estimate_cost(usage: Any) -> float:
    if not usage:
        return 0.0
    return round(
        (getattr(usage, "input_tokens", 0) or 0) / 1e6 * PRICE_IN
        + (getattr(usage, "output_tokens", 0) or 0) / 1e6 * PRICE_OUT
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0) / 1e6 * PRICE_CACHE_WRITE
        + (getattr(usage, "cache_read_input_tokens", 0) or 0) / 1e6 * PRICE_CACHE_READ,
        6,
    )


# --- query rewriting -------------------------------------------------------


@dataclass
class Rewrite:
    standalone_question: str
    query_translated: str = ""
    language: str = "en"
    is_followup: bool = False
    # Intent tier 2 of the hybrid listing router (rag/listing.py has tier 1, the
    # regex). "listing" routes to the catalogue path; anything else is research.
    intent: str = "research"
    year_from: int | None = None
    year_to: int | None = None
    listing_topic: str = ""
    ok: bool = True


def _safe_year(value: Any) -> int | None:
    """The schema says integer-or-null, but a routing decision this consequential
    does not get to trust a model output blindly."""
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1980 <= year <= 2049 else None


def rewrite_query(question: str, history: list[dict[str, Any]] | None = None) -> Rewrite:
    """
    Resolve pronouns against the conversation AND translate.

    The translation is not a nicety: measured on the golden set, an Indonesian question
    ranks the correct English paper at 7-or-missing, while its English phrasing ranks
    it at 1. Supplying both phrasings is what makes cross-lingual retrieval work at all
    against this mostly-English corpus.

    Never raises. A failed rewrite degrades to the original question, which is worse
    retrieval but a working answer.
    """
    if not ANTHROPIC_API_KEY:
        return Rewrite(standalone_question=question, ok=False)

    context = ""
    if history:
        recent = history[-6:]
        lines = [
            f"{'User' if m.get('role') == 'user' else 'PIKIS'}: {(m.get('content') or '')[:300]}"
            for m in recent
            if m.get("content")
        ]
        context = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

    try:
        response = client().messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=300,
            system=REWRITE_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": REWRITE_SCHEMA}},
            messages=[{"role": "user", "content": f"{context}Question: {question}"}],
        )
        if response.stop_reason == "refusal":
            return Rewrite(standalone_question=question, ok=False)
        payload = json.loads(next(b.text for b in response.content if b.type == "text"))
        return Rewrite(
            standalone_question=payload.get("standalone_question") or question,
            query_translated=payload.get("query_translated") or "",
            language=payload.get("language") or "en",
            is_followup=bool(payload.get("is_followup")),
            intent="listing" if payload.get("intent") == "listing" else "research",
            year_from=_safe_year(payload.get("year_from")),
            year_to=_safe_year(payload.get("year_to")),
            listing_topic=(payload.get("listing_topic") or "").strip(),
        )
    except Exception as exc:  # noqa: BLE001 -- degrade, never block the answer
        _log(f"rewrite: failed ({exc}); using the question as-is")
        return Rewrite(standalone_question=question, ok=False)


def title_for(question: str) -> str | None:
    """Session title. Runs as a background task; failure leaves the truncated question."""
    if not ANTHROPIC_API_KEY:
        return None
    try:
        response = client().messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=32,
            system=TITLE_SYSTEM,
            messages=[{"role": "user", "content": question[:500]}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "").strip()
        return text.strip('"').strip() or None
    except Exception:  # noqa: BLE001 -- a missing title is cosmetic
        return None


# --- answer generation -----------------------------------------------------

_ANSWER_OPEN = re.compile(r"<answer>", re.I)
_ANSWER_CLOSE = re.compile(r"</answer>", re.I)
_FINDINGS_OPEN = re.compile(r"<findings>", re.I)
_FINDINGS_CLOSE = re.compile(r"</findings>", re.I)
_NO_EVIDENCE = re.compile(r"<no_evidence>(.*?)</no_evidence>", re.I | re.S)


@dataclass
class TagStream:
    """
    Three-state parser over the streamed text.

    States: before -> in_answer -> in_findings. Text inside <answer> is emitted as
    deltas; each completed <findings> bullet is emitted whole. A <no_evidence> opening
    is recognised within the first ~40 characters, which is what lets the API convert
    it into the same no_evidence event the pre-LLM gate emits -- so the user sees one
    consistent state regardless of which gate fired.
    """

    buffer: str = ""
    state: str = "before"
    answer: str = ""
    findings: list[str] = field(default_factory=list)
    no_evidence: str | None = None
    _emitted: int = 0
    _findings_buffer: str = ""

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        """Return [(kind, payload)] where kind is 'delta' | 'finding' | 'no_evidence'."""
        self.buffer += chunk
        events: list[tuple[str, str]] = []

        if self.no_evidence is not None:
            return events

        if self.state == "before":
            match = _NO_EVIDENCE.search(self.buffer)
            if match:
                self.no_evidence = match.group(1).strip()
                return [("no_evidence", self.no_evidence)]
            # Only wait for a tag while one could still be forming.
            if "<no_evidence" in self.buffer and "</no_evidence>" not in self.buffer:
                return events
            opened = _ANSWER_OPEN.search(self.buffer)
            if opened:
                self.state = "in_answer"
                self.buffer = self.buffer[opened.end():]
                self._emitted = 0
            else:
                return events

        if self.state == "in_answer":
            closed = _ANSWER_CLOSE.search(self.buffer)
            body = self.buffer[: closed.start()] if closed else self.buffer
            # Hold back a partial closing tag so "</a" never reaches the client.
            safe = body if closed else re.sub(r"<[a-z/]*$", "", body, flags=re.I)
            if len(safe) > self._emitted:
                delta = safe[self._emitted:]
                self._emitted = len(safe)
                self.answer += delta
                events.append(("delta", delta))
            if closed:
                self.buffer = self.buffer[closed.end():]
                self.state = "in_findings"

        if self.state == "in_findings":
            # Accumulate FIRST, then look for the opening tag in the accumulated text.
            # Searching the current chunk instead lets a tag split across chunk
            # boundaries ("\n<findi" + "ngs>\n- P") slip through and be emitted as a
            # finding -- which is exactly what happened before this was restructured.
            self._findings_buffer += self.buffer
            self.buffer = ""
            opened = _FINDINGS_OPEN.search(self._findings_buffer)
            if opened:
                self._findings_buffer = self._findings_buffer[opened.end():]
            elif "<" in self._findings_buffer and "<findings>" not in self._findings_buffer:
                # A partial opening tag may still be forming; wait for the rest.
                if re.search(r"<f?i?n?d?i?n?g?s?$", self._findings_buffer):
                    return events
            closing = _FINDINGS_CLOSE.search(self._findings_buffer)
            body = self._findings_buffer[: closing.start()] if closing else self._findings_buffer
            lines = body.split("\n")
            # Keep the last (possibly incomplete) line buffered unless we saw the close.
            complete = lines if closing else lines[:-1]
            for line in complete:
                item = line.strip().lstrip("-*").strip()
                if item and item not in self.findings:
                    self.findings.append(item)
                    events.append(("finding", item))
            if not closing:
                self._findings_buffer = lines[-1] if lines else ""
            else:
                self._findings_buffer = ""
                self.state = "done"

        return events


def build_messages(
    question: str, sources_block: str, history_block: str = ""
) -> list[dict[str, Any]]:
    parts = []
    if history_block:
        parts.append(history_block)
    parts.append(sources_block)
    parts.append(f"Question: {question}")
    return [{"role": "user", "content": "\n\n".join(parts)}]


async def stream_answer(
    question: str, sources_block: str, history_block: str = "", max_tokens: int = LLM_MAX_TOKENS
) -> AsyncIterator[tuple[str, Any]]:
    """
    Yield ('delta'|'finding'|'no_evidence'|'done'|'error', payload).

    'done' carries {answer, key_findings, usage, cost_usd, finish_reason}.
    """
    parser = TagStream()
    usage = None
    finish_reason = None

    try:
        async with async_client().messages.stream(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=CHAT_SYSTEM,
            messages=build_messages(question, sources_block, history_block),
        ) as stream:
            async for chunk in stream.text_stream:
                for kind, payload in parser.feed(chunk):
                    yield kind, payload
                    if kind == "no_evidence":
                        return
            final = await stream.get_final_message()
            usage = final.usage
            finish_reason = final.stop_reason
            if final.stop_reason == "refusal":
                yield "error", {"code": "refusal", "message": "The model declined this request."}
                return
    except Exception as exc:  # noqa: BLE001 -- surface as an event, never a 500 on a
                              # connection the client is already streaming.
        _log(f"generate: stream failed ({exc})")
        yield "error", {"code": "stream_failed", "message": str(exc)}
        return

    yield "done", {
        "answer": parser.answer.strip(),
        "key_findings": parser.findings,
        "usage": usage,
        "cost_usd": estimate_cost(usage),
        # 'max_tokens' means the answer was cut off mid-thought; the API records it so
        # the UI can say so rather than presenting a truncated answer as complete.
        "finish_reason": finish_reason,
    }


def answer_sync(
    question: str, sources_block: str, history_block: str = "", max_tokens: int = LLM_MAX_TOKENS
) -> dict[str, Any]:
    """Non-streaming path. Here json_schema fits, so it is used."""
    try:
        response = client().messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=CHAT_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
            messages=build_messages(question, sources_block, history_block),
        )
        if response.stop_reason == "refusal":
            return {"error": "refusal", "answer": "", "key_findings": [], "no_evidence": True}
        payload = json.loads(next(b.text for b in response.content if b.type == "text"))
        return {
            "answer": payload.get("answer", ""),
            "key_findings": payload.get("key_findings", []),
            "no_evidence": bool(payload.get("insufficient_evidence")),
            "usage": response.usage,
            "cost_usd": estimate_cost(response.usage),
            "finish_reason": response.stop_reason,
        }
    except Exception as exc:  # noqa: BLE001
        _log(f"generate: sync failed ({exc})")
        return {"error": str(exc), "answer": "", "key_findings": [], "no_evidence": True}
