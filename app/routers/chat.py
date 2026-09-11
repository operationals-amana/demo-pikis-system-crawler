"""
Chat: sessions, grounded answers (streaming and not), and feedback.

The grounding contract is enforced in three places and this router wires them:

  Gate 1  retrieval score threshold -- the LLM is NEVER CALLED when evidence is thin.
          This is the only mechanism that actually guarantees the acceptance
          criterion; a prompt instruction is a request, a threshold is a guarantee.
  Gate 2  the prompt's <no_evidence> escape, for when the retrieved chunks are
          high-scoring but about the wrong aspect.
  Gate 3  citation validation, which strips indices the model invented.
"""

import asyncio
import json
import time
from typing import Any, AsyncIterator

from fastapi import APIRouter, BackgroundTasks, Depends, Header, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import text as sql
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.config import (
    CHAT_HISTORY_TURNS,
    ENABLE_QUERY_TRANSLATION,
    SSE_HEARTBEAT_SECONDS,
    TRANSLATE_FROM_LANGUAGES,
)
from app.deps import current_user, get_db
from app.errors import not_found
from app.logging_utils import _log
from app.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    SessionCreate,
    SessionOut,
    SessionUpdate,
)
from app.sse import SSE_HEADERS, comment, frame
from db.engine import SessionLocal
from rag.citations import build_citations, strip_content, validate
from rag.context import fit
from rag.prompts import format_history

router = APIRouter(prefix="/api", tags=["chat"])


# --- sessions --------------------------------------------------------------

def _session_row_to_out(row: dict[str, Any]) -> SessionOut:
    return SessionOut(
        id=row["id"],
        title=row.get("title"),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        last_message_at=str(row["last_message_at"]) if row.get("last_message_at") else None,
        message_count=row.get("message_count") or 0,
    )


@router.get("/sessions", response_model=list[SessionOut])
def list_sessions(
    db: Session = Depends(get_db), user: dict = Depends(current_user), limit: int = 50
) -> list[SessionOut]:
    rows = db.execute(
        sql(
            "SELECT id::text, title, created_at, updated_at, last_message_at, message_count "
            "FROM chat_sessions WHERE user_id = CAST(:u AS uuid) AND deleted_at IS NULL "
            "ORDER BY COALESCE(last_message_at, created_at) DESC LIMIT :l"
        ),
        {"u": user["id"], "l": min(limit, 200)},
    ).mappings().all()
    return [_session_row_to_out(dict(r)) for r in rows]


@router.post("/sessions", response_model=SessionOut)
def create_session(
    payload: SessionCreate, db: Session = Depends(get_db), user: dict = Depends(current_user)
) -> SessionOut:
    row = db.execute(
        sql(
            "INSERT INTO chat_sessions (user_id, title, filters) "
            "VALUES (CAST(:u AS uuid), :t, CAST(:f AS jsonb)) "
            "RETURNING id::text, title, created_at, updated_at, last_message_at, message_count"
        ),
        {
            "u": user["id"],
            "t": payload.title,
            "f": json.dumps(payload.filters.model_dump(mode="json") if payload.filters else {}),
        },
    ).mappings().first()
    db.commit()
    return _session_row_to_out(dict(row))


@router.get("/sessions/{session_id}")
def get_session(
    session_id: str, db: Session = Depends(get_db), user: dict = Depends(current_user)
) -> dict[str, Any]:
    row = db.execute(
        sql(
            "SELECT id::text, title, created_at, updated_at, last_message_at, message_count, filters "
            "FROM chat_sessions WHERE id = CAST(:i AS uuid) AND user_id = CAST(:u AS uuid) "
            "AND deleted_at IS NULL"
        ),
        {"i": session_id, "u": user["id"]},
    ).mappings().first()
    if not row:
        raise not_found("Session")

    messages = db.execute(
        sql(
            "SELECT id::text, seq, role, content, key_findings, citations, grounded, "
            "no_evidence, model, latency_ms, created_at "
            "FROM chat_messages WHERE session_id = CAST(:i AS uuid) ORDER BY seq"
        ),
        {"i": session_id},
    ).mappings().all()

    return {
        "session": _session_row_to_out(dict(row)).model_dump(),
        "filters": row.get("filters") or {},
        "messages": [
            {
                **{k: v for k, v in dict(m).items() if k != "created_at"},
                "created_at": str(m["created_at"]),
            }
            for m in messages
        ],
    }


@router.patch("/sessions/{session_id}", response_model=SessionOut)
def update_session(
    session_id: str,
    payload: SessionUpdate,
    db: Session = Depends(get_db),
    user: dict = Depends(current_user),
) -> SessionOut:
    row = db.execute(
        sql(
            "UPDATE chat_sessions SET "
            "title = COALESCE(:t, title), "
            "title_source = CASE WHEN :t IS NULL THEN title_source ELSE 'manual' END, "
            "filters = COALESCE(CAST(:f AS jsonb), filters), updated_at = now() "
            "WHERE id = CAST(:i AS uuid) AND user_id = CAST(:u AS uuid) AND deleted_at IS NULL "
            "RETURNING id::text, title, created_at, updated_at, last_message_at, message_count"
        ),
        {
            "i": session_id, "u": user["id"], "t": payload.title,
            "f": json.dumps(payload.filters.model_dump(mode="json")) if payload.filters else None,
        },
    ).mappings().first()
    if not row:
        raise not_found("Session")
    db.commit()
    return _session_row_to_out(dict(row))


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(
    session_id: str, db: Session = Depends(get_db), user: dict = Depends(current_user)
) -> None:
    """
    SOFT delete.

    A hard delete would destroy the audit log issue #1 requires -- query, sources used,
    response and feedback all hang off these rows. The session disappears from the
    user's sidebar, which is what "delete" means to them, while the audit trail
    survives. Flagged in the plan as a product decision for PYC to confirm.
    """
    result = db.execute(
        sql(
            "UPDATE chat_sessions SET deleted_at = now() "
            "WHERE id = CAST(:i AS uuid) AND user_id = CAST(:u AS uuid) AND deleted_at IS NULL"
        ),
        {"i": session_id, "u": user["id"]},
    )
    if not result.rowcount:
        raise not_found("Session")
    db.commit()


# --- the answer pipeline ---------------------------------------------------

def _history(db: Session, session_id: str | None) -> list[dict[str, Any]]:
    if not session_id:
        return []
    rows = db.execute(
        sql(
            "SELECT role, content, citations FROM chat_messages "
            "WHERE session_id = CAST(:i AS uuid) ORDER BY seq DESC LIMIT :n"
        ),
        {"i": session_id, "n": CHAT_HISTORY_TURNS * 2},
    ).mappings().all()
    return [dict(r) for r in reversed(rows)]


def _ensure_session(db: Session, user_id: str, session_id: str | None, question: str) -> str:
    """
    Lazily create the session on first message.

    This is why the UI's "New research" button creates nothing: a session row that
    exists before the first question produces a sidebar full of empty "New chat"
    ghosts, which is exactly what claude.ai avoids.
    """
    if session_id:
        owned = db.execute(
            sql(
                "SELECT 1 FROM chat_sessions WHERE id = CAST(:i AS uuid) "
                "AND user_id = CAST(:u AS uuid) AND deleted_at IS NULL"
            ),
            {"i": session_id, "u": user_id},
        ).scalar()
        if not owned:
            raise not_found("Session")
        return session_id

    new_id = db.execute(
        sql(
            "INSERT INTO chat_sessions (user_id, title, title_source) "
            "VALUES (CAST(:u AS uuid), :t, 'auto') RETURNING id::text"
        ),
        {"u": user_id, "t": question[:80]},
    ).scalar()
    db.commit()
    return str(new_id)


def _next_seq(db: Session, session_id: str) -> int:
    return int(
        db.execute(
            sql("SELECT COALESCE(MAX(seq), 0) + 1 FROM chat_messages WHERE session_id = CAST(:i AS uuid)"),
            {"i": session_id},
        ).scalar()
    )


def _persist_user_message(db: Session, session_id: str, content: str) -> str:
    seq = _next_seq(db, session_id)
    message_id = db.execute(
        sql(
            "INSERT INTO chat_messages (session_id, seq, role, content) "
            "VALUES (CAST(:s AS uuid), :q, 'user', :c) RETURNING id::text"
        ),
        {"s": session_id, "q": seq, "c": content},
    ).scalar()
    db.commit()
    return str(message_id)


def _persist_assistant(
    db: Session, session_id: str, payload: dict[str, Any]
) -> str:
    """
    Write the assistant turn.

    Called even when the stream was cut, which is what lets a client truncated by a
    Vercel maxDuration kill recover the full text via GET /api/sessions/{id}.
    """
    seq = _next_seq(db, session_id)
    usage = payload.get("usage")
    message_id = db.execute(
        sql(
            "INSERT INTO chat_messages (session_id, seq, role, content, key_findings, "
            "citations, retrieval, grounded, no_evidence, model, input_tokens, "
            "output_tokens, cost_usd, latency_ms, finish_reason, error) "
            "VALUES (CAST(:s AS uuid), :q, 'assistant', :c, CAST(:kf AS jsonb), "
            "CAST(:ci AS jsonb), CAST(:rt AS jsonb), :g, :ne, :m, :it, :ot, :cost, :ms, :fr, :err) "
            "RETURNING id::text"
        ),
        {
            "s": session_id, "q": seq, "c": payload.get("answer") or "",
            "kf": json.dumps(payload.get("key_findings") or []),
            "ci": json.dumps(payload.get("citations") or []),
            "rt": json.dumps(payload.get("retrieval") or {}, default=str),
            "g": payload.get("grounded", True), "ne": payload.get("no_evidence", False),
            "m": payload.get("model"),
            "it": getattr(usage, "input_tokens", None) if usage else None,
            "ot": getattr(usage, "output_tokens", None) if usage else None,
            "cost": payload.get("cost_usd"), "ms": payload.get("latency_ms"),
            "fr": payload.get("finish_reason"), "err": payload.get("error"),
        },
    ).scalar()
    db.execute(
        sql(
            "UPDATE chat_sessions SET last_message_at = now(), updated_at = now(), "
            "message_count = (SELECT count(*) FROM chat_messages WHERE session_id = CAST(:s AS uuid)) "
            "WHERE id = CAST(:s AS uuid)"
        ),
        {"s": session_id},
    )
    db.commit()
    return str(message_id)


def _log_query(db: Session, user_id: str, session_id: str, message_id: str | None,
               question: str, rewritten: str | None, filters: dict, result, had_evidence: bool,
               latency_ms: int) -> None:
    db.execute(
        sql(
            "INSERT INTO query_log (user_id, session_id, message_id, endpoint, query, "
            "query_rewritten, filters, mode, retrieved_chunk_ids, result_count, "
            "top_scores, had_evidence, latency_ms, index_version) "
            "VALUES (CAST(:u AS uuid), CAST(:s AS uuid), CAST(:m AS uuid), 'chat', :q, :qr, "
            "CAST(:f AS jsonb), 'hybrid', :ids, :n, :sc, :he, :ms, :iv)"
        ),
        {
            "u": user_id, "s": session_id, "m": message_id, "q": question, "qr": rewritten,
            "f": json.dumps(filters or {}), "ids": [c.get("id") for c in result.candidates][:50],
            "n": len(result.candidates),
            "sc": [round(float(c.get("rerank") or 0), 5) for c in result.candidates][:10],
            "he": had_evidence, "ms": latency_ms, "iv": result.index_version,
        },
    )
    db.commit()


def _log_listing(db: Session, user_id: str, session_id: str, message_id: str | None,
                 question: str, listing: dict, latency_ms: int) -> None:
    """query_log row for the deterministic listing path -- mode 'listing', no chunks."""
    db.execute(
        sql(
            "INSERT INTO query_log (user_id, session_id, message_id, endpoint, query, "
            "filters, mode, result_count, had_evidence, latency_ms) "
            "VALUES (CAST(:u AS uuid), CAST(:s AS uuid), CAST(:m AS uuid), 'chat', :q, "
            "CAST(:f AS jsonb), 'listing', :n, :he, :ms)"
        ),
        {
            "u": user_id, "s": session_id, "m": message_id, "q": question,
            "f": json.dumps(listing.get("filters") or {}),
            "n": listing.get("shown") or 0, "he": bool(listing.get("total")),
            "ms": latency_ms,
        },
    )
    db.commit()


def _listing_payload(listing: dict) -> dict[str, Any]:
    """The assistant-message dict shared by the sync and streaming listing paths."""
    return {
        "answer": listing["answer"], "citations": listing["citations"],
        "no_evidence": False, "grounded": True, "model": None,
        "retrieval": {"mode": "listing", "total": listing["total"],
                      "shown": listing["shown"], "filters": listing["filters"],
                      "topic": listing["topic"]},
    }


def _finish_listing_sync(db: Session, user: dict, session_id: str, question: str,
                         listing: dict, started: float) -> "ChatResponse":
    latency = int((time.perf_counter() - started) * 1000)
    message_id = _persist_assistant(
        db, session_id, {**_listing_payload(listing), "latency_ms": latency}
    )
    _log_listing(db, user["id"], session_id, message_id, question, listing, latency)
    return ChatResponse(
        message_id=message_id, session_id=session_id, answer=listing["answer"],
        citations=listing["citations"], grounded=True, latency_ms=latency,
    )


async def _emit_listing(db: Session, user: dict, session_id: str, question: str,
                        listing: dict, started: float, background: BackgroundTasks,
                        is_new: bool) -> AsyncIterator[str]:
    """The SSE tail of a listing answer, shared by both router tiers."""
    latency = int((time.perf_counter() - started) * 1000)
    message_id = await run_in_threadpool(
        _persist_assistant, db, session_id,
        {**_listing_payload(listing), "latency_ms": latency},
    )
    await run_in_threadpool(
        _log_listing, db, user["id"], session_id, message_id, question, listing, latency
    )
    # Sources first, so every [n] in the text is resolvable when it paints.
    yield frame("sources", {"citations": listing["citations"]})
    yield frame("status", {"stage": "generating", "sources": len(listing["citations"])})
    yield frame("delta", {"text": listing["answer"]})
    yield frame("done", {
        "message_id": message_id, "session_id": session_id,
        "answer": listing["answer"], "key_findings": [],
        "citations_used": [c["index"] for c in listing["citations"]],
        "citations": listing["citations"], "grounded": True, "no_evidence": False,
        "elapsed_ms": latency, "source_count": len(listing["citations"]),
        "usage": {"input_tokens": None, "output_tokens": None, "cost_usd": None},
        "finish_reason": "listing",
    })
    if is_new:
        background.add_task(_retitle, session_id, question)


def _prepare(db: Session, question: str, session_id: str | None, filters_spec, top_k: int):
    """Rewrite -> retrieve -> build citations -> fit the context budget."""
    from app.routers.search import to_filters
    from rag.generator import rewrite_query
    from rag.metadata_filters import extract_explicit_filters
    from rag.retriever import retrieve

    history = _history(db, session_id)
    rewritten_text = question
    extra: list[str] = []

    # Skipped entirely on the first turn with no history AND no translation need --
    # that saves ~400 ms on the most latency-sensitive request a user makes.
    if history or ENABLE_QUERY_TRANSLATION:
        rewritten = rewrite_query(question, history)
        rewritten_text = rewritten.standalone_question or question
        if (
            ENABLE_QUERY_TRANSLATION
            and rewritten.query_translated
            and rewritten.language in TRANSLATE_FROM_LANGUAGES
        ):
            extra.append(rewritten.query_translated)

        # LISTING ROUTER, TIER 2. Tier 1 (the regex in rag/listing.py) already ran on
        # the raw question and missed, so this phrasing is one the regex does not
        # know -- informal, misspelled, or a follow-up ("kalau 2019?") that only
        # becomes a listing once history resolves it. The rewrite call was happening
        # anyway, so this tier costs zero extra latency; and when the rewrite fails,
        # `intent` stays "research" and the request degrades to exactly the old
        # behavior. The asymmetric prompt ("when unsure, use research") plus this
        # ok-check keeps the dangerous direction -- a research question hijacked into
        # a catalogue dump -- behind two independent guards.
        if rewritten.ok and rewritten.intent == "listing":
            from rag.listing import ListingIntent, run_listing

            listing = run_listing(
                db,
                ListingIntent(
                    year_from=rewritten.year_from,
                    year_to=rewritten.year_to,
                    topic_text=rewritten.listing_topic,
                    language=rewritten.language or "en",
                ),
                extract_explicit_filters(db, rewritten_text, to_filters(filters_spec)),
            )
            return {"listing": listing, "rewritten": rewritten_text, "history": history}

    filters = extract_explicit_filters(db, rewritten_text, to_filters(filters_spec))
    result = retrieve(db, rewritten_text, filters=filters, top_k=top_k, extra_queries=extra or None)
    citations = build_citations(result.candidates, result.query_terms)
    kept, sources_block, budget_info = fit(citations)
    history_block = format_history(history, CHAT_HISTORY_TURNS)
    return {
        "history": history,
        "history_block": history_block,
        "rewritten": rewritten_text,
        "result": result,
        "citations": kept,
        "sources_block": sources_block,
        "budget": budget_info,
        "filters": filters,
    }


NO_EVIDENCE_TEXT = {
    "id": "Tidak ditemukan bukti yang memadai dalam koleksi PIKIS untuk menjawab pertanyaan ini.",
    "en": "No sufficient evidence was found in the PIKIS collection to answer this question.",
}


def _no_evidence_message(question: str) -> str:
    from ingest.language import detect

    language, _ = detect(question)
    return NO_EVIDENCE_TEXT.get(language or "en", NO_EVIDENCE_TEXT["en"])


@router.post("/chat")
async def chat(
    payload: ChatRequest,
    request: Request,
    background: BackgroundTasks,
    user: dict = Depends(current_user),
    accept: str | None = Header(default=None),
):
    """
    Grounded answer. SSE when `stream` is true or Accept: text/event-stream.

    This handler is `async def` -- the only one in the service. A 20-second stream in a
    sync handler would occupy a threadpool worker for the whole 20 seconds; the DB work
    inside it goes through run_in_threadpool instead.
    """
    wants_stream = payload.stream or (accept or "").startswith("text/event-stream")
    if not wants_stream:
        return await run_in_threadpool(_chat_sync, payload, user)
    return StreamingResponse(
        _chat_stream(payload, user, request, background),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


def _chat_sync(payload: ChatRequest, user: dict) -> ChatResponse:
    from app.config import ANTHROPIC_MODEL
    from rag.generator import answer_sync

    db = SessionLocal()
    started = time.perf_counter()
    try:
        session_id = _ensure_session(db, user["id"], payload.session_id, payload.message)
        _persist_user_message(db, session_id, payload.message)

        # LISTING ROUTER, TIER 1 -- "list artikel di 2018" is a catalogue ask, not a
        # research question. No chunk is semantically similar to it, so Gate 1 would
        # refuse; the database answers it exactly instead, with no LLM call at all.
        # Phrasings the regex misses get a second chance inside _prepare, where the
        # rewrite LLM classifies intent. See rag/listing.py.
        from app.routers.search import to_filters
        from rag.listing import detect_listing, run_listing
        from rag.metadata_filters import extract_explicit_filters

        intent = detect_listing(payload.message)
        if intent is not None:
            filters = extract_explicit_filters(db, payload.message, to_filters(payload.filters))
            listing = run_listing(db, intent, filters)
            return _finish_listing_sync(db, user, session_id, payload.message, listing, started)

        prepared = _prepare(db, payload.message, session_id, payload.filters, payload.top_k)
        if prepared.get("listing"):  # LISTING ROUTER, TIER 2 (LLM) -- see _prepare
            return _finish_listing_sync(
                db, user, session_id, payload.message, prepared["listing"], started
            )
        result = prepared["result"]

        # GATE 1 -- the LLM is never called when evidence is thin.
        if not result.has_evidence():
            text = _no_evidence_message(payload.message)
            latency = int((time.perf_counter() - started) * 1000)
            message_id = _persist_assistant(db, session_id, {
                "answer": text, "citations": [], "no_evidence": True, "grounded": True,
                "latency_ms": latency, "model": None,
                "retrieval": {"best_score": result.best_score(), "counts": result.counts},
            })
            _log_query(db, user["id"], session_id, message_id, payload.message,
                       prepared["rewritten"], result_filters(prepared), result, False, latency)
            return ChatResponse(message_id=message_id, session_id=session_id, answer=text,
                                no_evidence=True, grounded=True, latency_ms=latency)

        generated = answer_sync(payload.message, prepared["sources_block"], prepared["history_block"])
        # GATE 3 -- strip invented citation indices.
        checked = validate(generated.get("answer", ""), prepared["citations"])
        latency = int((time.perf_counter() - started) * 1000)

        message_id = _persist_assistant(db, session_id, {
            "answer": checked["answer"], "key_findings": generated.get("key_findings"),
            "citations": strip_content(checked["citations"]), "grounded": checked["grounded"],
            "no_evidence": bool(generated.get("no_evidence")), "model": ANTHROPIC_MODEL,
            "usage": generated.get("usage"), "cost_usd": generated.get("cost_usd"),
            "latency_ms": latency, "finish_reason": generated.get("finish_reason"),
            "error": generated.get("error"),
            "retrieval": {"counts": result.counts, "timings": result.timings,
                          "invalid_citations": checked["invalid"], "budget": prepared["budget"]},
        })
        _log_query(db, user["id"], session_id, message_id, payload.message,
                   prepared["rewritten"], result_filters(prepared), result, True, latency)

        return ChatResponse(
            message_id=message_id, session_id=session_id, answer=checked["answer"],
            key_findings=generated.get("key_findings") or [],
            citations=strip_content(checked["citations"]),
            no_evidence=bool(generated.get("no_evidence")), grounded=checked["grounded"],
            model=ANTHROPIC_MODEL, latency_ms=latency,
        )
    finally:
        db.close()


def result_filters(prepared: dict) -> dict:
    filters = prepared.get("filters")
    return filters.describe() if filters else {}


async def _chat_stream(
    payload: ChatRequest, user: dict, request: Request, background: BackgroundTasks
) -> AsyncIterator[str]:
    from app.config import ANTHROPIC_MODEL
    from rag.generator import stream_answer, title_for

    db = SessionLocal()
    started = time.perf_counter()
    heartbeat_at = time.monotonic()

    try:
        session_id = await run_in_threadpool(
            _ensure_session, db, user["id"], payload.session_id, payload.message
        )
        await run_in_threadpool(_persist_user_message, db, session_id, payload.message)

        is_new = payload.session_id is None
        yield frame("meta", {
            "session_id": session_id, "model": ANTHROPIC_MODEL, "is_new_session": is_new,
        })

        # LISTING ROUTER, TIER 1 -- catalogue asks are answered from article metadata
        # with no LLM call; Gate 1 would otherwise refuse them. See rag/listing.py.
        # The stage is reported as 'retrieving' because that is the closest state the
        # frontend's closed ChatPhase union knows.
        from app.routers.search import to_filters
        from rag.listing import detect_listing, run_listing
        from rag.metadata_filters import extract_explicit_filters

        intent = detect_listing(payload.message)
        if intent is not None:
            yield frame("status", {"stage": "retrieving"})
            filters = extract_explicit_filters(db, payload.message, to_filters(payload.filters))
            listing = await run_in_threadpool(
                run_listing, db, intent, filters
            )
            async for event in _emit_listing(
                db, user, session_id, payload.message, listing, started, background, is_new
            ):
                yield event
            return

        yield frame("status", {"stage": "rewriting"})
        yield comment()

        prepared = await run_in_threadpool(
            _prepare, db, payload.message, session_id, payload.filters, payload.top_k
        )
        if prepared.get("listing"):  # LISTING ROUTER, TIER 2 (LLM) -- see _prepare
            yield frame("status", {"stage": "retrieving"})
            async for event in _emit_listing(
                db, user, session_id, payload.message, prepared["listing"], started,
                background, is_new,
            ):
                yield event
            return
        result = prepared["result"]
        yield frame("status", {"stage": "retrieving", "candidates": result.counts.get("fused", 0)})

        # GATE 1 -- deterministic, pre-LLM. This is what actually guarantees the
        # "no substantive answer without evidence" acceptance criterion.
        if not result.has_evidence():
            text = _no_evidence_message(payload.message)
            latency = int((time.perf_counter() - started) * 1000)
            message_id = await run_in_threadpool(_persist_assistant, db, session_id, {
                "answer": text, "citations": [], "no_evidence": True, "grounded": True,
                "latency_ms": latency,
                "retrieval": {"best_score": result.best_score(), "counts": result.counts},
            })
            await run_in_threadpool(
                _log_query, db, user["id"], session_id, message_id, payload.message,
                prepared["rewritten"], result_filters(prepared), result, False, latency
            )
            yield frame("no_evidence", {
                "detail": text,
                "best_score": round(result.best_score(), 5),
                "filters_applied": result_filters(prepared),
            })
            yield frame("done", {
                "message_id": message_id, "session_id": session_id, "no_evidence": True,
                "elapsed_ms": latency, "source_count": 0,
            })
            return

        # SOURCES BEFORE THE FIRST TOKEN. The model emits [1] in its second sentence;
        # the evidence panel must already know what [1] is, or every citation pill is
        # dead text for the whole generation.
        yield frame("sources", {"citations": strip_content(prepared["citations"])})
        yield frame("status", {"stage": "generating", "sources": len(prepared["citations"])})

        answer_text = ""
        findings: list[str] = []
        final: dict[str, Any] = {}
        stream_error: str | None = None

        async for kind, data in stream_answer(
            payload.message, prepared["sources_block"], prepared["history_block"]
        ):
            # Closing the tab must stop generation, not pay for it to finish.
            if await request.is_disconnected():
                _log("chat: client disconnected; aborting generation")
                break

            if kind == "delta":
                answer_text += data
                yield frame("delta", {"text": data})
            elif kind == "finding":
                findings.append(data)
                yield frame("finding", {"index": len(findings) - 1, "text": data})
            elif kind == "no_evidence":
                # GATE 2 -- the model itself judged the sources insufficient. Reported
                # as the same event as Gate 1, so the user sees one consistent state.
                latency = int((time.perf_counter() - started) * 1000)
                message_id = await run_in_threadpool(_persist_assistant, db, session_id, {
                    "answer": data, "citations": [], "no_evidence": True, "grounded": True,
                    "model": ANTHROPIC_MODEL, "latency_ms": latency,
                })
                yield frame("no_evidence", {"detail": data, "filters_applied": result_filters(prepared)})
                yield frame("done", {"message_id": message_id, "session_id": session_id,
                                     "no_evidence": True, "elapsed_ms": latency, "source_count": 0})
                return
            elif kind == "error":
                stream_error = data.get("message")
                yield frame("error", data)
            elif kind == "done":
                final = data

            if time.monotonic() - heartbeat_at > SSE_HEARTBEAT_SECONDS:
                heartbeat_at = time.monotonic()
                yield comment()

        # GATE 3 -- strip citation indices the model invented.
        checked = validate(final.get("answer") or answer_text, prepared["citations"])
        latency = int((time.perf_counter() - started) * 1000)

        message_id = await run_in_threadpool(_persist_assistant, db, session_id, {
            "answer": checked["answer"],
            "key_findings": final.get("key_findings") or findings,
            "citations": strip_content(checked["citations"]), "grounded": checked["grounded"],
            "no_evidence": False, "model": ANTHROPIC_MODEL,
            "usage": final.get("usage"), "cost_usd": final.get("cost_usd"),
            "latency_ms": latency, "finish_reason": final.get("finish_reason"),
            "error": stream_error,
            "retrieval": {"counts": result.counts, "timings": result.timings,
                          "invalid_citations": checked["invalid"], "budget": prepared["budget"]},
        })
        await run_in_threadpool(
            _log_query, db, user["id"], session_id, message_id, payload.message,
            prepared["rewritten"], result_filters(prepared), result, True, latency
        )

        if is_new:
            # Titling never blocks the stream; a failure leaves the truncated question.
            background.add_task(_retitle, session_id, payload.message)

        # `done` carries the canonical persisted object, so the client renders from one
        # authoritative payload rather than its own reassembly of the deltas.
        yield frame("done", {
            "message_id": message_id, "session_id": session_id,
            "answer": checked["answer"],
            "key_findings": final.get("key_findings") or findings,
            "citations_used": checked["used"], "citations": strip_content(checked["citations"]),
            "grounded": checked["grounded"], "no_evidence": False,
            "elapsed_ms": latency, "source_count": len(checked["citations"]),
            "usage": {
                "input_tokens": getattr(final.get("usage"), "input_tokens", None),
                "output_tokens": getattr(final.get("usage"), "output_tokens", None),
                "cost_usd": final.get("cost_usd"),
            },
            "finish_reason": final.get("finish_reason"),
        })
    except Exception as exc:  # noqa: BLE001 -- the client is already streaming; an
                              # exception here must arrive as an event, not a 500 on a
                              # connection that has already sent a 200.
        _log(f"chat: stream failed -- {exc!r}")
        yield frame("error", {"code": "internal", "message": "The answer could not be completed."})
    finally:
        db.close()


def _retitle(session_id: str, question: str) -> None:
    from rag.generator import title_for

    db = SessionLocal()
    try:
        title = title_for(question)
        if title:
            db.execute(
                sql("UPDATE chat_sessions SET title = :t WHERE id = CAST(:i AS uuid)"),
                {"t": title[:200], "i": session_id},
            )
            db.commit()
    except Exception:  # noqa: BLE001 -- a missing title is cosmetic
        db.rollback()
    finally:
        db.close()


@router.post("/chat/{message_id}/feedback", status_code=204)
@router.post("/messages/{message_id}/feedback", status_code=204)
def feedback(
    message_id: str,
    payload: FeedbackRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(current_user),
) -> None:
    """
    `message_id` is the ASSISTANT message id.

    Issue #1 writes the path as POST /api/chat/{id}/feedback, which is ambiguous about
    whether {id} is a session or a message. It is the message -- feedback is about one
    answer. Both paths are registered so either spelling works.
    """
    owned = db.execute(
        sql(
            "SELECT 1 FROM chat_messages m JOIN chat_sessions s ON s.id = m.session_id "
            "WHERE m.id = CAST(:m AS uuid) AND s.user_id = CAST(:u AS uuid)"
        ),
        {"m": message_id, "u": user["id"]},
    ).scalar()
    if not owned:
        raise not_found("Message")

    # Upsert, so a user changing their mind updates rather than duplicating.
    db.execute(
        sql(
            "INSERT INTO message_feedback (message_id, user_id, rating, reason, comment) "
            "VALUES (CAST(:m AS uuid), CAST(:u AS uuid), :r, :why, :c) "
            "ON CONFLICT (message_id, user_id) DO UPDATE SET "
            "rating = EXCLUDED.rating, reason = EXCLUDED.reason, "
            "comment = EXCLUDED.comment, created_at = now()"
        ),
        {"m": message_id, "u": user["id"], "r": payload.rating,
         "why": payload.reason, "c": payload.comment},
    )
    db.commit()
