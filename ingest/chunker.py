"""
Token-aware chunking with overlap.

Target 450 tokens, NOT 512, and this is load-bearing: e5 models truncate at 512
tokens SILENTLY. A 520-token chunk loses its tail with no warning, while
chunks.content still holds the full text -- so retrieval and display would disagree,
and the disagreement would only ever show up as "the citation snippet mentions
something the answer didn't use". 450 leaves headroom for the "passage: {title}\\n\\n"
prefix that embedder.py prepends.

Token counting uses the model's real tokenizer, not len(text)//4. Indonesian
tokenises roughly 20% worse than English under a multilingual vocab, so a character
heuristic would systematically over-fill exactly the Indonesian chunks.

Split hierarchy: paragraphs -> sentences -> hard character split. A chunk boundary
mid-sentence costs retrieval quality, because the embedding then represents half a
thought.
"""

import hashlib
import re
from typing import Any, Iterator

from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.config import CHUNK_MIN_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNK_TARGET_TOKENS
from app.logging_utils import _log
from ingest.language import detect as detect_language
from ingest.store import _record_error

# Sentence boundary. Python's re only supports FIXED-width lookbehind, so the
# abbreviation guard cannot live in the pattern (the alternatives differ in length).
# Instead: split on every candidate boundary, then re-join any split whose left side
# ends in a known abbreviation. Same result, and the abbreviation list stays readable.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9À-ɏ])")
_ABBREVIATIONS = {
    "fig", "figs", "tab", "no", "vol", "eq", "ref", "refs", "etc", "al", "dr",
    "prof", "mr", "mrs", "ms", "sr", "jr", "vs", "hlm", "dkk", "approx", "est",
    "i.e", "e.g", "cf", "pp", "ed", "eds", "inc", "ltd", "co", "st",
}
_TRAILING_WORD = re.compile(r"([A-Za-z.]+)\.$")
_PARAGRAPH = re.compile(r"\n\s*\n")

# Headings we can recognise, so a chunk can say which section it came from. Purely
# additive: an unrecognised document simply has section = NULL.
_SECTION = re.compile(
    r"^\s*(?:\d+\.?\s*)?(ABSTRACT|ABSTRAK|INTRODUCTION|PENDAHULUAN|BACKGROUND|"
    r"LITERATURE\s+REVIEW|TINJAUAN\s+PUSTAKA|METHOD(?:OLOGY|S)?|METODE|METODOLOGI|"
    r"RESULTS?(?:\s+AND\s+DISCUSSIONS?)?|HASIL(?:\s+DAN\s+PEMBAHASAN)?|DISCUSSIONS?|"
    r"PEMBAHASAN|CONCLUSIONS?|KESIMPULAN|RECOMMENDATIONS?|SARAN|"
    r"ACKNOWLEDGE?MENTS?|UCAPAN\s+TERIMA\s+KASIH)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# e5 truncates above this SILENTLY. Nothing may be emitted above it.
MODEL_MAX_TOKENS = 512

_tokenizer = None


def _tok():
    """The real model tokenizer, loaded once. Falls back to a word count if the model
    is unavailable, which keeps chunking usable offline at slightly worse accuracy."""
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer

            from app.config import EMBEDDING_MODEL

            _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        except Exception:  # noqa: BLE001 -- degrade rather than block ingestion
            _log("chunker: tokenizer unavailable; falling back to word-count estimate")
            _tokenizer = False
    return _tokenizer


def count_tokens(text_body: str) -> int:
    tk = _tok()
    if tk is False:
        # ~1.4 tokens per whitespace word is the observed multilingual-e5 ratio; it
        # over-estimates slightly, which is the safe direction against a 512 ceiling.
        return int(len(text_body.split()) * 1.4) + 1
    return len(tk.encode(text_body, add_special_tokens=False))


def _ends_in_abbreviation(fragment: str) -> bool:
    match = _TRAILING_WORD.search(fragment.strip())
    if not match:
        return False
    word = match.group(1).rstrip(".").lower()
    # A single capital ("J.") is an initial in an author list, not a sentence end.
    return word in _ABBREVIATIONS or len(word) == 1


def _split_sentences(paragraph: str) -> list[str]:
    raw = [p.strip() for p in _SENTENCE.split(paragraph) if p.strip()]
    if not raw:
        return [paragraph.strip()] if paragraph.strip() else []
    merged: list[str] = []
    for part in raw:
        if merged and _ends_in_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def _hard_split(unit: str, budget: int) -> list[str]:
    """Last resort for a single sentence longer than the whole budget."""
    words, out, current = unit.split(), [], []
    for word in words:
        current.append(word)
        if count_tokens(" ".join(current)) >= budget:
            out.append(" ".join(current))
            current = []
    if current:
        out.append(" ".join(current))
    return out or [unit]


def _sections(body: str) -> list[tuple[int, str]]:
    return [(m.start(), m.group(1).title()) for m in _SECTION.finditer(body)]


def _section_at(sections: list[tuple[int, str]], offset: int) -> str | None:
    name = None
    for start, label in sections:
        if start <= offset:
            name = label
        else:
            break
    return name


def chunk_text(body: str, target: int | None = None, overlap: int | None = None) -> Iterator[dict[str, Any]]:
    """
    Yield {content, char_start, char_end, token_count, section} in document order.

    char offsets index into `body` exactly, so a citation can be highlighted in the
    original text without a second retrieval.
    """
    target = target or CHUNK_TARGET_TOKENS
    overlap = overlap or CHUNK_OVERLAP_TOKENS
    if not body or not body.strip():
        return

    sections = _sections(body)

    # (text, char_start) for every atomic unit, in order.
    units: list[tuple[str, int]] = []
    cursor = 0
    for para in _PARAGRAPH.split(body):
        if not para.strip():
            cursor += len(para) + 2
            continue
        para_start = body.find(para, cursor)
        if para_start < 0:
            para_start = cursor
        inner = para_start
        for sentence in _split_sentences(para):
            found = body.find(sentence, inner)
            start = found if found >= 0 else inner
            if count_tokens(sentence) > target:
                piece_at = start
                for piece in _hard_split(sentence, target):
                    found_piece = body.find(piece, piece_at)
                    p_start = found_piece if found_piece >= 0 else piece_at
                    units.append((piece, p_start))
                    piece_at = p_start + len(piece)
            else:
                units.append((sentence, start))
            inner = start + len(sentence)
        cursor = para_start + len(para)

    if not units:
        return

    buffer: list[tuple[str, int]] = []
    buffer_tokens = 0

    def emit(buf: list[tuple[str, int]]) -> dict[str, Any] | None:
        if not buf:
            return None
        content = " ".join(t for t, _ in buf).strip()
        if not content:
            return None
        start = buf[0][1]
        end = buf[-1][1] + len(buf[-1][0])
        tokens = count_tokens(content)
        if tokens > MODEL_MAX_TOKENS:
            # Should be unreachable given the carry budget above, but a silent
            # truncation is unacceptable enough to warrant a hard backstop: trim
            # whole words off the tail until the real tokenizer agrees we fit.
            words = content.split()
            while words and count_tokens(" ".join(words)) > MODEL_MAX_TOKENS:
                words.pop()
            content = " ".join(words)
            end = start + len(content)
            tokens = count_tokens(content)
        return {
            "content": content,
            "char_start": start,
            "char_end": end,
            "token_count": tokens,
            "section": _section_at(sections, start),
        }

    for unit_text, unit_start in units:
        unit_tokens = count_tokens(unit_text)
        if buffer and buffer_tokens + unit_tokens > target:
            chunk = emit(buffer)
            if chunk:
                yield chunk
            # Carry the tail back as overlap so a fact spanning a boundary survives
            # in at least one chunk whole.
            #
            # The carry must leave room for the unit that is about to be appended,
            # or the next chunk lands at overlap + target tokens. With the defaults
            # that is 80 + 450 = 530, which silently overflows e5's 512 ceiling --
            # observed as 7 chunks at up to 529 tokens before this guard existed.
            # The whole point of targeting 450 is to never reach 512, so the carry
            # budget is whatever is left after reserving space for `unit_tokens`.
            carry_budget = max(0, min(overlap, target - unit_tokens))
            carry: list[tuple[str, int]] = []
            carried = 0
            for prev_text, prev_start in reversed(buffer):
                prev_tokens = count_tokens(prev_text)
                if carried + prev_tokens > carry_budget:
                    break
                carry.insert(0, (prev_text, prev_start))
                carried += prev_tokens
            buffer = carry
            buffer_tokens = carried
        buffer.append((unit_text, unit_start))
        buffer_tokens += unit_tokens

    final = emit(buffer)
    # Drop a trailing sliver, but never drop the ONLY chunk -- a 20-token
    # media-coverage excerpt is still the whole document.
    if final and (final["token_count"] >= CHUNK_MIN_TOKENS or not units[:-1]):
        yield final


_PENDING_ARTICLES = """
SELECT a.id::text, a.title, a.full_text, a.abstract, a.language, a.content_hash
FROM articles a
-- 'processing' MUST be included: the chunker itself sets that status, so leaving it
-- out makes a re-chunk silently match zero articles the second time it runs.
WHERE a.processing_status IN ('pending', 'processing', 'ready')
  AND a.duplicate_of_id IS NULL
  AND (
        a.chunk_count = 0
     OR NOT EXISTS (SELECT 1 FROM chunks c WHERE c.article_id = a.id)
     -- The change signal is the CONTENT fingerprint, not updated_at. updated_at is
     -- bumped by every status write (dedupe, embed rollups), which made a no-change
     -- daily run re-chunk -- and therefore re-embed -- the entire corpus. See
     -- migration 0008 for the incident note.
     OR a.content_hash IS DISTINCT FROM a.chunked_content_hash
  )
ORDER BY a.published_at DESC NULLS LAST
"""


def chunk_pending(db: Session, run_id: str | None = None, limit: int | None = None) -> dict[str, int]:
    rows = db.execute(sql(_PENDING_ARTICLES)).all()
    if limit:
        rows = rows[:limit]
    stats = {"articles": 0, "chunks": 0, "skipped_empty": 0, "failed": 0}
    if not rows:
        _log("chunker: nothing to chunk")
        return stats

    _log(f"chunker: {len(rows)} article(s) to chunk")
    for i, (article_id, title, full_text, abstract, language, _hash) in enumerate(rows, 1):
        body = (full_text or abstract or "").strip()
        try:
            if not body:
                # Title-only documents (podcasts, infographics). Still worth one chunk:
                # a title is genuinely searchable, and excluding them would make the
                # corpus silently smaller than the admin dashboard claims.
                body = (title or "").strip()
                if not body:
                    stats["skipped_empty"] += 1
                    continue

            chunks = list(chunk_text(body))
            if not chunks:
                stats["skipped_empty"] += 1
                continue

            # Delete + insert inside ONE transaction, so MVCC readers keep seeing the
            # old chunks until commit. Without this there is a window where an article
            # has zero chunks and simply vanishes from search.
            db.execute(sql("DELETE FROM chunks WHERE article_id = CAST(:id AS uuid)"), {"id": article_id})
            for index, chunk in enumerate(chunks):
                # Re-detect per chunk: an Indonesian paper's English abstract has to be
                # lexed under 'english', or its stems are wrong.
                guess, _conf = detect_language(chunk["content"])
                chunk_language = guess or language or "en"
                cfg = "indonesian" if chunk_language == "id" else "english"
                db.execute(
                    sql(
                        "INSERT INTO chunks (article_id, chunk_index, content, char_start, "
                        "char_end, section, token_count, language, lexemes, content_hash) "
                        "VALUES (CAST(:aid AS uuid), :idx, :content, :cs, :ce, :section, "
                        ":tokens, :lang, tsvector_to_array(to_tsvector(CAST(:cfg AS regconfig), :content)), :chash)"
                    ),
                    {
                        "aid": article_id, "idx": index, "content": chunk["content"],
                        "cs": chunk["char_start"], "ce": chunk["char_end"],
                        "section": chunk["section"], "tokens": chunk["token_count"],
                        "lang": chunk_language, "cfg": cfg,
                        "chash": hashlib.sha256(chunk["content"].encode("utf-8")).hexdigest(),
                    },
                )
            db.execute(
                sql(
                    "UPDATE articles SET chunk_count = :n, "
                    # Record what we chunked, so the next run can tell "changed" from
                    # "merely touched". _hash is the content_hash read in the SELECT.
                    "chunked_content_hash = :chash, processing_status = "
                    "CASE WHEN processing_status = 'pending' THEN 'processing' ELSE processing_status END "
                    "WHERE id = CAST(:id AS uuid)"
                ),
                {"n": len(chunks), "chash": _hash, "id": article_id},
            )
            db.commit()
            stats["articles"] += 1
            stats["chunks"] += len(chunks)
            if i % 200 == 0:
                _log(f"chunker: {i}/{len(rows)} articles, {stats['chunks']} chunks")
        except Exception as exc:  # noqa: BLE001 -- per-article; one bad body must not
                                  # cost the corpus.
            db.rollback()
            stats["failed"] += 1
            _record_error(db, run_id, "chunk", None, article_id, str(exc))

    _log(f"chunker: {stats}")
    return stats
