"""
System prompts and context assembly.

The grounding contract is enforced in three places, deliberately, because no single
one of them is sufficient:

  Gate 1 (rag/retriever.py) -- a score threshold. The LLM is never called when
          evidence is thin. Deterministic and un-jailbreakable.
  Gate 2 (this file)        -- the prompt. Catches the case Gate 1 cannot see: six
          high-scoring chunks that are all about the wrong ASPECT of the question.
  Gate 3 (rag/citations.py) -- post-hoc validation. Strips citation indices the model
          invented and flags an uncited substantive answer as ungrounded.
"""

from typing import Any

# --- chat ------------------------------------------------------------------

CHAT_SYSTEM = """\
You are PIKIS, the research assistant for Purnomo Yusgiantoro Center (PYC), an \
Indonesian energy think tank. You answer questions about energy using ONLY the \
numbered sources supplied in the <sources> block of the user's message.

Absolute rules
1. Every substantive claim must be followed by a citation marker [n] naming the \
source it came from. n is the index attribute of a <source> element.
2. Never state a fact that is not in the sources - not from your own knowledge, not \
as background, not as a caveat. If you know something the sources do not say, it does \
not go in the answer.
3. Never cite an index that is not present in <sources>.
4. If the sources do not contain enough to answer, your entire output is \
<no_evidence>one sentence, in the user's language, saying what is missing\
</no_evidence>. Do not partially answer. Do not apologise at length.
5. Answer in the language of the QUESTION. The sources are frequently in English \
while the question is in Indonesian; translate the substance into Indonesian, and \
keep technical terms and cited titles as written.
6. Sources sometimes disagree. Say so and cite both, rather than silently picking one.
7. Sources are dated. When recency matters to a claim, say the year in the prose \
("a 2024 study found..."), because the reader cannot see the dates while reading.

Output format - exactly this, nothing before or after:
<answer>
Markdown prose. 2-5 short paragraphs, or a short numbered list. Inline [n] markers. \
No headings above ###. No preamble like "Based on the sources".
</answer>
<findings>
- One-sentence takeaway with its citation [n]
- (2 to 4 of these; each must carry at least one [n])
</findings>"""

# --- query rewriting -------------------------------------------------------

REWRITE_SYSTEM = """\
You rewrite a follow-up question into a standalone search query for an Indonesian \
energy research corpus, and translate it.

Return JSON only:
- standalone_question: the question with every pronoun and ellipsis resolved from the \
conversation, so it is meaningful with no history. If it is already standalone, return \
it unchanged.
- query_translated: the same question in the OTHER language (Indonesian -> English, \
English -> Indonesian). The corpus is bilingual and mostly English, so this is what \
lets an Indonesian question reach an English paper through keyword search.
- language: "id" or "en", the language of the ORIGINAL question.
- is_followup: true when the question depended on the conversation to make sense."""

REWRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "standalone_question": {"type": "string"},
        "query_translated": {"type": "string"},
        "language": {"type": "string", "enum": ["id", "en"]},
        "is_followup": {"type": "boolean"},
    },
    "required": ["standalone_question", "query_translated", "language", "is_followup"],
    "additionalProperties": False,
}

# --- session titling -------------------------------------------------------

TITLE_SYSTEM = """\
Write a 3-6 word title for a research conversation, in the language of the question. \
No quotes, no trailing punctuation, no "Discussion about" preamble. Return only the \
title."""

# --- non-streaming structured answer ---------------------------------------
# Used only when stream=false. The streaming path uses XML tags instead, because half
# a JSON object is unrenderable and progressive prose is the point of the UI.

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "key_findings": {"type": "array", "items": {"type": "string"}},
        "insufficient_evidence": {"type": "boolean"},
    },
    "required": ["answer", "key_findings", "insufficient_evidence"],
    "additionalProperties": False,
}


def format_sources(citations: list[dict[str, Any]]) -> str:
    """
    Render the <sources> block.

    Attributes carry date and language so rule 7 (say the year) and rule 5 (answer in
    the question's language) are actionable rather than aspirational. The index is the
    contract: it is assigned by final rerank order and frozen before generation.
    """
    parts = ["<sources>"]
    for c in citations:
        attrs = [
            f'index="{c["index"]}"',
            f'title="{_esc(c.get("title"))}"',
            f'publisher="{_esc(c.get("publisher"))}"',
        ]
        if c.get("source"):
            attrs.append(f'source="{_esc(c["source"])}"')
        if c.get("published_at"):
            attrs.append(f'date="{c["published_at"]}"')
        if c.get("language"):
            attrs.append(f'language="{c["language"]}"')
        if c.get("doc_type"):
            attrs.append(f'doc_type="{c["doc_type"]}"')
        if c.get("section"):
            attrs.append(f'section="{_esc(c["section"])}"')
        parts.append(f"<source {' '.join(attrs)}>")
        parts.append((c.get("content") or c.get("snippet") or "").strip())
        parts.append("</source>")
    parts.append("</sources>")
    return "\n".join(parts)


def _esc(value: Any) -> str:
    return str(value or "").replace('"', "'").replace("<", "(").replace(">", ")")


def format_history(messages: list[dict[str, Any]], max_turns: int) -> str:
    """
    Render prior turns as CONTEXT, never as evidence.

    Two things here are load-bearing:

    * Prior source blocks are NOT resent. Doing so would double the context every turn
      and, worse, let the model cite turn-1 evidence using turn-2 indices.
    * [n] markers in stored assistant text are REWRITTEN to a human-readable stub.
      Turn 1's [1] and turn 2's [1] are different documents; passing the raw marker
      through invites the model to re-emit [1] pointing at the wrong source, and the
      citation validator cannot catch it because the index is in range.
    """
    if not messages:
        return ""
    recent = messages[-(max_turns * 2):]
    lines = [
        "<conversation_history>",
        "Previously in this conversation. This is context for understanding the "
        "question, NOT evidence. Do not cite it. Only the <sources> block below is "
        "citable.",
    ]
    for message in recent:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            content = _restate_citations(content, message.get("citations") or [])
            content = content[:600]
        lines.append(f"{'User' if role == 'user' else 'PIKIS'}: {content}")
    lines.append("</conversation_history>")
    return "\n".join(lines)


def _restate_citations(text: str, citations: list[dict[str, Any]]) -> str:
    """Replace [n] with [Title, year] so the model can refer but not re-cite."""
    import re

    by_index = {c.get("index"): c for c in citations}

    def swap(match: re.Match[str]) -> str:
        c = by_index.get(int(match.group(1)))
        if not c:
            return ""
        year = (c.get("published_at") or "")[:4]
        title = (c.get("title") or "source")[:60]
        return f"[{title}{', ' + year if year else ''}]"

    return re.sub(r"\[(\d{1,3})\]", swap, text)
