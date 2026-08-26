"""
Token budgeting for the <sources> block.

Drops the lowest-ranked sources until the block fits, then RENUMBERS the survivors
contiguously. The renumbering matters: a gap in the index sequence ([1] [2] [4])
measurably confuses the model into citing the missing number.

Budget is estimated with the embedding model's tokenizer, which is not Claude's -- but
it is consistently within ~10% on this corpus and costs nothing, whereas a
count_tokens round trip per request would add latency to every single answer.
"""

from typing import Any

from app.config import CONTEXT_TOKEN_BUDGET
from rag.prompts import format_sources


def _estimate_tokens(text: str) -> int:
    from ingest.chunker import count_tokens

    return count_tokens(text)


def fit(
    citations: list[dict[str, Any]], budget: int = CONTEXT_TOKEN_BUDGET
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """
    Return (kept_citations_renumbered, sources_block, info).

    `info` records what was dropped so it can land in chat_messages.retrieval -- a
    silently truncated context is the kind of thing that makes an answer look
    inexplicably incomplete weeks later.
    """
    if not citations:
        return [], "", {"dropped": 0, "tokens": 0}

    kept = list(citations)
    dropped: list[int] = []

    while kept:
        block = format_sources(kept)
        tokens = _estimate_tokens(block)
        if tokens <= budget or len(kept) == 1:
            renumbered = []
            for new_index, c in enumerate(kept, start=1):
                copy = dict(c)
                copy["original_index"] = c["index"]
                copy["index"] = new_index
                renumbered.append(copy)
            block = format_sources(renumbered)
            return renumbered, block, {
                "dropped": len(dropped),
                "dropped_indices": dropped,
                "tokens": tokens,
                "kept": len(renumbered),
            }
        dropped.append(kept[-1]["index"])
        kept = kept[:-1]

    return [], "", {"dropped": len(dropped), "dropped_indices": dropped, "tokens": 0}
