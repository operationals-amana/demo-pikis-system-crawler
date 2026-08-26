"""
Duplicate detection: link, never merge.

Four layers, in descending confidence. The canonical row is the earliest first_seen_at,
tie-broken by source priority (IJE > PYC) then id -- deterministic, and it is the row
people have already been looking at.

The asymmetry is deliberate and matches the sibling crawler's hard-won lesson: this
errs toward MISSING a link. A missed link shows an extra row a reader dismisses in a
second. A wrong link HIDES an article behind another one, and nobody goes looking for
what the list did not show them.

Known gap, documented rather than hidden: the PYC site is bilingual with /id/ and /en/
variants that have different titles AND different bodies. None of these layers catches
those without a shared DOI. Expect a few percent residual, backstopped by the admin's
Duplicate filter and manual linking (duplicate_reason='manual', never auto-overwritten).
"""

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.logging_utils import _log

# Layer 2: a DOI is an identity. A PYC post announcing an IJE paper carries the same
# one. Only meaningful across sources.
_DOI_SQL = """
WITH groups AS (
    SELECT lower(doi) AS key, array_agg(id ORDER BY first_seen_at, source_id, id) AS ids
    FROM articles
    WHERE doi IS NOT NULL AND processing_status <> 'archived'
    GROUP BY lower(doi) HAVING count(*) > 1
)
UPDATE articles a SET duplicate_of_id = g.ids[1], duplicate_reason = 'doi',
                      processing_status = 'duplicate'
FROM groups g
WHERE a.id = ANY(g.ids[2:]) AND a.duplicate_reason IS DISTINCT FROM 'manual'
"""

# Layer 3: identical body text. Catches WordPress cross-posts filed under two types.
_HASH_SQL = """
WITH groups AS (
    SELECT content_hash AS key, array_agg(id ORDER BY first_seen_at, source_id, id) AS ids
    FROM articles
    WHERE content_hash IS NOT NULL AND processing_status <> 'archived'
      AND duplicate_of_id IS NULL
    GROUP BY content_hash HAVING count(*) > 1
)
UPDATE articles a SET duplicate_of_id = g.ids[1], duplicate_reason = 'content_hash',
                      processing_status = 'duplicate'
FROM groups g
WHERE a.id = ANY(g.ids[2:]) AND a.duplicate_reason IS DISTINCT FROM 'manual'
"""

# Layer 4: title+year fingerprint -- the fuzzy one, so it is deliberately fenced.
# Only fires when the group spans more than one source OR more than one doc_type.
# Within a single feed a shared title fingerprint is a "Part 1"/"Part 2" series, not
# a duplicate; linking those would hide half a series.
_TITLE_SQL = """
WITH groups AS (
    SELECT dedup_key AS key, array_agg(id ORDER BY first_seen_at, source_id, id) AS ids
    FROM articles
    WHERE dedup_key IS NOT NULL AND processing_status <> 'archived'
      AND duplicate_of_id IS NULL
    GROUP BY dedup_key
    HAVING count(*) > 1
       AND (count(DISTINCT source_id) > 1 OR count(DISTINCT doc_type) > 1)
)
UPDATE articles a SET duplicate_of_id = g.ids[1], duplicate_reason = 'title_year',
                      processing_status = 'duplicate'
FROM groups g
WHERE a.id = ANY(g.ids[2:]) AND a.duplicate_reason IS DISTINCT FROM 'manual'
"""

# Full recomputation each run, so an article whose DOI was corrected stops being a
# duplicate. Manual links are preserved.
_RESET_SQL = """
UPDATE articles
SET duplicate_of_id = NULL, duplicate_reason = NULL,
    processing_status = CASE WHEN processing_status = 'duplicate' THEN 'pending'
                             ELSE processing_status END
WHERE duplicate_reason IS DISTINCT FROM 'manual' AND duplicate_of_id IS NOT NULL
"""


def relink(db: Session) -> dict[str, int]:
    db.execute(text(_RESET_SQL))
    counts: dict[str, int] = {}
    for label, sql in (
        ("doi", _DOI_SQL),
        ("content_hash", _HASH_SQL),
        ("title_year", _TITLE_SQL),
    ):
        counts[label] = db.execute(text(sql)).rowcount or 0
    db.commit()
    total = sum(counts.values())
    _log(f"dedupe: linked {total} duplicate(s) {counts}")
    return counts
