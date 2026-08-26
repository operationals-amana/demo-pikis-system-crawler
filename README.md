# PIKIS backend

Crawler, retrieval engine, and the entire HTTP API for PIKIS — the citation-backed
energy research assistant for Purnomo Yusgiantoro Center. Deployed to Railway; the
Next.js frontend (Vercel) is a pure client and holds no database access.

The directory is named `-crawler` for historical reasons. It is the whole backend.

## What it does

`Scrapy spiders (IJE over OAI-PMH, PYC over WordPress REST)` → normalize → dedupe →
PDF full text → chunk → embed → hybrid retrieval → grounded answer with verifiable
citations.

## The crawler

Two Scrapy spiders (`crawler/spiders/`), sharing the sibling tender-intelligence
crawler's operational shape:

- **Every crawl cycle is a subprocess** (`scripts/run_crawl.py`), never in-process:
  Twisted's reactor cannot be restarted within one process, and the subprocess
  boundary is also what lets the admin "ingest now" endpoint work from inside the
  long-lived API.
- Spider outcomes are captured via the `spider_closed` signal — `process.start()`
  returns normally even when a spider aborted, so `reason != "finished"` is the only
  reliable failure signal. (Handlers are kept in a strong-reference list: Scrapy's
  SignalManager holds receivers weakly, and a loop-defined closure otherwise gets
  garbage-collected before its signal fires.)
- Both `async def start()` and `def start_requests()` are defined, for Scrapy 2.13's
  silent replacement of the latter.
- Parsing hands off to `ingest/normalize.py`, and persistence reuses
  `ingest/store.upsert_articles` (batch-then-row-by-row, `metadata_locked_fields`
  honoured) — the Scrapy layer owns fetching and nothing else.

### Schedule — daily at 01:00 WIB

- **Railway (recommended):** the `railway.ingest.json` service with a platform Cron
  Schedule of `0 18 * * *` (18:00 UTC = 01:00 WIB). Each firing runs
  `docker-entrypoint.sh ingest`: an **incremental** cycle (OAI `from=` + WordPress
  `modified_after=`, scoped to the last successful run minus a 2-day overlap),
  re-chunking and re-embedding only what changed, then bumping the BM25 index
  version, which the running API hot-swaps within `INDEX_POLL_SECONDS`. A no-change
  day completes in about a minute.
- **No cron available:** `docker-entrypoint.sh worker` runs the same cycle in a
  sleep-until-01:00 loop (`CRAWL_HOUR_LOCAL` / `CRAWL_TIMEZONE`).
- **From the dashboard:** `POST /api/admin/ingest/run` starts a cycle in the
  background and answers 409 with the live run's details while one is going. Stale
  `running` rows left by a killed process are reclaimed automatically.

## Corpus, as actually harvested

| | |
|---|---|
| Articles | **1,420** (93 IJE journal articles + 1,327 PYC documents) |
| IJE full text | 92 of 93 extracted from galley PDFs; 1 upstream 404, visible as `failed` |
| Chunks | 4,120, max 453 tokens (e5 truncates silently at 512) |
| Languages | 937 English / 483 Indonesian |
| Duplicates linked | 43 (38 content-hash cross-posts, 5 title+year) |

Two IJE records are `status="deleted"` upstream (withdrawn), so 93 is complete, not short.

## Retrieval quality

`python -m scripts.eval_retrieval` is the gate — 20 hand-written question/article
pairs, half Indonesian, half English.

| channel | recall@5 | recall@10 | MRR |
|---|---|---|---|
| lexical (BM25) | 0.50 | 0.55 | 0.461 |
| semantic (pgvector) | 0.70 | 0.80 | 0.715 |
| **fused (RRF + rerank)** | **0.95** | **0.95** | **0.950** |

English 1.00, Indonesian 0.88. Cross-lingual only works because of directional query
translation — without it Indonesian recall@5 is **0.00**, since 93 of 95 journal
articles are English and the two languages share no stems.

## Grounding — three gates

1. **Pre-LLM score gate** (`rag/retriever.py`) — the answer model is never called when
   the best cosine falls below `MIN_SEMANTIC_SIMILARITY`. Deterministic; this is what
   actually satisfies "no substantive answer without evidence".
2. **Prompt gate** — `<no_evidence>` for when the retrieved chunks score well but are
   about the wrong aspect.
3. **Citation validation** (`rag/citations.py`) — indices the model invented are
   stripped, and only sources the answer actually cited reach the evidence panel.

## Run it

```bash
source ~/.pyenv/versions/pikis/bin/activate
pip install --no-cache-dir -r requirements.txt      # torch comes from the CPU index
cp .env.example .env                                 # set JWT_SECRET + ANTHROPIC_API_KEY

python -m scripts.init_db                            # creates the DB, extension, schema, seed
python -m scripts.create_admin --email you@example.com --password '...'

python -m scripts.run_ingest --skip-pdf --skip-embed # ~40s, Scrapy crawl + metadata only
python -m scripts.run_ingest                         # full: PDFs, chunks, embeddings, index
python -m scripts.run_ingest --incremental           # what the 01:00 cron runs
python -m scripts.run_crawl --source ije             # one spider, crawl only
python -m scripts.eval_retrieval                     # THE gate; must report >= 0.80

uvicorn app.main:app --reload --port 8000 --timeout-keep-alive 75
```

Verify streaming by hand — `curl -N` is the only way to see buffering before the proxy does:

```bash
curl -N -X POST localhost:8000/api/chat -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"message":"Apa tantangan utama transisi energi Indonesia?","stream":true}'
```

## API

```
GET    /health                       GET  /api/meta
POST   /api/auth/{register,login,refresh,logout}    GET /api/auth/me
POST   /api/search
POST   /api/chat                     # SSE: meta, status, sources, delta, finding, done
GET    /api/sessions                 POST /api/sessions
GET    /api/sessions/{id}            PATCH /api/sessions/{id}   DELETE /api/sessions/{id}
POST   /api/chat/{message_id}/feedback
```

`sources` always precedes the first `delta`, so every `[n]` pill is resolvable the
moment it appears. Citation indices restart at 1 **per message** — scope lookups by
`message_id`.

## Known limits, stated plainly

- **The 1,000-article criterion is met by count, not uniformly by substance.** Roughly
  500 PYC documents are media links, podcasts and infographics with no body; 175 are
  title-only. The reranker's document-type prior down-weights them and `doc_type` is
  exposed as a filter, but this should be said to PYC rather than papered over.
- **Session delete is soft.** A hard delete would destroy the audit log the brief
  requires. A product decision to confirm.
- **No ANN index on `chunks.embedding`, deliberately.** See `db/migrations/0007_hnsw.sql.optional`.
- **Cross-lingual depends on one extra LLM call.** If it fails, retrieval degrades to
  the untranslated query rather than erroring.
