"""
Every knob this service has, resolved once at import time.

Module-level `os.getenv` with inline defaults rather than a settings class, matching
the tender-intelligence crawler. The defaults are chosen so that a bare `python -m
scripts.init_db` against a local Postgres works with no .env at all; anything whose
absence would be a security problem in production (JWT_SECRET) fails loudly instead
of defaulting.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env before any getenv below. Scripts and the API both import this module
# first, so this is the single load point.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv(name: str, default: str) -> list[str]:
    return [p.strip() for p in (os.getenv(name) or default).split(",") if p.strip()]


# --- database --------------------------------------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://pguser:pgpass123@localhost:5433/pikis"
)
DB_POOL_SIZE = _int("DB_POOL_SIZE", 5)
DB_MAX_OVERFLOW = _int("DB_MAX_OVERFLOW", 10)
DB_POOL_TIMEOUT = _int("DB_POOL_TIMEOUT", 30)

# --- app -------------------------------------------------------------------
PORT = _int("PORT", 8000)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# Exact Vercel origins. allow_credentials stays False because the browser sends a
# Bearer token, not a cookie -- which sidesteps the wildcard-plus-credentials trap.
CORS_ORIGINS = _csv("CORS_ORIGINS", "http://localhost:3000")

# --- auth ------------------------------------------------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = _int("ACCESS_TOKEN_MINUTES", 720)  # 12h, matches SESSION_TTL_HOURS
REFRESH_TOKEN_DAYS = _int("REFRESH_TOKEN_DAYS", 30)
ALLOW_SELF_REGISTER = _bool("ALLOW_SELF_REGISTER", False)

# --- harvest ---------------------------------------------------------------
IJE_OAI_URL = os.getenv("IJE_OAI_URL", "https://ije-pyc.org/IJE/oai")
IJE_OAI_SET = os.getenv("IJE_OAI_SET", "IJE")
PYC_WP_URL = os.getenv(
    "PYC_WP_URL", "https://purnomoyusgiantorocenter.org/wp-json/wp/v2"
)
# The ten public post types, verified against X-WP-Total. 'pages' is excluded: it is
# site chrome (About, Contact), not publications.
PYC_POST_TYPES = _csv(
    "PYC_POST_TYPES",
    "posts,media-coverage,opinion,archive,podcast,e-newsletter,book,"
    "brief-report,works-archive,infographics",
)
HTTP_TIMEOUT = _float("HTTP_TIMEOUT", 60.0)
HTTP_RETRIES = _int("HTTP_RETRIES", 3)
USER_AGENT = os.getenv("USER_AGENT", "PIKIS/1.0 (+https://amana.id)")
INGEST_STALE_MINUTES = _int("INGEST_STALE_MINUTES", 60)

# --- embeddings ------------------------------------------------------------
# e5-small (384-dim) over e5-base (768) because query embedding sits in the SSE
# latency path: ~55ms for base on a throttled shared CPU is 120-200ms of dead air
# before retrieval starts, every turn. Swapping is one ALTER TABLE + scripts/reembed.py.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBEDDING_DIM = _int("EMBEDDING_DIM", 384)
EMBEDDING_BATCH = _int("EMBEDDING_BATCH", 32)
# Set BEFORE torch is imported (ingest/embedder.py re-exports these into os.environ).
# On a shared-CPU container torch grabs every visible core, all throttled, and the
# oversubscription makes it measurably slower.
EMBED_THREADS = _int("EMBED_THREADS", 2)
EMBEDDING_LAZY = _bool("EMBEDDING_LAZY", False)

# --- chunking --------------------------------------------------------------
# 450 not 512: e5 truncates at 512 SILENTLY, and a truncated chunk still shows its
# full text in chunks.content, so retrieval and display would disagree. 450 leaves
# headroom for the "passage: {title}\n\n" prefix.
CHUNK_TARGET_TOKENS = _int("CHUNK_TARGET_TOKENS", 450)
CHUNK_OVERLAP_TOKENS = _int("CHUNK_OVERLAP_TOKENS", 80)
CHUNK_MIN_TOKENS = _int("CHUNK_MIN_TOKENS", 24)

# --- retrieval -------------------------------------------------------------
RETRIEVAL_CANDIDATES = _int("RETRIEVAL_CANDIDATES", 60)   # per channel, before fusion
RRF_K = _int("RRF_K", 60)
W_SEMANTIC = _float("W_SEMANTIC", 1.0)
W_LEXICAL = _float("W_LEXICAL", 1.0)
BM25_K1 = _float("BM25_K1", 1.2)
BM25_B = _float("BM25_B", 0.75)
TOP_K_CONTEXT = _int("TOP_K_CONTEXT", 8)
MAX_CHUNKS_PER_ARTICLE = _int("MAX_CHUNKS_PER_ARTICLE", 2)
INDEX_POLL_SECONDS = _int("INDEX_POLL_SECONDS", 60)
RERANK_MODE = os.getenv("RERANK_MODE", "lexical")  # lexical | llm

# The pre-LLM no-evidence gate. This, not the prompt, is what actually guarantees
# "sistem tidak membuat jawaban substantif ketika evidence tidak ditemukan".
MIN_EVIDENCE_CHUNKS = _int("MIN_EVIDENCE_CHUNKS", 2)
MIN_EVIDENCE_SCORE = _float("MIN_EVIDENCE_SCORE", 0.018)
# Gate 1 must key on an ABSOLUTE similarity, not the RRF score.
#
# RRF is rank-based: the top chunk scores ~1/(60+1) per channel whether it is a
# perfect match or the least-bad of 4,120 irrelevant ones. So an RRF threshold cannot
# express "nothing here is relevant" -- measured, an off-corpus question about EUV
# photolithography produced the same ~0.038 top score as a genuine energy question.
#
# Raw cosine does separate them, but only just, because e5 compresses similarities
# into a narrow high band:
#     on-corpus  0.859 - 0.905   (lowest: "coal phase out policy" at 0.8589)
#     off-corpus 0.769 - 0.844   (highest: EUV photolithography at 0.8435)
#
# The floor is therefore set CONSERVATIVELY at 0.82. It catches the clearly-unrelated
# questions without risking a legitimate one being refused -- rejecting a real question
# is a far worse failure than falling through to Gate 2, which demonstrably works (the
# EUV case lands in the 0.82-0.86 band and the model correctly answers <no_evidence>).
# This is defence in depth behaving as designed, not a gap.
MIN_SEMANTIC_SIMILARITY = _float("MIN_SEMANTIC_SIMILARITY", 0.82)

ENABLE_QUERY_TRANSLATION = _bool("ENABLE_QUERY_TRANSLATION", True)
# Translation is DIRECTIONAL, and this is measured rather than assumed.
#
# The corpus is lopsided: its substantive half is the 93 English IJE papers, while its
# Indonesian half is mostly event notices and media links. So:
#   id -> en  is essential. Golden-set recall@5 for Indonesian questions goes from
#             0.00 to 0.88, because an Indonesian query otherwise cannot reach an
#             English paper through either channel.
#   en -> id  actively HURTS. "renewable energy financing barriers" returns four
#             on-point financing papers untranslated, and generic Indonesian event
#             posts once the Indonesian phrasing is merged in -- the translated query
#             matches the weak half of the corpus loosely and wins on merge-by-max.
# Hence: translate only queries whose language is listed here.
TRANSLATE_FROM_LANGUAGES = _csv("TRANSLATE_FROM_LANGUAGES", "id")

# --- llm -------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 1600)
LLM_CONCURRENCY = _int("LLM_CONCURRENCY", 4)
CHAT_TIMEOUT_SECONDS = _int("CHAT_TIMEOUT_SECONDS", 90)
CHAT_HISTORY_TURNS = _int("CHAT_HISTORY_TURNS", 4)
CONTEXT_TOKEN_BUDGET = _int("CONTEXT_TOKEN_BUDGET", 12000)
SSE_HEARTBEAT_SECONDS = _int("SSE_HEARTBEAT_SECONDS", 15)

# Haiku 4.5 list rates, USD per million tokens. Used for the admin cost rollup only.
PRICE_IN = _float("PRICE_IN", 1.00)
PRICE_CACHE_WRITE = _float("PRICE_CACHE_WRITE", 1.25)
PRICE_CACHE_READ = _float("PRICE_CACHE_READ", 0.10)
PRICE_OUT = _float("PRICE_OUT", 5.00)

# --- crawl schedule --------------------------------------------------------
# The worker role (docker-entrypoint.sh worker) crawls once a day at this local
# hour. Railway deployments should prefer the platform cron on the ingest service
# ('0 18 * * *' UTC = 01:00 WIB) -- the worker loop exists for platforms without
# a scheduler, and for docker-compose.
CRAWL_TIMEZONE = os.getenv("CRAWL_TIMEZONE", "Asia/Jakarta")
CRAWL_HOUR_LOCAL = _int("CRAWL_HOUR_LOCAL", 1)   # 1 => 01:00 local time
