"""
The PIKIS API.

Sync SQLAlchemy inside async FastAPI: handlers are plain `def` (FastAPI runs them in
the threadpool) except the SSE chat handler, which is `async def` and reaches the DB
through a threadpool call. Async SQLAlchemy would double the driver surface -- scripts
have no event loop and would still need psycopg2 -- for a workload measured in single
digit requests per second. The blocking work here is the torch forward pass and the
numpy BM25, both of which release the GIL, plus the Anthropic HTTP call, which is the
one thing that genuinely wants async.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import errors
from app.config import CORS_ORIGINS, EMBEDDING_LAZY
from app.logging_utils import _log, configure_logging
from app.routers import admin, articles, auth, chat, health, search


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    configure_logging()
    _log("api: starting")

    # Starlette's default threadpool is 40 workers while the DB pool defaults to ~15.
    # A burst of 40 concurrent searches would queue 25 of them on the connection pool
    # and they would fail on pool_timeout -- a failure no log makes obvious. Pinning
    # the limiter to the pool size puts backpressure where it is visible instead.
    try:
        import anyio.to_thread

        from app.config import DB_MAX_OVERFLOW, DB_POOL_SIZE

        anyio.to_thread.current_default_thread_limiter().total_tokens = (
            DB_POOL_SIZE + DB_MAX_OVERFLOW
        )
        _log(f"api: threadpool pinned to {DB_POOL_SIZE + DB_MAX_OVERFLOW} (matches the DB pool)")
    except Exception as exc:  # noqa: BLE001 -- a tuning miss must not stop startup
        _log(f"api: could not pin the threadpool ({exc})")

    from db.engine import SessionLocal

    db = SessionLocal()
    try:
        from rag import lexical

        lexical.ensure_loaded(db, force=True)
    except Exception as exc:  # noqa: BLE001 -- serve without the keyword channel
                              # rather than refusing to boot; /health reports it.
        _log(f"api: BM25 artifact unavailable at startup ({exc})")
    finally:
        db.close()

    if not EMBEDDING_LAZY:
        try:
            from ingest.embedder import get_model

            get_model()
        except Exception as exc:  # noqa: BLE001
            _log(f"api: embedding model failed to preload ({exc})")

    _log("api: ready")
    yield
    _log("api: shutting down")


app = FastAPI(
    title="PIKIS API",
    description="Citation-backed energy research retrieval for Purnomo Yusgiantoro Center.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    # Bearer tokens, not cookies -- which sidesteps the wildcard-plus-credentials trap
    # entirely and means the browser never needs a preflight exemption.
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
    expose_headers=["X-Request-Id"],
)

# NOTE: deliberately NO GZipMiddleware. It buffers, and a compressed SSE stream
# arrives all at once at the end -- which defeats the entire streaming UI.

errors.install(app)
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(articles.router)
app.include_router(admin.router)
