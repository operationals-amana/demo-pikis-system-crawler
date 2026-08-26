"""
The single SQLAlchemy engine and session factory.

Deliberately does NOT call Base.metadata.create_all(). The SQL files in
db/migrations are the schema's only source of truth; db/models.py is a read/write
mapping over them. Two tools creating the same tables is how you end up with a
database neither of them can migrate -- the same reasoning the tender-intelligence
crawler documents, except here this service owns the migrations rather than deferring
to Drizzle.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import DB_MAX_OVERFLOW, DB_POOL_SIZE, DB_POOL_TIMEOUT, DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,       # Railway drops idle connections; without this the first
                             # query after an idle period raises instead of reconnecting
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
    pool_timeout=DB_POOL_TIMEOUT,
    future=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)

Base = declarative_base()


def check_connection() -> bool:
    """Liveness only -- deliberately not a schema check, so /health stays fast."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001 -- health probes must report, never raise
        return False
