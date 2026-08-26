-- Extensions PIKIS depends on.
--
-- `vector` is not optional: chunks.embedding is a vector(384) column and every
-- semantic query uses the <=> cosine operator. If this fails on a managed Postgres,
-- the platform does not ship pgvector and no amount of application code works around
-- it -- so it fails here, loudly, at migration time rather than at first query.
--
-- Verified available on the target server: SELECT * FROM pg_available_extensions
-- WHERE name='vector' -> 0.5.1 locally, ~0.8.x on Railway. Everything in this schema
-- uses exact scans, which behave identically across that skew. See 0007.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- typo-tolerant title search in the admin table
CREATE EXTENSION IF NOT EXISTS unaccent;  -- fold diacritics in title_normalized
