-- Operational tables: ingest bookkeeping, metadata edit history, the BM25 artifact.

CREATE TABLE IF NOT EXISTS ingest_runs (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    kind         text NOT NULL,           -- full | incremental | reindex | classify
    source_id    smallint REFERENCES sources(id),
    status       text NOT NULL DEFAULT 'running'
                 CHECK (status IN ('running','ok','partial','failed')),
    triggered_by text,                    -- cron | admin:<user_id> | cli
    started_at   timestamptz NOT NULL DEFAULT now(),
    finished_at  timestamptz,
    stats        jsonb,
    error        text
);
CREATE INDEX IF NOT EXISTS ingest_runs_started_idx ON ingest_runs (started_at DESC);

-- The "record" half of degrade-record-continue. A stage-boundary broad except with
-- nowhere to write is just a swallowed error.
CREATE TABLE IF NOT EXISTS ingest_errors (
    id          bigserial PRIMARY KEY,
    run_id      uuid REFERENCES ingest_runs(id) ON DELETE CASCADE,
    stage       text NOT NULL,
    source_id   smallint,
    external_id text,
    article_id  uuid,
    message     text NOT NULL,
    detail      jsonb,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ingest_errors_run_idx ON ingest_errors (run_id);

CREATE TABLE IF NOT EXISTS article_revisions (
    id         bigserial PRIMARY KEY,
    article_id uuid NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    user_id    uuid REFERENCES users(id) ON DELETE SET NULL,
    changed    jsonb NOT NULL,   -- {"title": {"before": ..., "after": ...}}
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS article_revisions_article_idx ON article_revisions (article_id, created_at DESC);

-- The BM25 artifact lives in the database, not on disk: Railway's filesystem is
-- ephemeral and replicas must agree on which index version they are serving.
-- The API loads the newest row at startup and polls MAX(version) to hot-swap, which
-- is how a re-ingest reaches a running API without a redeploy.
CREATE TABLE IF NOT EXISTS search_index_artifacts (
    id          serial PRIMARY KEY,
    kind        text NOT NULL DEFAULT 'bm25',
    version     int  NOT NULL,
    payload     bytea NOT NULL,
    chunk_count int,
    vocab_size  int,
    checksum    char(64),
    built_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (kind, version)
);

CREATE TABLE IF NOT EXISTS app_settings (
    key        text PRIMARY KEY,
    value      jsonb NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
