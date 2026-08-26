-- Chat sessions, messages, feedback, and the query audit log.

CREATE TABLE IF NOT EXISTS chat_sessions (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title        text,
    title_source text NOT NULL DEFAULT 'auto' CHECK (title_source IN ('auto','manual')),
    filters      jsonb,          -- sticky per-session filter state
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now(),
    last_message_at timestamptz,
    message_count int NOT NULL DEFAULT 0,
    -- SOFT delete. A hard delete would destroy the audit log the issue requires;
    -- this is a product decision to confirm with PYC, not a technical detail.
    deleted_at   timestamptz
);
CREATE INDEX IF NOT EXISTS chat_sessions_user_idx
    ON chat_sessions (user_id, last_message_at DESC) WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS chat_messages (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id uuid NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    seq        int  NOT NULL,
    UNIQUE (session_id, seq),
    role    text NOT NULL CHECK (role IN ('user','assistant')),
    content text NOT NULL DEFAULT '',

    key_findings jsonb,
    -- A FROZEN snapshot of what the user was actually shown: title, publisher, date,
    -- url and snippet stored inline. Deliberately JSONB rather than a relational
    -- message_citations table -- a later re-chunk or an admin metadata edit must not
    -- be able to silently rewrite a historical citation. article_id/chunk_id ride
    -- along as soft references with NO foreign key, so archiving an article does not
    -- erase the audit trail.
    citations jsonb,
    retrieval jsonb,   -- rewritten query, filters, candidate ids+scores, stage timings

    grounded    boolean,
    no_evidence boolean NOT NULL DEFAULT false,

    model              text,
    input_tokens       int,
    output_tokens      int,
    cache_read_tokens  int,
    cache_write_tokens int,
    cost_usd      numeric(10,6),
    latency_ms    int,
    finish_reason text,
    error         text,

    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chat_messages_session_idx ON chat_messages (session_id, seq);

CREATE TABLE IF NOT EXISTS message_feedback (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id uuid NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    user_id    uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    rating     text NOT NULL CHECK (rating IN ('helpful','needs_review')),
    reason     text,
    comment    text,
    created_at timestamptz NOT NULL DEFAULT now(),
    -- Upsert target: a user changing their mind updates rather than duplicating.
    UNIQUE (message_id, user_id)
);

-- The audit log issue #1 asks for: query, sources used, response, feedback.
CREATE TABLE IF NOT EXISTS query_log (
    id         bigserial PRIMARY KEY,
    user_id    uuid,
    session_id uuid,
    message_id uuid,
    endpoint   text NOT NULL CHECK (endpoint IN ('search','chat')),
    query           text NOT NULL,
    query_rewritten text,
    filters         jsonb,
    mode            text,
    retrieved_chunk_ids bigint[],
    result_count  int,
    top_scores    real[],
    had_evidence  boolean,
    latency_ms    int,
    index_version int,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS query_log_created_idx ON query_log (created_at DESC);
CREATE INDEX IF NOT EXISTS query_log_user_idx    ON query_log (user_id, created_at DESC);
