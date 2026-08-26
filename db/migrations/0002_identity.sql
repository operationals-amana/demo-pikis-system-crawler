-- Users, sessions-worth-of-refresh-tokens, and the two reference tables.

CREATE TABLE IF NOT EXISTS users (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),  -- built into PG13+, no pgcrypto
    email         text        NOT NULL,
    password_hash text        NOT NULL,
    full_name     text,
    role          text        NOT NULL DEFAULT 'researcher'
                              CHECK (role IN ('admin', 'researcher')),
    is_active     boolean     NOT NULL DEFAULT true,
    created_at    timestamptz NOT NULL DEFAULT now(),
    last_login_at timestamptz
);
-- Case-insensitive uniqueness without depending on citext being installed.
CREATE UNIQUE INDEX IF NOT EXISTS users_email_lower_key ON users (lower(email));

CREATE TABLE IF NOT EXISTS refresh_tokens (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     uuid        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    -- sha256 of the token, never the token itself: a DB leak must not be a session leak.
    token_hash  char(64)    NOT NULL UNIQUE,
    expires_at  timestamptz NOT NULL,
    revoked_at  timestamptz,
    replaced_by uuid,
    user_agent  text,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS refresh_tokens_user_idx ON refresh_tokens (user_id);

-- Makes the admin's "count of sources/publishers" a real query instead of a
-- count(distinct publisher) over free text that drifts with every harvest.
CREATE TABLE IF NOT EXISTS sources (
    id        smallserial PRIMARY KEY,
    slug      text NOT NULL UNIQUE,
    name      text NOT NULL,
    publisher text,
    base_url  text,
    kind      text
);

-- A TABLE, not a Python enum: the topic classifier's json_schema enum is built at
-- runtime from these rows, so editing the taxonomy needs no code release.
CREATE TABLE IF NOT EXISTS topics (
    id          smallserial PRIMARY KEY,
    slug        text NOT NULL UNIQUE,
    label_en    text NOT NULL,
    label_id    text NOT NULL,
    description text,
    keywords    text[] NOT NULL DEFAULT '{}',
    sort_order  int    NOT NULL DEFAULT 100,
    is_active   boolean NOT NULL DEFAULT true
);
