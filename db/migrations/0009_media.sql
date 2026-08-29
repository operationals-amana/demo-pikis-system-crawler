-- Media Intelligence: monitoring news coverage of energy issues.
--
-- These tables are deliberately SEPARATE from the RAG corpus (`articles`). News
-- monitoring items are high-volume, shallow (title + feed summary, no full text),
-- and expire from interest in weeks; putting them in `articles` would drag every
-- one of them through dedupe/chunk/embed and pollute the knowledge base the chat
-- answers from. The two worlds meet only in the UI.

-- Outlets are rows, not config, for the same reason `sources` is a table: the
-- monitoring spider reads `WHERE is_active AND feed_url IS NOT NULL` at run time,
-- so an analyst enabling an outlet changes the next crawl with no code release.
CREATE TABLE IF NOT EXISTS media_outlets (
    id         smallserial PRIMARY KEY,
    slug       text NOT NULL UNIQUE,
    name       text NOT NULL,
    -- 'print' marks digitised print dailies (arsip cetak in the issue brief);
    -- their web feeds are the archive access path, there is no OCR pipeline.
    media_type text NOT NULL CHECK (media_type IN ('national', 'regional', 'print')),
    region     text,           -- province-level, NULL for national/print
    base_url   text,
    feed_url   text,           -- RSS/Atom endpoint polled by the news spider
    -- Outlets discovered via Google News get auto-created rows (discovered=true)
    -- so attribution is never lost; curated rows come from the seed migration.
    discovered boolean NOT NULL DEFAULT false,
    is_active  boolean NOT NULL DEFAULT true
);

-- A tracked issue mirrors the design of `topics`: an admin-editable row whose
-- keywords drive both the Google News queries and the relevance prefilter, so
-- tuning what "Energy Subsidy Reform" means needs no deploy.
CREATE TABLE IF NOT EXISTS tracked_issues (
    id                  smallserial PRIMARY KEY,
    slug                text NOT NULL UNIQUE,
    name                text NOT NULL,
    description         text,
    keywords            text[] NOT NULL DEFAULT '{}',
    default_period_days int  NOT NULL DEFAULT 30 CHECK (default_period_days IN (7, 30, 90)),
    is_active           boolean NOT NULL DEFAULT true,
    created_at          timestamptz NOT NULL DEFAULT now()
);

-- Which outlets an issue monitors. A row present = monitored; the configure UI
-- writes this join, and every dashboard query filters through it.
CREATE TABLE IF NOT EXISTS tracked_issue_outlets (
    issue_id  int NOT NULL REFERENCES tracked_issues(id) ON DELETE CASCADE,
    outlet_id int NOT NULL REFERENCES media_outlets(id) ON DELETE CASCADE,
    PRIMARY KEY (issue_id, outlet_id)
);

CREATE TABLE IF NOT EXISTS media_articles (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    outlet_id    int  NOT NULL REFERENCES media_outlets(id),
    -- The article URL (or feed guid when the URL is a redirector). Same upsert
    -- key shape as articles(source_id, external_id).
    external_id  text NOT NULL,
    url          text NOT NULL,
    title        text NOT NULL,
    summary      text,
    published_at timestamptz,
    language     text NOT NULL DEFAULT 'id',
    -- 'feed' (outlet RSS) or 'google-news' (keyword search); kept so the
    -- evidence feed can say where a hit came from and dedupe can prefer feeds.
    found_via    text NOT NULL DEFAULT 'feed' CHECK (found_via IN ('feed', 'google-news')),
    fetched_at   timestamptz NOT NULL DEFAULT now(),
    raw          jsonb,
    UNIQUE (outlet_id, external_id)
);
CREATE INDEX IF NOT EXISTS media_articles_published_idx ON media_articles (published_at DESC);
CREATE INDEX IF NOT EXISTS media_articles_outlet_idx ON media_articles (outlet_id, published_at DESC);
-- Cross-outlet dedupe: the same wire story appears on several Antara bureaus.
CREATE INDEX IF NOT EXISTS media_articles_title_idx ON media_articles (lower(title));

-- Narratives are per-issue and grow at run time: the analysis stage classifies
-- against the enum of existing rows and may propose a new one, which lands here
-- with first_seen_at = now(). "New narrative" alerts key off that timestamp.
CREATE TABLE IF NOT EXISTS media_narratives (
    id            smallserial PRIMARY KEY,
    issue_id      int  NOT NULL REFERENCES tracked_issues(id) ON DELETE CASCADE,
    slug          text NOT NULL,
    label         text NOT NULL,
    description   text,
    -- 'seed' = analyst-curated starting set, 'llm' = proposed by the analysis
    -- stage. Surfaced in the UI so analysts know what to review.
    created_by    text NOT NULL DEFAULT 'seed' CHECK (created_by IN ('seed', 'llm')),
    first_seen_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (issue_id, slug)
);

-- One row per (issue, article) the analysis stage has looked at -- including the
-- irrelevant ones (is_relevant = false), which is what makes the pending query
-- "no row yet" instead of a re-scan of everything every night.
CREATE TABLE IF NOT EXISTS media_article_analysis (
    issue_id        int  NOT NULL REFERENCES tracked_issues(id) ON DELETE CASCADE,
    article_id      uuid NOT NULL REFERENCES media_articles(id) ON DELETE CASCADE,
    relevance       real NOT NULL DEFAULT 0,        -- 0..1 from the model
    is_relevant     boolean NOT NULL DEFAULT false, -- relevance >= threshold at analysis time
    narrative_id    int REFERENCES media_narratives(id) ON DELETE SET NULL,
    -- Canonical short names ("Kementerian ESDM", "PLN", "DPR Komisi XII").
    -- An array, not a join table: actors are aggregated with unnest() and never
    -- edited row-by-row, so the join table would buy nothing but writes.
    actors          text[] NOT NULL DEFAULT '{}',
    analysis_source text NOT NULL DEFAULT 'llm' CHECK (analysis_source IN ('llm', 'rule')),
    analyzed_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (issue_id, article_id)
);
CREATE INDEX IF NOT EXISTS media_analysis_relevant_idx
    ON media_article_analysis (issue_id, is_relevant, narrative_id);

CREATE TABLE IF NOT EXISTS media_alerts (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    issue_id     int  NOT NULL REFERENCES tracked_issues(id) ON DELETE CASCADE,
    kind         text NOT NULL CHECK (kind IN
                     ('volume_spike', 'new_narrative', 'new_actor', 'regional_spread')),
    severity     text NOT NULL DEFAULT 'medium' CHECK (severity IN ('high', 'medium', 'low')),
    title        text NOT NULL,
    detail       text,
    confidence   real,
    -- Supporting media_articles ids (uuid strings). jsonb, not a join table:
    -- evidence is written once at detection and only ever read back whole.
    evidence     jsonb NOT NULL DEFAULT '[]',
    -- Detection runs nightly over sliding windows, so the same condition would
    -- re-fire every night. The dedup key (e.g. 'volume_spike:2026-08-29' or
    -- 'new_narrative:transition-financing') makes re-detection a no-op upsert.
    dedup_key    text NOT NULL,
    status       text NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'confirmed', 'dismissed')),
    triggered_at timestamptz NOT NULL DEFAULT now(),
    resolved_by  uuid REFERENCES users(id),
    resolved_at  timestamptz,
    UNIQUE (issue_id, dedup_key)
);
CREATE INDEX IF NOT EXISTS media_alerts_pending_idx
    ON media_alerts (issue_id, status, triggered_at DESC);
