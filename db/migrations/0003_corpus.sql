-- Articles and chunks: the corpus.

CREATE TABLE IF NOT EXISTS articles (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id   smallint NOT NULL REFERENCES sources(id),
    external_id text     NOT NULL,
    -- Re-harvest idempotency. This is upsert, NOT duplicate detection -- the four
    -- duplicate layers live in ingest/dedupe.py and write duplicate_of_id.
    UNIQUE (source_id, external_id),

    doc_type    text NOT NULL,   -- journal-article | post | media-coverage | opinion |
                                 -- brief-report | book | podcast | e-newsletter |
                                 -- infographic | archive | works-archive
    title            text NOT NULL,
    title_normalized text,       -- lowercased, unaccented, punctuation-stripped
    abstract         text,
    full_text        text,
    full_text_status text NOT NULL DEFAULT 'none'
        CHECK (full_text_status IN ('none','abstract','extracted','poor','failed')),

    authors   text[] NOT NULL DEFAULT '{}',
    keywords  text[] NOT NULL DEFAULT '{}',
    publisher text,
    journal   text,
    volume    text,
    issue     text,
    pages     text,
    doi       text,
    issn      text[] NOT NULL DEFAULT '{}',

    published_at   date,
    published_year int,          -- denormalised by trigger; the by-year chart groups on it
    url            text,
    pdf_url        text,
    galley_id      text,

    language            char(2) CHECK (language IN ('id','en')),
    language_confidence real,
    language_source     text,    -- declared | detected | manual

    topic_primary_id smallint REFERENCES topics(id),
    topic_confidence real,
    topic_source     text,       -- llm | rule | manual

    word_count int,
    char_count int,

    content_hash    char(64),
    dedup_key       char(64),
    duplicate_of_id uuid REFERENCES articles(id) ON DELETE SET NULL,
    duplicate_reason text,       -- doi | content_hash | title_year | manual

    processing_status text NOT NULL DEFAULT 'pending'
        CHECK (processing_status IN ('pending','processing','ready','failed','duplicate','archived')),
    processing_error    text,
    processing_attempts int NOT NULL DEFAULT 0,
    processed_at        timestamptz,

    chunk_count          int NOT NULL DEFAULT 0,
    embedded_chunk_count int NOT NULL DEFAULT 0,
    embedding_model      text,
    embedded_at          timestamptz,

    -- Fields an admin edited. Ingestion never overwrites these, which is the single
    -- mechanism that makes "admin can edit metadata" survive the next harvest. The
    -- alternative is a *_source column per editable field -- eleven columns, eleven checks.
    metadata_locked_fields text[] NOT NULL DEFAULT '{}',

    -- The untouched upstream record. A parser bug is then fixed by re-running
    -- normalize over stored rows instead of re-hitting IJE and WordPress.
    raw jsonb,

    search_vector tsvector,      -- admin table search ONLY; RAG's lexical channel is BM25

    first_seen_at    timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now(),
    last_harvested_at timestamptz
);

-- search_vector is maintained by a TRIGGER, not a GENERATED ... STORED column.
-- A generated column needs an IMMUTABLE wrapper (to_tsvector(text, text) -- config
-- by name -- is only STABLE), and Postgres then refuses to ALTER that function
-- without dropping the column. Search weighting is exactly what you will want to
-- tune, so it must stay alterable: this is retunable by UPDATE articles SET id = id.
CREATE OR REPLACE FUNCTION articles_search_vector_trigger() RETURNS trigger AS $$
DECLARE cfg regconfig;
BEGIN
    cfg := CASE WHEN NEW.language = 'id' THEN 'indonesian' ELSE 'english' END::regconfig;
    NEW.published_year := EXTRACT(YEAR FROM NEW.published_at)::int;
    NEW.search_vector :=
          setweight(to_tsvector(cfg, coalesce(NEW.title, '')), 'A')
       || setweight(to_tsvector(cfg, coalesce(array_to_string(NEW.keywords, ' '), '')), 'B')
       || setweight(to_tsvector(cfg, coalesce(NEW.abstract, '')), 'C');
    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS articles_search_vector_update ON articles;
CREATE TRIGGER articles_search_vector_update
    BEFORE INSERT OR UPDATE ON articles
    FOR EACH ROW EXECUTE FUNCTION articles_search_vector_trigger();

CREATE INDEX IF NOT EXISTS articles_search_idx      ON articles USING gin (search_vector);
CREATE INDEX IF NOT EXISTS articles_title_trgm_idx  ON articles USING gin (title_normalized gin_trgm_ops);
CREATE INDEX IF NOT EXISTS articles_status_idx      ON articles (processing_status);
CREATE INDEX IF NOT EXISTS articles_published_idx   ON articles (published_at);
CREATE INDEX IF NOT EXISTS articles_year_idx        ON articles (published_year);
CREATE INDEX IF NOT EXISTS articles_source_idx      ON articles (source_id);
CREATE INDEX IF NOT EXISTS articles_language_idx    ON articles (language);
CREATE INDEX IF NOT EXISTS articles_topic_idx       ON articles (topic_primary_id);
CREATE INDEX IF NOT EXISTS articles_dedup_key_idx   ON articles (dedup_key);
CREATE INDEX IF NOT EXISTS articles_duplicate_idx   ON articles (duplicate_of_id);
CREATE INDEX IF NOT EXISTS articles_admin_sort_idx  ON articles (source_id, processing_status, published_at DESC);
-- A DOI is an identity. Partial unique so a collision is loud rather than silent.
CREATE UNIQUE INDEX IF NOT EXISTS articles_doi_key ON articles (lower(doi)) WHERE doi IS NOT NULL;

-- Many-to-many so an article can carry secondary topics; topic_primary_id is the
-- one the admin chart and the filter dropdown use.
CREATE TABLE IF NOT EXISTS article_topics (
    article_id uuid     NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    topic_id   smallint NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    rank       smallint NOT NULL DEFAULT 1,
    confidence real,
    PRIMARY KEY (article_id, topic_id)
);
CREATE INDEX IF NOT EXISTS article_topics_topic_idx ON article_topics (topic_id);

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    article_id  uuid NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    chunk_index int  NOT NULL,
    UNIQUE (article_id, chunk_index),

    content    text NOT NULL,
    -- Offsets into articles.full_text, so a citation can be highlighted in place
    -- without a second retrieval.
    char_start int,
    char_end   int,
    section    text,
    token_count int,
    -- May differ from the article's: an Indonesian paper's English abstract must be
    -- lexed under 'english', not 'indonesian'.
    language   char(2),
    -- to_tsvector(lang_cfg, content) flattened. Postgres does the stemming (its
    -- indonesian_stem is genuinely good); Python does the BM25 scoring, because
    -- ts_rank has no IDF at all and every document here is about energy.
    lexemes    text[],
    content_hash char(64),       -- skip re-embedding an unchanged chunk on re-ingest

    embedding       vector(384),
    embedding_model text,
    embedded_at     timestamptz,

    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chunks_article_idx ON chunks (article_id);
-- Covering index for the BM25 id -> row hop.
CREATE INDEX IF NOT EXISTS chunks_id_article_idx ON chunks (id) INCLUDE (article_id);

-- NOTE: there is deliberately NO ANN index on chunks.embedding. See 0007 for the
-- full reasoning and the exact condition under which to add one.
