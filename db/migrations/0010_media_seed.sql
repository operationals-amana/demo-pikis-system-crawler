-- Seed: curated media outlets and the first tracked issue.
--
-- Idempotent upserts, same bargain as 0006: re-running updates names/URLs but
-- deliberately does NOT overwrite `keywords` on tracked_issues -- analysts tune
-- those in the configure UI and a migration re-run must not clobber the tuning.

-- National outlets: the economy/energy desks of the major wires and portals.
-- Feed URLs are the long-stable RSS endpoints; a dead feed degrades to zero items
-- for that outlet (the spider's errback records it), it does not fail the crawl.
INSERT INTO media_outlets (slug, name, media_type, region, base_url, feed_url) VALUES
    ('antara',        'ANTARA News',    'national', NULL, 'https://www.antaranews.com',    'https://www.antaranews.com/rss/ekonomi.xml'),
    ('cnn-indonesia', 'CNN Indonesia',  'national', NULL, 'https://www.cnnindonesia.com',  'https://www.cnnindonesia.com/ekonomi/rss'),
    ('tempo',         'Tempo',          'national', NULL, 'https://www.tempo.co',          'https://rss.tempo.co/bisnis'),
    ('detik-finance', 'detikFinance',   'national', NULL, 'https://finance.detik.com',     'https://rss.detik.com/index.php/finance'),
    ('bisnis',        'Bisnis Indonesia', 'national', NULL, 'https://www.bisnis.com',      'https://ekonomi.bisnis.com/rss'),
    ('republika',     'Republika',      'national', NULL, 'https://www.republika.co.id',   'https://www.republika.co.id/rss/ekonomi'),
    ('katadata',      'Katadata',       'national', NULL, 'https://katadata.co.id',        'https://katadata.co.id/rss'),

-- Regional: the ANTARA provincial bureaus publish reliable per-province RSS, which
-- is what makes "isu mulai banyak dibahas media regional" detectable at all.
    ('antara-kalsel', 'ANTARA Kalsel',  'regional', 'Kalimantan Selatan', 'https://kalsel.antaranews.com',   'https://kalsel.antaranews.com/rss/ekonomi.xml'),
    ('antara-kaltim', 'ANTARA Kaltim',  'regional', 'Kalimantan Timur',   'https://kaltim.antaranews.com',   'https://kaltim.antaranews.com/rss/ekonomi.xml'),
    ('antara-sulsel', 'ANTARA Sulsel',  'regional', 'Sulawesi Selatan',   'https://makassar.antaranews.com', 'https://makassar.antaranews.com/rss/ekonomi.xml'),
    ('antara-jatim',  'ANTARA Jatim',   'regional', 'Jawa Timur',         'https://jatim.antaranews.com',    'https://jatim.antaranews.com/rss/ekonomi.xml'),
    ('antara-sumbar', 'ANTARA Sumbar',  'regional', 'Sumatera Barat',     'https://sumbar.antaranews.com',   'https://sumbar.antaranews.com/rss/ekonomi.xml'),
    ('antara-papua',  'ANTARA Papua',   'regional', 'Papua',              'https://papua.antaranews.com',    'https://papua.antaranews.com/rss/ekonomi.xml'),
    ('antara-ntt',    'ANTARA NTT',     'regional', 'Nusa Tenggara Timur','https://kupang.antaranews.com',   'https://kupang.antaranews.com/rss/ekonomi.xml'),
    ('tribun-kaltim', 'Tribun Kaltim',  'regional', 'Kalimantan Timur',   'https://kaltim.tribunnews.com',   'https://kaltim.tribunnews.com/rss'),
    ('tribun-timur',  'Tribun Timur',   'regional', 'Sulawesi Selatan',   'https://makassar.tribunnews.com', 'https://makassar.tribunnews.com/rss'),

-- Print: digitised print dailies. Their sites carry the print edition's stories,
-- which is the closest thing to "arsip cetak" without an OCR pipeline (explicitly
-- out of scope). Attribution via Google News also lands on these rows by domain.
    ('media-indonesia', 'Media Indonesia', 'print', NULL, 'https://mediaindonesia.com', 'https://mediaindonesia.com/feed'),
    ('investor-daily',  'Investor Daily',  'print', NULL, 'https://investor.id',        'https://investor.id/rss')
ON CONFLICT (slug) DO UPDATE
    SET name = EXCLUDED.name, media_type = EXCLUDED.media_type,
        region = EXCLUDED.region, base_url = EXCLUDED.base_url,
        feed_url = EXCLUDED.feed_url;

-- The demo issue from the brief. Keywords are the Google News queries AND the
-- relevance prefilter, so they are phrased the way Indonesian headlines are.
INSERT INTO tracked_issues (slug, name, description, keywords, default_period_days) VALUES
    ('energy-subsidy-reform', 'Energy Subsidy Reform',
     'Coverage of Indonesian energy subsidy policy: fuel (BBM), LPG and electricity subsidies, targeting reform, fiscal burden, and compensation schemes.',
     ARRAY['subsidi BBM','subsidi energi','subsidi listrik','subsidi LPG','LPG 3 kg',
           'kompensasi energi','harga BBM','BBM bersubsidi','pertalite','solar subsidi',
           'tarif listrik','subsidi tepat sasaran','pengalihan subsidi','anggaran subsidi energi',
           'bansos energi'],
     30)
ON CONFLICT (slug) DO UPDATE
    SET name = EXCLUDED.name, description = EXCLUDED.description;

-- Analyst-curated starting narratives. The analysis stage classifies against
-- these and may propose additions (created_by = 'llm').
INSERT INTO media_narratives (issue_id, slug, label, description)
SELECT i.id, n.slug, n.label, n.description
FROM tracked_issues i,
     (VALUES
        ('fiscal-burden', 'Fiscal burden & targeting',
         'Subsidies framed as APBN pressure: cost overruns, quota breaches, calls for tighter targeting of who may buy subsidised fuel.'),
        ('household-affordability', 'Household affordability',
         'Subsidies framed through purchasing power: price rises, inflation, protests, effects on poor and near-poor households.'),
        ('transition-financing', 'Energy transition financing',
         'Redirecting subsidy spending toward renewables, EV incentives and grid investment; subsidy reform as transition finance.'),
        ('regional-equity', 'Regional equity & distribution',
         'Distribution problems outside Java: scarcity, one-price policy (BBM satu harga), transport costs, remote-area access.')
     ) AS n(slug, label, description)
WHERE i.slug = 'energy-subsidy-reform'
ON CONFLICT (issue_id, slug) DO UPDATE
    SET label = EXCLUDED.label, description = EXCLUDED.description;

-- Monitor every curated outlet for the seeded issue.
INSERT INTO tracked_issue_outlets (issue_id, outlet_id)
SELECT i.id, o.id
FROM tracked_issues i, media_outlets o
WHERE i.slug = 'energy-subsidy-reform' AND NOT o.discovered
ON CONFLICT DO NOTHING;
