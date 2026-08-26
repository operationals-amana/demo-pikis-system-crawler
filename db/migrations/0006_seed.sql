-- Seed: the two sources, and the energy topic taxonomy.
--
-- Idempotent upserts so re-running migrations never duplicates, and so editing a
-- label here and re-running actually updates it. `keywords` deliberately does NOT
-- get overwritten on conflict: it is tuned against the live corpus after the first
-- classification pass, and a migration re-run must not clobber that tuning.

INSERT INTO sources (slug, name, publisher, base_url, kind) VALUES
    ('ije',    'Indonesian Journal of Energy', 'Purnomo Yusgiantoro Center',
               'https://ije-pyc.org/IJE/', 'journal'),
    ('pyc-wp', 'Purnomo Yusgiantoro Center',   'Purnomo Yusgiantoro Center',
               'https://purnomoyusgiantorocenter.org/', 'website')
ON CONFLICT (slug) DO UPDATE
    SET name = EXCLUDED.name, publisher = EXCLUDED.publisher, base_url = EXCLUDED.base_url;

-- The taxonomy is derived from the corpus, not invented: these seed keywords are the
-- actual dc:subject tags attached to IJE articles (442 tags, 361 unique), with the
-- frequency of the headline term noted where it is telling.
INSERT INTO topics (slug, label_en, label_id, description, keywords, sort_order) VALUES
    ('energy-transition', 'Energy Transition & Climate', 'Transisi Energi & Iklim',
     'Decarbonisation pathways, net-zero targets, carbon pricing and capture, climate policy instruments. A paper on coal-plant EMISSIONS belongs here, not in fossil-fuels.',
     ARRAY['energy transition','transisi energi','decarbonisation','decarbonization','net zero','climate change','perubahan iklim','carbon pricing','carbon capture','ccs','ccus','cbam','carbon border adjustment','emission','emisi','paris agreement'], 10),

    ('renewables', 'Renewable Energy', 'Energi Terbarukan',
     'Solar, wind, hydro, geothermal generation technology, potential and deployment.',
     ARRAY['renewable energy','energi terbarukan','solar','surya','photovoltaic','solar pv','wind','angin','hydro','hidro','geothermal','panas bumi','ebt'], 20),

    ('bioenergy-waste', 'Bioenergy & Waste', 'Bioenergi & Limbah',
     'Biomass, biogas, biofuels, palm-oil derived energy, waste-to-energy.',
     ARRAY['biomass','biomassa','biogas','biofuel','bioenergy','bioenergi','palm oil','kelapa sawit','empty fruit bunch','waste to energy','pyrolysis','pirolisis','gasification','gasifikasi'], 30),

    ('oil-gas', 'Oil & Gas', 'Minyak & Gas',
     'Upstream and downstream petroleum and natural gas, LNG, refining, enhanced oil recovery.',
     ARRAY['oil','minyak','gas','natural gas','gas alam','lng','petroleum','migas','upstream','hulu','downstream','hilir','refinery','kilang','enhanced oil recovery','eor','drilling'], 40),

    ('coal-mining', 'Coal & Mining', 'Batubara & Pertambangan',
     'Coal production, mining, coal trade and coal-fired generation as a fuel question.',
     ARRAY['coal','batubara','mining','pertambangan','tambang','lignite','coal fired','pltu','hybrid coal'], 50),

    ('power-grid', 'Electricity & Grid', 'Kelistrikan & Jaringan',
     'Power systems, transmission and distribution, grid integration, rural electrification, EV charging.',
     ARRAY['electricity','listrik','kelistrikan','grid','jaringan','transmission','transmisi','distribution','distribusi','power system','sistem tenaga','pln','electrification','elektrifikasi','rural electrification','charging station','street lighting','microgrid','smart grid'], 60),

    ('energy-policy', 'Energy Policy & Security', 'Kebijakan & Ketahanan Energi',
     'Regulation, governance, energy security, subsidies, international energy relations and geopolitics.',
     ARRAY['energy policy','kebijakan energi','energy security','ketahanan energi','regulation','regulasi','governance','tata kelola','subsidy','subsidi','geopolitics','geopolitik','international relations','energy law','ruen','kebijakan energi nasional'], 70),

    ('energy-economics', 'Energy Economics & Investment', 'Ekonomi & Investasi Energi',
     'Financing, investment, project economics, LCOE, tariffs, energy markets and trade.',
     ARRAY['investment','investasi','financing','pembiayaan','economics','ekonomi','lcoe','levelized cost','tariff','tarif','market','pasar','trade','perdagangan','cost','biaya','feasibility','kelayakan'], 80),

    ('energy-efficiency', 'Energy Efficiency', 'Efisiensi Energi',
     'Conservation, demand-side management, efficiency standards, energy consumption and management.',
     ARRAY['energy efficiency','efisiensi energi','energy conservation','konservasi energi','energy management','manajemen energi','energy consumption','konsumsi energi','demand side','audit energi'], 90),

    ('nuclear', 'Nuclear', 'Nuklir',
     'Nuclear power, reactor technology, nuclear policy and safety.',
     ARRAY['nuclear','nuklir','reactor','reaktor','uranium','thorium','smr','pltn'], 100),

    ('energy-access', 'Energy Access & Society', 'Akses Energi & Masyarakat',
     'Energy poverty, access and affordability, social impact, just transition, community energy.',
     ARRAY['energy access','akses energi','energy poverty','kemiskinan energi','just transition','transisi berkeadilan','community','masyarakat','social','sosial','affordability','keterjangkauan','rural','pedesaan'], 110),

    ('energy-modelling', 'Energy Modelling & Data', 'Pemodelan & Data Energi',
     'Forecasting, scenario and system modelling, machine learning and geospatial analysis applied to energy.',
     ARRAY['modelling','modeling','pemodelan','forecast','proyeksi','scenario','skenario','machine learning','big data','geospatial','simulation','simulasi','homer','optimization','optimasi'], 120),

    ('other', 'Other', 'Lainnya',
     'Does not fit any energy topic above -- typically institutional news, events, or non-energy content.',
     ARRAY[]::text[], 999)
ON CONFLICT (slug) DO UPDATE
    SET label_en    = EXCLUDED.label_en,
        label_id    = EXCLUDED.label_id,
        description = EXCLUDED.description,
        sort_order  = EXCLUDED.sort_order;
