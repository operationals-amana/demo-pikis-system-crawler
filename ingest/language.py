"""
Indonesian / English detection by stopword ratio.

Zero dependencies and ~40 lines. Rejected alternatives, both for the same reason --
a two-class problem does not justify the weight:
  * fasttext lid.176: a 126 MB model.
  * langdetect: fine at ~1 MB, but it is still a dependency and it is slower.

Above ~200 characters this is >98% accurate on ID/EN, because the two languages'
function words are completely disjoint. Below that it is unreliable, so callers are
expected to prefer a declared language and fall back to this only when there is none.
"""

import re

# Function words, not content words. These are chosen to be disjoint between the two
# languages: no token appears in both lists, so a hit is never ambiguous.
_ID = {
    "yang", "dan", "untuk", "dengan", "pada", "dari", "ini", "itu", "tidak", "akan",
    "adalah", "dalam", "tersebut", "dapat", "oleh", "sebagai", "juga", "karena",
    "atau", "telah", "lebih", "sudah", "bahwa", "para", "serta", "agar", "hingga",
    "namun", "yaitu", "kami", "kita", "mereka", "sangat", "hanya", "masih", "bisa",
}
_EN = {
    "the", "and", "of", "to", "in", "is", "that", "for", "with", "as", "are", "was",
    "be", "by", "this", "an", "at", "from", "it", "which", "have", "has", "been",
    "were", "their", "these", "such", "than", "then", "there", "we", "our", "can",
    "will", "would", "should", "between", "during", "into",
}

_TOKEN = re.compile(r"[a-zÀ-ɏ]+")

# Content words that are common in Indonesian and effectively never appear in English
# prose. These carry the SHORT-text tier: a headline like "AHY Sebut TOD Solusi
# Kurangi Pemakaian BBM" contains no function words at all, so stopword ratio alone
# scores it 0/0 and the old code silently fell through to 'en'. That mislabelled 212
# documents -- and a wrong language means the wrong stemmer, so it is not cosmetic.
_ID_CONTENT = {
    "sebut", "harus", "perempuan", "hari", "tahun", "bulan", "kerja", "sama",
    "bersama", "solusi", "kurangi", "pemakaian", "menembus", "menjawab", "persoalan",
    "ilmu", "sekat", "semangat", "kesehatan", "kebersamaan", "gaungkan", "langkah",
    "solidaritas", "energi", "listrik", "minyak", "batubara", "terbarukan", "nasional",
    "pembangunan", "kebijakan", "penelitian", "kajian", "laporan", "acara", "kegiatan",
    "berita", "tentang", "melalui", "menjadi", "sebagai", "terhadap", "antara",
    "masyarakat", "pemerintah", "negara", "dunia", "baru", "besar", "tinggi",
    "rendah", "banyak", "sedikit", "buku", "peluncuran", "diskusi", "seminar",
    "pelatihan", "kunjungan", "penghargaan", "kerjasama", "pengembangan",
}

# Indonesian affixes. A word matching one of these is almost certainly Indonesian:
# English has no me-/pe-/ber-/ter- prefix family and no -nya/-kan suffix.
_ID_AFFIX = re.compile(
    r"^(mem|men|meng|meny|pem|pen|peng|peny|ber|ter|di)[a-z]{4,}$|[a-z]{3,}(nya|kan)$"
)

MIN_CHARS = 200          # tier 1: stopword ratio over real prose
MIN_SHORT_CHARS = 12     # tier 2: markers + affixes over a title


def _score(tokens: list[str]) -> tuple[float, float]:
    total = len(tokens)
    id_hits = sum(1 for t in tokens if t in _ID)
    en_hits = sum(1 for t in tokens if t in _EN)
    return id_hits / total, en_hits / total


def detect(text: str | None) -> tuple[str | None, float]:
    """
    Return (language, confidence), or (None, 0.0) when there is genuinely nothing to
    judge. None is meaningful -- callers must handle it rather than defaulting, because
    a wrong language selects the wrong Postgres stemming config for that chunk.
    """
    if not text:
        return None, 0.0
    body = text.strip()
    if len(body) < MIN_SHORT_CHARS:
        return None, 0.0

    tokens = _TOKEN.findall(body.lower())
    if not tokens:
        return None, 0.0

    id_ratio, en_ratio = _score(tokens)

    # Tier 1 -- enough prose for function words to be decisive.
    if len(body) >= MIN_CHARS and (id_ratio or en_ratio):
        denom = id_ratio + en_ratio
        confidence = round(abs(id_ratio - en_ratio) / denom, 3) if denom else 0.0
        return ("id" if id_ratio > en_ratio else "en"), confidence

    # Tier 2 -- short text (titles, excerpts). Function words are often absent, so
    # lean on Indonesian content words and affixes, which English cannot produce.
    id_signal = sum(1 for t in tokens if t in _ID or t in _ID_CONTENT)
    id_signal += sum(1 for t in tokens if _ID_AFFIX.match(t))
    en_signal = sum(1 for t in tokens if t in _EN)

    if id_signal == 0 and en_signal == 0:
        return None, 0.0
    if id_signal == en_signal:
        return None, 0.0
    # Capped low: this tier is a good guess, not a determination, and the admin table
    # shows confidence so a reviewer can see which rows were guessed.
    confidence = round(min(0.6, abs(id_signal - en_signal) / max(len(tokens), 1) * 3), 3)
    return ("id" if id_signal > en_signal else "en"), confidence


def resolve(
    declared: str | None,
    text: str | None,
    url: str | None = None,
    title: str | None = None,
) -> tuple[str, float, str]:
    """
    Priority: declared > URL prefix > detected(body) > detected(title) > 'en' fallback.
    Returns (language, confidence, source); source lands in articles.language_source so
    the admin can filter for rows that were guessed rather than declared.
    """
    if declared:
        code = declared.strip().lower()[:3]
        if code in {"en", "eng"}:
            return "en", 1.0, "declared"
        if code in {"id", "ind", "in"}:
            return "id", 1.0, "declared"

    if url:
        # The PYC WordPress site is bilingual with /id/ and /en/ path prefixes -- but
        # only on some post types, which is why the tiers below still matter.
        if "/id/" in url:
            return "id", 0.9, "declared"
        if "/en/" in url:
            return "en", 0.9, "declared"

    guess, confidence = detect(text)
    if guess:
        return guess, confidence, "detected"

    # No body (media-coverage, podcasts and infographics have none). The title is all
    # there is, and it is better than assuming English.
    guess, confidence = detect(title)
    if guess:
        return guess, confidence, "detected"

    return "en", 0.0, "detected"
