"""
Combined Indonesian + English stopword list for the BM25 channel.

This exists because of a specific, verified gap: Postgres 15's `indonesian`
text-search config has REAL stemming (terbarukan -> baru, pembiayaan -> biaya) but an
EMPTY stopword file. So `dan`, `yang`, `untuk` survive into chunks.lexemes as ordinary
lexemes. Left in, they are the highest-frequency terms in the corpus and BM25's IDF
would be computed over a vocabulary dominated by function words.

English lexemes come through Postgres' english config, which does strip stopwords --
the English half here is belt-and-braces for chunks whose language was misdetected.
"""

# Postgres' indonesian snowball stems these before we ever see them, so the list is
# written in STEMMED form where stemming applies (e.g. "adalah" -> "adalah",
# "merupakan" -> "rupa"). Both surface and stemmed forms are included: a redundant
# entry costs nothing, a missing one costs IDF quality.
INDONESIAN = {
    "yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan", "ini", "itu",
    "atau", "juga", "akan", "telah", "sudah", "belum", "tidak", "bukan", "adalah",
    "ialah", "merupakan", "rupa", "dalam", "oleh", "sebagai", "karena", "sebab",
    "agar", "supaya", "hingga", "sampai", "namun", "tetapi", "tapi", "melainkan",
    "yaitu", "yakni", "serta", "para", "kami", "kita", "mereka", "dia", "ia",
    "saya", "anda", "nya", "sangat", "lebih", "paling", "hanya", "saja", "masih",
    "bisa", "dapat", "harus", "perlu", "boleh", "ada", "tersebut", "sebuah",
    "suatu", "seperti", "antara", "terhadap", "tentang", "mengenai", "bagi",
    "atas", "bawah", "setelah", "sebelum", "ketika", "saat", "selama", "guna",
    "melalui", "menjadi", "jadi", "lain", "lainnya", "yg", "dll", "dsb",
}

ENGLISH = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "by",
    "for", "with", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "which", "who", "whom", "whose",
    "what", "when", "where", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "can", "will", "just", "should", "now",
    "also", "may", "might", "must", "shall", "would", "could", "have", "has",
    "had", "do", "does", "did", "we", "our", "they", "their", "there", "here",
    "into", "during", "between", "among", "within", "without", "about", "through",
}

# Corpus-specific noise. Every document here is a PYC energy publication, so these
# terms carry almost no discriminating signal -- but they are NOT removed, only
# down-weighted naturally by IDF. Listed here for documentation, not filtering:
# "energy", "energi", "indonesia", "pyc". Removing them would break the legitimate
# query "energy security in Indonesia".

STOPWORDS = INDONESIAN | ENGLISH


def strip(lexemes: list[str] | None) -> list[str]:
    if not lexemes:
        return []
    return [lx for lx in lexemes if lx not in STOPWORDS and len(lx) > 1]
