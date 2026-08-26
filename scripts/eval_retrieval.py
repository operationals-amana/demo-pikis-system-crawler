"""
The retrieval quality gate.

Twenty hand-written (question -> expected article) pairs, half Indonesian and half
English, spanning IJE papers and PYC briefs. Prints recall@5, recall@10 and MRR for
each channel separately AND for the fusion, so a regression can be attributed to the
channel that caused it.

PHASE 3 DOES NOT COMPLETE UNTIL FUSED recall@5 >= 0.80.

The reason this exists before any prompt code: a wrong answer from a RAG system is
almost always a retrieval failure wearing a generation costume. Without this number
you end up tuning prompts to compensate for a retriever that never surfaced the right
passage, which cannot work and wastes days.

The Indonesian half is not decoration. 93 of 95 IJE articles are English while the
wireframe's flagship demo query is Indonesian, so cross-lingual retrieval is the
single most likely thing to look bad in a demo. These cases measure exactly that.
"""

import argparse
import sys

# (question, substring that must appear in the title of a correct article)
# Matching on a title substring rather than a UUID keeps the file readable and
# survives a re-harvest that regenerates ids.
GOLDEN: list[tuple[str, str]] = [
    # --- English, IJE papers -------------------------------------------------
    ("carbon capture and utilization in the oil and gas industry", "Carbon Capture and Utilization"),
    ("EU carbon border adjustment mechanism impact on Indonesian trade", "Carbon Border Adjustment"),
    ("transparent photovoltaics for building windows", "Transparent Photovoltaics"),
    ("thermoelectric generator as a renewable energy source", "Thermoelectric Generator"),
    ("PEM fuel cell purging interval optimization", "PEM Fuel Cell"),
    ("Indonesia energy relations with Gulf Cooperation Council", "Gulf Cooperation Council"),
    ("waste to energy nationally determined contributions", "Waste-to-E"),
    ("kinetic facade typology for buildings in Indonesia", "Kinetic Fa"),
    ("securitization of energy issues in security studies", "Securitization of Energy"),
    ("bacterio-algal fuel cells for future energy", "Bacterio-Algal"),
    ("renewable energy investment opportunities overview", "Renewable Energy Studies"),
    ("green financial mechanism for smart renewable investment", "Green Financial Mechanism"),
    # --- Indonesian questions against a mostly-English corpus ----------------
    # This is the cross-lingual soft spot: 93/95 IJE articles are English, so the
    # lexical channel contributes almost nothing here and semantic carries it.
    ("penangkapan dan pemanfaatan karbon di industri minyak dan gas", "Carbon Capture and Utilization"),
    ("dampak mekanisme penyesuaian batas karbon Uni Eropa bagi Indonesia", "Carbon Border Adjustment"),
    ("sel surya transparan untuk jendela bangunan", "Transparent Photovoltaics"),
    ("hubungan energi Indonesia dengan negara Teluk", "Gulf Cooperation Council"),
    ("pembangkit termoelektrik sebagai energi terbarukan", "Thermoelectric Generator"),
    ("pengelolaan sampah menjadi energi listrik", "Waste-to-E"),
    ("investasi energi terbarukan di Indonesia", "Renewable Energy Studies"),
    ("keamanan dan ketahanan energi nasional", "Securitization of Energy"),
]


def _rank_of(results: list[dict], needle: str) -> int | None:
    """1-based rank of the first result whose article title contains `needle`."""
    seen_articles: list[str] = []
    for row in results:
        article_id = row.get("article_id")
        if article_id in seen_articles:
            continue
        seen_articles.append(article_id)
        if needle.lower() in (row.get("title") or "").lower():
            return len(seen_articles)
    return None


def _metrics(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks)
    r5 = sum(1 for r in ranks if r and r <= 5) / n
    r10 = sum(1 for r in ranks if r and r <= 10) / n
    mrr = sum((1.0 / r) for r in ranks if r) / n
    return {"recall@5": r5, "recall@10": r10, "mrr": mrr}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=float, default=0.80, help="minimum fused recall@5")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-translate", action="store_true",
                        help="disable the cross-lingual query translation, to measure its effect")
    args = parser.parse_args()

    from app.logging_utils import configure_logging

    configure_logging()

    from db.engine import SessionLocal
    from ingest.embedder import embed_query
    from rag import fusion, lexical, semantic
    from rag.generator import rewrite_query
    from rag.retriever import retrieve

    db = SessionLocal()
    lexical.ensure_loaded(db, force=True)

    per_channel: dict[str, list[int | None]] = {"lexical": [], "semantic": [], "fused": []}
    misses: list[tuple[str, str]] = []

    try:
        for question, needle in GOLDEN:
            vector = embed_query(question)

            # Each channel alone, so a regression is attributable.
            lex = lexical.search(db, question, limit=20)
            sem = semantic.search(db, vector, limit=20)

            def hydrate(pairs: list[tuple[int, float]]) -> list[dict]:
                if not pairs:
                    return []
                from sqlalchemy import text as sql

                ids = [p[0] for p in pairs]
                rows = db.execute(
                    sql(
                        "SELECT c.id, c.article_id::text AS article_id, a.title "
                        "FROM chunks c JOIN articles a ON a.id=c.article_id WHERE c.id = ANY(:ids)"
                    ),
                    {"ids": ids},
                ).mappings().all()
                by_id = {r["id"]: dict(r) for r in rows}
                return [by_id[i] for i, _ in pairs if i in by_id]

            per_channel["lexical"].append(_rank_of(hydrate(lex), needle))
            per_channel["semantic"].append(_rank_of(hydrate(sem), needle))

            # The translated phrasing is what makes cross-lingual retrieval work at
            # all: an Indonesian question ranks the right English paper at
            # 7-or-missing, its English phrasing at 1. Measured, not assumed --
            # --no-translate reproduces the failure.
            extra = None
            if not args.no_translate:
                from app.config import TRANSLATE_FROM_LANGUAGES

                rewritten = rewrite_query(question)
                # Directional, matching production: id -> en only.
                if rewritten.query_translated and rewritten.language in TRANSLATE_FROM_LANGUAGES:
                    extra = [rewritten.query_translated]
            result = retrieve(db, question, top_k=10, query_vector=vector, extra_queries=extra)
            rank = _rank_of(result.candidates, needle)
            per_channel["fused"].append(rank)

            if rank is None or rank > 5:
                misses.append((question, needle))
            if args.verbose:
                print(f"  {str(rank or '-'):>3}  {question[:62]}")
    finally:
        db.close()

    print()
    print(f"{'channel':<10} {'recall@5':>9} {'recall@10':>10} {'MRR':>7}")
    print("-" * 39)
    for channel in ("lexical", "semantic", "fused"):
        m = _metrics(per_channel[channel])
        print(f"{channel:<10} {m['recall@5']:>9.2f} {m['recall@10']:>10.2f} {m['mrr']:>7.3f}")

    # Indonesian half reported separately -- it is the cross-lingual soft spot and
    # an aggregate number hides it.
    id_ranks = per_channel["fused"][12:]
    en_ranks = per_channel["fused"][:12]
    print()
    print(f"  english questions  recall@5 {_metrics(en_ranks)['recall@5']:.2f}")
    print(f"  indonesian questions recall@5 {_metrics(id_ranks)['recall@5']:.2f}   <- cross-lingual")

    if misses:
        print(f"\nmissed or ranked below 5 ({len(misses)}):")
        for question, needle in misses:
            print(f"  - {question[:66]}  (wanted: {needle})")

    fused_r5 = _metrics(per_channel["fused"])["recall@5"]
    print()
    if fused_r5 >= args.gate:
        print(f"GATE PASSED: fused recall@5 {fused_r5:.2f} >= {args.gate:.2f}")
        return 0
    print(f"GATE FAILED: fused recall@5 {fused_r5:.2f} < {args.gate:.2f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
