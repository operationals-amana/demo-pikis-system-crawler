"""Regression coverage for metadata constraints being applied before ranking."""

import unittest
from datetime import date

from pydantic import ValidationError

from app.schemas import FilterSpec
from rag.filters import Filters, build
from rag.metadata_filters import extract_explicit_filters


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _MetadataDb:
    def execute(self, statement):
        query = str(statement)
        if "FROM sources" in query:
            return _Rows([("pyc-wp", "Purnomo Yusgiantoro Center"), ("ije", "IJE")])
        if "FROM topics" in query:
            return _Rows([("electricity-grid", "Electricity & Grid", "Kelistrikan")])
        raise AssertionError(f"unexpected query: {query}")


class MetadataFilterExtractionTests(unittest.TestCase):
    def test_single_publication_year_is_a_closed_hard_range(self):
        filters = extract_explicit_filters(
            _MetadataDb(), "Show articles published in 2026 about subsidy reform"
        )
        self.assertEqual(filters.date_from, date(2026, 1, 1))
        self.assertEqual(filters.date_to, date(2026, 12, 31))

    def test_indonesian_article_query_with_bare_year_is_filtered(self):
        filters = extract_explicit_filters(_MetadataDb(), "Artikel energi terbaru 2026")
        self.assertEqual(filters.date_from, date(2026, 1, 1))
        self.assertEqual(filters.date_to, date(2026, 12, 31))

    def test_open_and_closed_year_ranges(self):
        since = extract_explicit_filters(_MetadataDb(), "articles published since 2020")
        self.assertEqual((since.date_from, since.date_to), (date(2020, 1, 1), None))
        between = extract_explicit_filters(
            _MetadataDb(), "articles published between 2020 and 2023"
        )
        self.assertEqual(
            (between.date_from, between.date_to),
            (date(2020, 1, 1), date(2023, 12, 31)),
        )

    def test_explicit_iso_date_range_overrides_year_boundaries(self):
        filters = extract_explicit_filters(
            _MetadataDb(),
            "articles year:2026 date_from:2026-03-01 date_to:2026-06-30",
        )
        self.assertEqual(
            (filters.date_from, filters.date_to),
            (date(2026, 3, 1), date(2026, 6, 30)),
        )

    def test_subject_year_without_metadata_signal_is_not_a_filter(self):
        filters = extract_explicit_filters(_MetadataDb(), "Indonesia 2045 energy strategy")
        self.assertTrue(filters.is_empty())

    def test_explicit_source_author_and_topic_are_extracted(self):
        filters = extract_explicit_filters(
            _MetadataDb(),
            'articles from Purnomo Yusgiantoro Center by "Jane Doe" '
            'in category Electricity & Grid',
        )
        self.assertEqual(filters.source_slugs, ["pyc-wp"])
        self.assertEqual(filters.authors, ["Jane Doe"])
        self.assertEqual(filters.topics, ["electricity-grid"])

        aliases = extract_explicit_filters(
            _MetadataDb(), 'source:PYC topic:"Electricity & Grid"'
        )
        self.assertEqual(aliases.source_slugs, ["pyc-wp"])
        self.assertEqual(aliases.topics, ["electricity-grid"])


class MetadataPredicateTests(unittest.TestCase):
    def test_every_constraint_is_in_the_shared_predicate(self):
        fragment, params = build(
            Filters(
                source_slugs=["pyc"],
                authors=["Jane Doe"],
                doc_types=["journal-article"],
                topics=["electricity-grid"],
                languages=["en"],
                date_from=date(2026, 1, 1),
                date_to=date(2026, 12, 31),
            )
        )
        for token in (
            "s.slug = ANY",
            "unnest(a.authors)",
            "a.doc_type = ANY",
            "a.language = ANY",
            "a.published_at >=",
            "a.published_at <=",
            "article_topics",
        ):
            self.assertIn(token, fragment)
        self.assertEqual(params["f_authors"], ["jane doe"])

    def test_invalid_date_range_is_rejected_at_api_boundary(self):
        with self.assertRaises(ValidationError):
            FilterSpec(date_from=date(2026, 2, 1), date_to=date(2026, 1, 1))


if __name__ == "__main__":
    unittest.main()
