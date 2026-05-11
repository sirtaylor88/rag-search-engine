"""Tests for cli.core.hybrid_search."""

from unittest.mock import MagicMock, patch

import pytest

from cli.core.hybrid_search import (
    HybridSearch,
    hybrid_score,
    normalize_scores,
    rrf_score,
)
from cli.core.keyword_search import InvertedIndex
from cli.core.semantic_search import ChunkedSemanticSearch


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset all relevant singletons before each test."""
    monkeypatch.setattr(HybridSearch, "_instance", None)
    monkeypatch.setattr(InvertedIndex, "_instance", None)
    monkeypatch.setattr(ChunkedSemanticSearch, "_instance", None)


def _make_docs() -> list[dict]:
    return [
        {"id": 1, "title": "Movie A", "description": "Action adventure film."},
        {"id": 2, "title": "Movie B", "description": "Romantic comedy film."},
    ]


def _make_hybrid_search(docs: list[dict]) -> HybridSearch:
    """Construct a HybridSearch with all heavy I/O mocked out."""
    mock_model = MagicMock()
    mock_index_path = MagicMock()
    mock_index_path.exists.return_value = True
    with (
        patch("cli.core.semantic_search.SentenceTransformer", return_value=mock_model),
        patch.object(ChunkedSemanticSearch, "load_or_create_chunk_embeddings"),
        patch.object(InvertedIndex, "index_path", mock_index_path),
    ):
        return HybridSearch(docs)  # type: ignore[arg-type]


class TestNormalizeScores:
    """Tests for the normalize_scores function."""

    def test_empty_list_returns_empty(self) -> None:
        """normalize_scores on an empty list returns an empty list."""
        assert normalize_scores([]) == []

    def test_single_score_returns_one(self) -> None:
        """A single score where min == max should return [1.0] (ZeroDivision guard)."""
        assert normalize_scores([0.5]) == pytest.approx([1.0])

    def test_all_equal_returns_ones(self) -> None:
        """All-equal scores should each normalise to 1.0 (ZeroDivision guard)."""
        assert normalize_scores([0.5, 0.5, 0.5]) == pytest.approx([1.0, 1.0, 1.0])

    def test_scales_min_to_zero_and_max_to_one(self) -> None:
        """Scores should be linearly scaled so min maps to 0.0 and max maps to 1.0."""
        result = normalize_scores([0.3, 0.6, 0.9])
        assert result == pytest.approx([0.0, 0.5, 1.0])


class TestHybridScore:
    """Tests for the hybrid_score function."""

    def test_equal_alpha_averages_scores(self) -> None:
        """With alpha=0.5 the result should be the average of both scores."""
        assert hybrid_score(
            bm25_score=0.4, semantic_score=0.8, alpha=0.5
        ) == pytest.approx(0.6)

    def test_alpha_one_returns_bm25_only(self) -> None:
        """With alpha=1.0 the result should equal the BM25 score."""
        assert hybrid_score(
            bm25_score=0.7, semantic_score=0.2, alpha=1.0
        ) == pytest.approx(0.7)

    def test_alpha_zero_returns_semantic_only(self) -> None:
        """With alpha=0.0 the result should equal the semantic score."""
        assert hybrid_score(
            bm25_score=0.7, semantic_score=0.2, alpha=0.0
        ) == pytest.approx(0.2)


class TestRRFScore:
    """Tests for the rrf_score function."""

    def test_default_k(self) -> None:
        """rrf_score with rank=1 and default k should return 1/(60+1)."""
        assert rrf_score(1) == pytest.approx(1 / 61)

    def test_custom_k(self) -> None:
        """rrf_score with a custom k should return 1/(k+rank)."""
        assert rrf_score(rank=5, k=10) == pytest.approx(1 / 15)


class TestHybridSearch:
    """Tests for the HybridSearch class."""

    def test_returns_same_instance_on_multiple_calls(self) -> None:
        """Two HybridSearch(documents) calls should return the identical object."""
        docs = _make_docs()
        hs1 = _make_hybrid_search(docs)
        hs2 = _make_hybrid_search(docs)
        assert hs1 is hs2

    def test_builds_index_when_cache_absent(self) -> None:
        """HybridSearch should build and save the index when no cache file exists."""
        docs = _make_docs()
        mock_model = MagicMock()
        mock_index_path = MagicMock()
        mock_index_path.exists.return_value = False
        with (
            patch(
                "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
            ),
            patch.object(ChunkedSemanticSearch, "load_or_create_chunk_embeddings"),
            patch.object(InvertedIndex, "index_path", mock_index_path),
            patch.object(InvertedIndex, "build") as mock_build,
            patch.object(InvertedIndex, "save") as mock_save,
        ):
            HybridSearch(docs)  # type: ignore[arg-type]

        mock_build.assert_called_once()
        mock_save.assert_called_once()

    def test_bm25_search_loads_and_queries_index(self) -> None:
        """_bm25_search should call load() then bm25_search() on the index."""
        docs = _make_docs()
        hs = _make_hybrid_search(docs)
        expected = [(1, 2.5), (2, 1.0)]
        with (
            patch.object(hs.idx, "load"),
            patch.object(hs.idx, "bm25_search", return_value=expected) as mock_bm25,
            patch.object(hs.semantic_search, "search_chunks", return_value=[]),
        ):
            results = hs.weighted_search("action", alpha=1.0, limit=10)

        mock_bm25.assert_called_once_with("action", 5000)
        assert results[0]["bm25_score"] == pytest.approx(1.0)

    def test_weighted_search_returns_sorted_flat_dicts(self) -> None:
        """weighted_search should return dicts sorted by descending hybrid score."""
        docs = _make_docs()
        hs = _make_hybrid_search(docs)

        bm25_results = [(1, 2.0), (2, 0.5)]
        semantic_results = [
            {"id": 1, "score": 0.9, "title": "Movie A", "document": "Action..."},
            {"id": 2, "score": 0.3, "title": "Movie B", "document": "Comedy..."},
        ]
        with (
            patch.object(hs, "_bm25_search", return_value=bm25_results),
            patch.object(
                hs.semantic_search, "search_chunks", return_value=semantic_results
            ),
        ):
            results = hs.weighted_search("action", alpha=0.5, limit=2)

        assert len(results) == 2
        assert results[0]["hybrid_score"] >= results[1]["hybrid_score"]
        required_keys = {
            "id",
            "title",
            "document",
            "bm25_score",
            "semantic_score",
            "hybrid_score",
        }
        assert required_keys.issubset(results[0].keys())

    def test_weighted_search_respects_limit(self) -> None:
        """weighted_search should return at most `limit` results."""
        docs = _make_docs()
        hs = _make_hybrid_search(docs)

        bm25_results = [(1, 2.0), (2, 1.0)]
        semantic_results = [
            {"id": 1, "score": 0.9, "title": "Movie A", "document": "..."},
            {"id": 2, "score": 0.4, "title": "Movie B", "document": "..."},
        ]
        with (
            patch.object(hs, "_bm25_search", return_value=bm25_results),
            patch.object(
                hs.semantic_search, "search_chunks", return_value=semantic_results
            ),
        ):
            results = hs.weighted_search("action", alpha=0.5, limit=1)

        assert len(results) == 1

    def test_weighted_search_combines_scores_correctly(self) -> None:
        """weighted_search should normalise both score lists before combining.

        doc 1 leads on BM25, doc 2 leads on semantic — cross-scoring ensures
        the normalization and alpha weighting are both exercised.
        norm_bm25:    doc1=1.0, doc2=0.0
        norm_semantic: doc1=0.0, doc2=1.0
        alpha=0.7 => hybrid doc1=0.7, doc2=0.3
        """
        docs = _make_docs()
        hs = _make_hybrid_search(docs)

        bm25_results = [(1, 6.0), (2, 2.0)]
        semantic_results = [
            {"id": 1, "score": 0.2, "title": "Movie A", "document": "..."},
            {"id": 2, "score": 0.8, "title": "Movie B", "document": "..."},
        ]
        with (
            patch.object(hs, "_bm25_search", return_value=bm25_results),
            patch.object(
                hs.semantic_search, "search_chunks", return_value=semantic_results
            ),
        ):
            results = hs.weighted_search("query", alpha=0.7, limit=5)

        assert results[0]["hybrid_score"] == pytest.approx(0.7)
        assert results[1]["hybrid_score"] == pytest.approx(0.3)

    def test_weighted_search_normalises_across_retrievers(self) -> None:
        """A doc absent from one retriever gets 0.0 before normalisation.

        doc 1 appears only in BM25 (semantic defaults to 0.0).
        doc 2 appears only in semantic (BM25 defaults to 0.0).
        norm_bm25:    doc1=1.0, doc2=0.0
        norm_semantic: doc1=0.0, doc2=1.0
        alpha=0.5 => both hybrids equal 0.5 (tie broken by sort stability).
        """
        docs = _make_docs()
        hs = _make_hybrid_search(docs)

        bm25_results = [(1, 5.0)]
        semantic_results = [
            {"id": 2, "score": 0.8, "title": "Movie B", "document": "..."}
        ]
        with (
            patch.object(hs, "_bm25_search", return_value=bm25_results),
            patch.object(
                hs.semantic_search, "search_chunks", return_value=semantic_results
            ),
        ):
            results = hs.weighted_search("query", alpha=0.5, limit=5)

        assert len(results) == 2
        assert results[0]["hybrid_score"] == pytest.approx(0.5)
        assert results[1]["hybrid_score"] == pytest.approx(0.5)

    def test_weighted_search_document_is_full_description(self) -> None:
        """weighted_search should return the full description untruncated."""
        docs = _make_docs()
        hs = _make_hybrid_search(docs)

        bm25_results = [(1, 1.0)]
        semantic_results = [
            {"id": 1, "score": 0.5, "title": "Movie A", "document": "..."}
        ]
        with (
            patch.object(hs, "_bm25_search", return_value=bm25_results),
            patch.object(
                hs.semantic_search, "search_chunks", return_value=semantic_results
            ),
        ):
            results = hs.weighted_search("action", alpha=0.5, limit=5)

        assert results[0]["document"] == docs[0]["description"]


class TestRRFSearch:
    """Tests for HybridSearch.rrf_search."""

    def test_returns_sorted_by_rrf_score(self) -> None:
        """rrf_search should return results sorted by descending RRF score."""
        docs = _make_docs()
        hs = _make_hybrid_search(docs)

        bm25_results = [(1, 5.0), (2, 2.0)]
        semantic_results = [
            {"id": 1, "score": 0.9, "title": "Movie A", "document": "..."},
            {"id": 2, "score": 0.3, "title": "Movie B", "document": "..."},
        ]
        with (
            patch.object(hs, "_bm25_search", return_value=bm25_results),
            patch.object(
                hs.semantic_search, "search_chunks", return_value=semantic_results
            ),
        ):
            results = hs.rrf_search("action", k=60)

        assert len(results) >= 1
        assert results[0]["rrf_score"] >= results[-1]["rrf_score"]
        required_keys = {
            "id",
            "title",
            "document",
            "bm25_rank",
            "semantic_rank",
            "rrf_score",
        }
        assert required_keys.issubset(results[0].keys())

    def test_respects_limit(self) -> None:
        """rrf_search should return at most limit results."""
        docs = _make_docs()
        hs = _make_hybrid_search(docs)

        bm25_results = [(1, 5.0), (2, 2.0)]
        semantic_results = [
            {"id": 1, "score": 0.9, "title": "Movie A", "document": "..."},
            {"id": 2, "score": 0.3, "title": "Movie B", "document": "..."},
        ]
        with (
            patch.object(hs, "_bm25_search", return_value=bm25_results),
            patch.object(
                hs.semantic_search, "search_chunks", return_value=semantic_results
            ),
        ):
            results = hs.rrf_search("action", k=60, limit=1)

        assert len(results) == 1

    def test_doc_absent_from_bm25_has_none_bm25_rank(self) -> None:
        """A doc only in semantic results should have bm25_rank=None."""
        docs = _make_docs()
        hs = _make_hybrid_search(docs)

        bm25_results = [(1, 5.0)]
        semantic_results = [
            {"id": 1, "score": 0.9, "title": "Movie A", "document": "..."},
            {"id": 2, "score": 0.5, "title": "Movie B", "document": "..."},
        ]
        with (
            patch.object(hs, "_bm25_search", return_value=bm25_results),
            patch.object(
                hs.semantic_search, "search_chunks", return_value=semantic_results
            ),
        ):
            results = hs.rrf_search("action", k=60, limit=5)

        doc2 = next(r for r in results if r["id"] == 2)
        assert doc2["bm25_rank"] is None
        assert doc2["semantic_rank"] is not None

    def test_doc_absent_from_semantic_has_none_semantic_rank(self) -> None:
        """A doc only in BM25 results should have semantic_rank=None."""
        docs = _make_docs()
        hs = _make_hybrid_search(docs)

        bm25_results = [(1, 5.0), (2, 2.0)]
        semantic_results = [
            {"id": 1, "score": 0.9, "title": "Movie A", "document": "..."},
        ]
        with (
            patch.object(hs, "_bm25_search", return_value=bm25_results),
            patch.object(
                hs.semantic_search, "search_chunks", return_value=semantic_results
            ),
        ):
            results = hs.rrf_search("action", k=60, limit=5)

        doc2 = next(r for r in results if r["id"] == 2)
        assert doc2["semantic_rank"] is None
        assert doc2["bm25_rank"] is not None

    def test_rrf_scores_are_sum_of_individual_rrf_scores(self) -> None:
        """rrf_score for a doc in both lists should be sum of its two RRF scores."""
        docs = _make_docs()
        hs = _make_hybrid_search(docs)

        bm25_results = [(1, 5.0)]
        semantic_results = [
            {"id": 1, "score": 0.9, "title": "Movie A", "document": "..."}
        ]
        k = 60
        with (
            patch.object(hs, "_bm25_search", return_value=bm25_results),
            patch.object(
                hs.semantic_search, "search_chunks", return_value=semantic_results
            ),
        ):
            results = hs.rrf_search("action", k=k, limit=5)

        expected = rrf_score(1, k=k) + rrf_score(1, k=k)
        assert results[0]["rrf_score"] == pytest.approx(expected)

    def test_document_is_full_description(self) -> None:
        """rrf_search should return the full description untruncated."""
        docs = _make_docs()
        hs = _make_hybrid_search(docs)

        bm25_results = [(1, 1.0)]
        semantic_results = [
            {"id": 1, "score": 0.5, "title": "Movie A", "document": "..."}
        ]
        with (
            patch.object(hs, "_bm25_search", return_value=bm25_results),
            patch.object(
                hs.semantic_search, "search_chunks", return_value=semantic_results
            ),
        ):
            results = hs.rrf_search("action", k=60, limit=5)

        assert results[0]["document"] == docs[0]["description"]
