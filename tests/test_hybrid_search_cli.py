"""Tests for cli.hybrid_search_cli, NormalizeCommand, and WeightedSearchCommand."""

from unittest.mock import MagicMock, patch

import pytest
from pytest import CaptureFixture

from cli.core.hybrid_search import HybridSearch
from cli.core.keyword_search import InvertedIndex
from cli.core.semantic_search import ChunkedSemanticSearch
from cli.hybrid_search_cli import main


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset all relevant singletons before each test."""
    monkeypatch.setattr(HybridSearch, "_instance", None)
    monkeypatch.setattr(InvertedIndex, "_instance", None)
    monkeypatch.setattr(ChunkedSemanticSearch, "_instance", None)


def test_normalize_command_prints_scaled_scores(capsys: CaptureFixture[str]) -> None:
    """The normalize command should print each score scaled to [0, 1]."""
    with patch("sys.argv", ["cli", "normalize", "0.3", "0.6", "0.9"]):
        main()

    out = capsys.readouterr().out
    assert "* 0.0000" in out
    assert "* 0.5000" in out
    assert "* 1.0000" in out


def test_normalize_command_single_score_prints_one(capsys: CaptureFixture[str]) -> None:
    """A single score where min == max should produce 1.0000 (ZeroDivision guard)."""
    with patch("sys.argv", ["cli", "normalize", "0.5"]):
        main()

    out = capsys.readouterr().out
    assert "* 1.0000" in out


def test_normalize_command_all_equal_scores(capsys: CaptureFixture[str]) -> None:
    """All-equal scores should each normalise to 1.0000 (ZeroDivision guard)."""
    with patch("sys.argv", ["cli", "normalize", "0.5", "0.5", "0.5"]):
        main()

    out = capsys.readouterr().out
    assert out.count("* 1.0000") == 3


def test_normalize_command_empty_scores_prints_nothing(
    capsys: CaptureFixture[str],
) -> None:
    """An empty score list should produce no output."""
    with patch("sys.argv", ["cli", "normalize"]):
        main()

    assert capsys.readouterr().out == ""


def test_normalize_command_rejects_non_positive_score() -> None:
    """A non-positive score should be rejected by ScoreListPayload validation."""
    with (
        patch("sys.argv", ["cli", "normalize", "0.5", "0.0"]),
        pytest.raises(Exception),
    ):
        main()


def test_no_command_prints_help(capsys: CaptureFixture[str]) -> None:
    """Running without a subcommand should print the help message."""
    with patch("sys.argv", ["cli"]):
        main()
    assert capsys.readouterr().out != ""


class TestWeightedSearchCommand:
    """Tests for the weighted-search subcommand in hybrid_search_cli."""

    _mock_results = [
        {
            "id": 1,
            "title": "Movie A",
            "document": "Action adventure.",
            "hybrid_score": 0.75,
            "bm25_score": 0.6,
            "semantic_score": 0.9,
        }
    ]
    _mock_docs = [{"id": 1, "title": "Movie A", "description": "Action adventure."}]

    def test_prints_results_for_query(self, capsys: CaptureFixture[str]) -> None:
        """weighted-search should print title and scores for each result."""
        mock_model = MagicMock()
        with (
            patch("sys.argv", ["cli", "weighted-search", "action"]),
            patch(
                "cli.commands.search.hybrid_search_commands.load_movies",
                return_value=self._mock_docs,
            ),
            patch(
                "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
            ),
            patch.object(ChunkedSemanticSearch, "load_or_create_chunk_embeddings"),
            patch.object(
                InvertedIndex,
                "index_path",
                MagicMock(exists=lambda: True),
            ),
            patch.object(
                HybridSearch, "weighted_search", return_value=self._mock_results
            ),
        ):
            main()

        out = capsys.readouterr().out
        assert "Movie A" in out
        assert "0.7500" in out
        assert "0.6000" in out
        assert "0.9000" in out
        assert "Action adventure." in out
        assert "..." in out

    def test_passes_custom_alpha_to_weighted_search(self) -> None:
        """weighted-search --alpha should forward the value to weighted_search."""
        mock_model = MagicMock()
        with (
            patch("sys.argv", ["cli", "weighted-search", "action", "--alpha", "0.3"]),
            patch(
                "cli.commands.search.hybrid_search_commands.load_movies",
                return_value=self._mock_docs,
            ),
            patch(
                "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
            ),
            patch.object(ChunkedSemanticSearch, "load_or_create_chunk_embeddings"),
            patch.object(
                InvertedIndex,
                "index_path",
                MagicMock(exists=lambda: True),
            ),
            patch.object(
                HybridSearch, "weighted_search", return_value=self._mock_results
            ) as mock_ws,
        ):
            main()

        call_args, _ = mock_ws.call_args
        assert call_args[1] == pytest.approx(0.3)

    def test_empty_results_prints_only_banner(
        self, capsys: CaptureFixture[str]
    ) -> None:
        """weighted-search with no results should print the banner but no entries."""
        mock_model = MagicMock()
        with (
            patch("sys.argv", ["cli", "weighted-search", "xyzzy"]),
            patch(
                "cli.commands.search.hybrid_search_commands.load_movies",
                return_value=self._mock_docs,
            ),
            patch(
                "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
            ),
            patch.object(ChunkedSemanticSearch, "load_or_create_chunk_embeddings"),
            patch.object(
                InvertedIndex,
                "index_path",
                MagicMock(exists=lambda: True),
            ),
            patch.object(HybridSearch, "weighted_search", return_value=[]),
        ):
            main()

        out = capsys.readouterr().out
        assert "xyzzy" in out


class TestRRFSearchCommand:
    """Tests for the rrf-search subcommand in hybrid_search_cli."""

    _mock_results = [
        {
            "id": 1,
            "title": "Movie A",
            "document": "Action adventure.",
            "rrf_score": 0.032,
            "bm25_rank": 1,
            "semantic_rank": 2,
        }
    ]
    _mock_docs = [{"id": 1, "title": "Movie A", "description": "Action adventure."}]

    def test_prints_results_for_query(self, capsys: CaptureFixture[str]) -> None:
        """rrf-search should print title, RRF score, and ranks for each result."""
        mock_model = MagicMock()
        with (
            patch("sys.argv", ["cli", "rrf-search", "action"]),
            patch(
                "cli.commands.search.hybrid_search_commands.load_movies",
                return_value=self._mock_docs,
            ),
            patch(
                "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
            ),
            patch.object(ChunkedSemanticSearch, "load_or_create_chunk_embeddings"),
            patch.object(
                InvertedIndex,
                "index_path",
                MagicMock(exists=lambda: True),
            ),
            patch.object(HybridSearch, "rrf_search", return_value=self._mock_results),
        ):
            main()

        out = capsys.readouterr().out
        assert "Movie A" in out
        assert "0.032" in out
        assert "BM25 Rank: 1" in out
        assert "Semantic Rank: 2" in out
        assert "Action adventure." in out

    def test_passes_custom_k_to_rrf_search(self) -> None:
        """rrf-search --k should forward the value to rrf_search."""
        mock_model = MagicMock()
        with (
            patch("sys.argv", ["cli", "rrf-search", "action", "--k", "30"]),
            patch(
                "cli.commands.search.hybrid_search_commands.load_movies",
                return_value=self._mock_docs,
            ),
            patch(
                "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
            ),
            patch.object(ChunkedSemanticSearch, "load_or_create_chunk_embeddings"),
            patch.object(
                InvertedIndex,
                "index_path",
                MagicMock(exists=lambda: True),
            ),
            patch.object(
                HybridSearch, "rrf_search", return_value=self._mock_results
            ) as mock_rrf,
        ):
            main()

        call_args, _ = mock_rrf.call_args
        assert call_args[1] == 30

    def test_enhance_flag_calls_enhance_query(self) -> None:
        """rrf-search --enhance spell should call enhance_query before searching."""
        mock_model = MagicMock()
        with (
            patch("sys.argv", ["cli", "rrf-search", "acton", "--enhance", "spell"]),
            patch(
                "cli.commands.search.hybrid_search_commands.load_movies",
                return_value=self._mock_docs,
            ),
            patch(
                "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
            ),
            patch.object(ChunkedSemanticSearch, "load_or_create_chunk_embeddings"),
            patch.object(
                InvertedIndex,
                "index_path",
                MagicMock(exists=lambda: True),
            ),
            patch.object(HybridSearch, "rrf_search", return_value=self._mock_results),
            patch(
                "cli.commands.search.hybrid_search_commands.enhance_query",
                return_value="action",
            ) as mock_enhance,
        ):
            main()

        mock_enhance.assert_called_once_with("acton", method="spell")
