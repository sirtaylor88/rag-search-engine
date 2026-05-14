"""Tests for cli.hybrid_search_cli, NormalizeCommand, and WeightedSearchCommand."""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
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

    @pytest.mark.parametrize(
        ("query", "method", "enhanced"),
        [
            ("acton", "spell", "action"),
            ("bear moovie", "rewrite", "bear movie"),
        ],
    )
    def test_enhance_flag_calls_enhance_query(
        self,
        capsys: CaptureFixture[str],
        query: str,
        method: str,
        enhanced: str,
    ) -> None:
        """rrf-search --enhance should call enhance_query with the given method."""
        mock_model = MagicMock()
        with (
            patch("sys.argv", ["cli", "rrf-search", query, "--enhance", method]),
            patch(
                "cli.commands.search.hybrid_search_commands.load_movies",
                return_value=self._mock_docs,
            ),
            patch(
                "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
            ),
            patch.object(ChunkedSemanticSearch, "load_or_create_chunk_embeddings"),
            patch.object(InvertedIndex, "index_path", MagicMock(exists=lambda: True)),
            patch.object(HybridSearch, "rrf_search", return_value=self._mock_results),
            patch(
                "cli.commands.search.hybrid_search_commands.enhance_query",
                return_value=enhanced,
            ) as mock_enhance,
        ):
            main()

        mock_enhance.assert_called_once_with(query, method=method)
        assert f'Results for: "{enhanced}"' in capsys.readouterr().out

    def test_rerank_method_calls_rerank_query_and_prints_score(
        self, capsys: CaptureFixture[str]
    ) -> None:
        """--rerank-method individual should call rerank_query and show the score."""
        mock_model = MagicMock()
        with (
            patch(
                "sys.argv",
                ["cli", "rrf-search", "action", "--rerank-method", "individual"],
            ),
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
                "cli.commands.search.hybrid_search_commands.rerank_query",
                return_value="8.5",
            ) as mock_rerank,
            patch("cli.commands.search.hybrid_search_commands.sleep"),
        ):
            main()

        mock_rerank.assert_called_once()
        out = capsys.readouterr().out
        assert "Re-rank Score" in out
        assert "8.500" in out

    def test_batch_rerank_method_calls_rerank_query_once_and_prints_rank(
        self, capsys: CaptureFixture[str]
    ) -> None:
        """--rerank-method batch should call rerank_query once and show rank."""
        mock_model = MagicMock()
        with (
            patch(
                "sys.argv",
                ["cli", "rrf-search", "action", "--rerank-method", "batch"],
            ),
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
                "cli.commands.search.hybrid_search_commands.rerank_query",
                return_value="[1]",
            ) as mock_rerank,
        ):
            main()

        mock_rerank.assert_called_once()
        out = capsys.readouterr().out
        assert "Re-rank Rank" in out

    def test_batch_rerank_empty_response_preserves_original_order(
        self, capsys: CaptureFixture[str]
    ) -> None:
        """Batch rerank with empty response should keep original RRF order."""
        mock_model = MagicMock()
        with (
            patch(
                "sys.argv",
                ["cli", "rrf-search", "action", "--rerank-method", "batch"],
            ),
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
                "cli.commands.search.hybrid_search_commands.rerank_query",
                return_value="",
            ),
        ):
            main()

        out = capsys.readouterr().out
        assert "Movie A" in out

    def test_cross_encoder_rerank_calls_predict_and_prints_score(
        self, capsys: CaptureFixture[str]
    ) -> None:
        """--rerank-method cross_encoder should call predict and show the score."""
        mock_model = MagicMock()
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.return_value = np.array([7.2])
        with (
            patch(
                "sys.argv",
                ["cli", "rrf-search", "action", "--rerank-method", "cross_encoder"],
            ),
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
                "cli.commands.search.hybrid_search_commands.CrossEncoder",
                return_value=mock_cross_encoder,
            ),
        ):
            main()

        mock_cross_encoder.predict.assert_called_once()
        out = capsys.readouterr().out
        assert "Cross Encoder Score" in out

    def test_verbose_flag_emits_debug_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """--verbose should emit DEBUG-level logs for query and RRF results."""
        mock_model = MagicMock()
        module = "cli.commands.search.hybrid_search_commands"
        with (
            caplog.at_level(logging.DEBUG, logger=module),
            patch("sys.argv", ["cli", "rrf-search", "action", "--verbose"]),
            patch(
                "cli.commands.search.hybrid_search_commands.load_movies",
                return_value=self._mock_docs,
            ),
            patch(
                "cli.core.semantic_search.SentenceTransformer",
                return_value=mock_model,
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

        messages = [r.message for r in caplog.records]
        assert any("Original query" in m for m in messages)
        assert any("RRF results" in m for m in messages)

    def test_rerank_method_prints_reranking_banner(
        self, capsys: CaptureFixture[str]
    ) -> None:
        """Any --rerank-method should print the re-ranking banner, not plain banner."""
        mock_model = MagicMock()
        with (
            patch(
                "sys.argv",
                ["cli", "rrf-search", "action", "--rerank-method", "individual"],
            ),
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
                "cli.commands.search.hybrid_search_commands.rerank_query",
                return_value="8.0",
            ),
            patch("cli.commands.search.hybrid_search_commands.sleep"),
        ):
            main()

        out = capsys.readouterr().out
        assert "Re-ranking top" in out
        assert "individual method" in out

    def test_evaluate_flag_calls_evaluate_result_and_prints_scores(
        self, capsys: CaptureFixture[str]
    ) -> None:
        """--evaluate should call evaluate_result and print per-result scores."""
        mock_model = MagicMock()
        with (
            patch("sys.argv", ["cli", "rrf-search", "action", "--evaluate"]),
            patch(
                "cli.commands.search.hybrid_search_commands.load_movies",
                return_value=self._mock_docs,
            ),
            patch(
                "cli.core.semantic_search.SentenceTransformer",
                return_value=mock_model,
            ),
            patch.object(ChunkedSemanticSearch, "load_or_create_chunk_embeddings"),
            patch.object(
                InvertedIndex,
                "index_path",
                MagicMock(exists=lambda: True),
            ),
            patch.object(HybridSearch, "rrf_search", return_value=self._mock_results),
            patch(
                "cli.commands.search.hybrid_search_commands.evaluate_result",
                return_value="[3]",
            ) as mock_evaluate,
        ):
            main()

        mock_evaluate.assert_called_once()
        out = capsys.readouterr().out
        assert "Movie A: 3/3" in out
