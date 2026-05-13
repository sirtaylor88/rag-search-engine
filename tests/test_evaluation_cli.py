"""Tests for cli.evaluation_cli."""

import json
from unittest.mock import MagicMock, mock_open, patch

import pytest
from pytest import CaptureFixture

from cli.core.hybrid_search import HybridSearch
from cli.core.keyword_search import InvertedIndex
from cli.core.semantic_search import ChunkedSemanticSearch
from cli.evaluation_cli import main


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset all relevant singletons before each test."""
    monkeypatch.setattr(HybridSearch, "_instance", None)
    monkeypatch.setattr(InvertedIndex, "_instance", None)
    monkeypatch.setattr(ChunkedSemanticSearch, "_instance", None)


_MOCK_DOCS = [{"id": 1, "title": "Ted", "description": "A talking bear."}]

_GOLDEN_DATA = {
    "test_cases": [
        {"query": "talking bear", "relevant_docs": ["Ted"]},
    ]
}

_MOCK_RESULTS = [
    {"id": 1, "title": "Ted", "rrf_score": 0.032, "bm25_rank": 1, "semantic_rank": 1}
]


def _run_main(argv: list[str] | None = None) -> MagicMock:
    """Run main() with all heavy I/O mocked; return the rrf_search mock."""
    golden_json = json.dumps(_GOLDEN_DATA)
    mock_rrf = MagicMock(return_value=_MOCK_RESULTS)
    with (
        patch("sys.argv", argv or ["eval"]),
        patch("cli.evaluation_cli.load_movies", return_value=_MOCK_DOCS),
        patch("cli.core.semantic_search.SentenceTransformer", return_value=MagicMock()),
        patch.object(ChunkedSemanticSearch, "load_or_create_chunk_embeddings"),
        patch.object(InvertedIndex, "index_path", MagicMock(exists=lambda: True)),
        patch.object(HybridSearch, "rrf_search", mock_rrf),
        patch("builtins.open", mock_open(read_data=golden_json)),
    ):
        main()
    return mock_rrf


class TestEvaluationCLI:
    """Tests for the evaluation CLI main function."""

    def test_prints_query_precision_and_titles(
        self, capsys: CaptureFixture[str]
    ) -> None:
        """main() should print the query, Precision@k, and titles."""
        _run_main()
        out = capsys.readouterr().out
        assert "k=5 (Top results)" in out
        assert "talking bear" in out
        assert "Precision@5: 1.0000" in out
        assert "Recall@5: 1.0000" in out
        assert "F1 Score: 1.0000" in out
        assert "Ted" in out

    def test_custom_limit_forwarded_to_rrf_search(self) -> None:
        """--limit N should be passed as limit=N to rrf_search."""
        mock_rrf = _run_main(argv=["eval", "--limit", "3"])
        _, kwargs = mock_rrf.call_args
        assert kwargs["limit"] == 3

    def test_empty_retrieved_prints_all_zeros_without_error(
        self, capsys: CaptureFixture[str]
    ) -> None:
        """rrf_search returning [] should print zeros for all metrics, not crash."""
        with (
            patch("sys.argv", ["eval"]),
            patch("cli.evaluation_cli.load_movies", return_value=_MOCK_DOCS),
            patch(
                "cli.core.semantic_search.SentenceTransformer",
                return_value=MagicMock(),
            ),
            patch.object(ChunkedSemanticSearch, "load_or_create_chunk_embeddings"),
            patch.object(InvertedIndex, "index_path", MagicMock(exists=lambda: True)),
            patch.object(HybridSearch, "rrf_search", return_value=[]),
            patch(
                "builtins.open",
                mock_open(read_data=json.dumps(_GOLDEN_DATA)),
            ),
        ):
            main()
        out = capsys.readouterr().out
        assert "Precision@5: 0.0000" in out
        assert "Recall@5: 0.0000" in out
        assert "F1 Score: 0.0000" in out

    def test_zero_hit_query_prints_f1_zero_without_error(
        self, capsys: CaptureFixture[str]
    ) -> None:
        """A query with no relevant hits should print F1 0.0000, not crash."""
        golden_json = json.dumps(
            {"test_cases": [{"query": "no hits", "relevant_docs": ["Missing"]}]}
        )
        no_hit_results = [
            {
                "id": 99,
                "title": "Unrelated",
                "rrf_score": 0.01,
                "bm25_rank": 1,
                "semantic_rank": 1,
            }
        ]
        with (
            patch("sys.argv", ["eval"]),
            patch("cli.evaluation_cli.load_movies", return_value=_MOCK_DOCS),
            patch(
                "cli.core.semantic_search.SentenceTransformer",
                return_value=MagicMock(),
            ),
            patch.object(ChunkedSemanticSearch, "load_or_create_chunk_embeddings"),
            patch.object(InvertedIndex, "index_path", MagicMock(exists=lambda: True)),
            patch.object(HybridSearch, "rrf_search", return_value=no_hit_results),
            patch("builtins.open", mock_open(read_data=golden_json)),
        ):
            main()
        out = capsys.readouterr().out
        assert "F1 Score: 0.0000" in out
