"""Tests for cli.augmented_generation_cli."""

from unittest.mock import MagicMock, patch

import pytest
from pytest import CaptureFixture

from cli.augmented_generation_cli import main
from cli.core.hybrid_search import HybridSearch
from cli.core.keyword_search import InvertedIndex
from cli.core.semantic_search import ChunkedSemanticSearch


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset all relevant singletons before each test."""
    monkeypatch.setattr(HybridSearch, "_instance", None)
    monkeypatch.setattr(InvertedIndex, "_instance", None)
    monkeypatch.setattr(ChunkedSemanticSearch, "_instance", None)


_MOCK_DOCS = [{"id": 1, "title": "Paddington", "description": "A bear in London."}]
_MOCK_RESULTS = [
    {
        "id": 1,
        "title": "Paddington",
        "document": "A bear in London.",
        "rrf_score": 0.032,
        "bm25_rank": 1,
        "semantic_rank": 1,
    }
]


def _run_main(
    argv: list[str] | None = None,
    augment_return: str | None = "Paddington is a movie about a bear.",
) -> None:
    """Run main() with all heavy I/O mocked."""
    with (
        patch("sys.argv", argv or ["rag-cli", "rag", "bear london"]),
        patch(
            "cli.commands.search.augmented_generation_commands.load_movies",
            return_value=_MOCK_DOCS,
        ),
        patch("cli.core.semantic_search.SentenceTransformer", return_value=MagicMock()),
        patch.object(ChunkedSemanticSearch, "load_or_create_chunk_embeddings"),
        patch.object(InvertedIndex, "index_path", MagicMock(exists=lambda: True)),
        patch.object(HybridSearch, "rrf_search", return_value=_MOCK_RESULTS),
        patch(
            "cli.commands.search.augmented_generation_commands.augment_result",
            return_value=augment_return,
        ),
    ):
        main()


def test_no_command_prints_help(capsys: CaptureFixture[str]) -> None:
    """Running without a subcommand should print the help message."""
    with patch("sys.argv", ["rag-cli"]):
        main()
    assert capsys.readouterr().out != ""


@pytest.mark.parametrize(
    ("subcommand", "label"),
    [
        ("rag", "RAG Response:"),
        ("summarize", "LLM Summary:"),
        ("citations", "LLM Answer:"),
    ],
)
class TestAugmentedCommands:
    """Tests for all augmented-generation subcommands."""

    def test_prints_search_results_and_answer(
        self, capsys: CaptureFixture[str], subcommand: str, label: str
    ) -> None:
        """Command should print retrieved titles and the generated answer."""
        _run_main(argv=["rag-cli", subcommand, "bear london"])
        out = capsys.readouterr().out
        assert "Search results:" in out
        assert "Paddington" in out
        assert label in out
        assert "Paddington is a movie about a bear." in out

    def test_answer_none_prints_empty_string(
        self, capsys: CaptureFixture[str], subcommand: str, label: str
    ) -> None:
        """When augment_result returns None the response block should be empty."""
        _run_main(argv=["rag-cli", subcommand, "bear london"], augment_return=None)
        out = capsys.readouterr().out
        assert label in out
