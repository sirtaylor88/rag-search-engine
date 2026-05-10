"""Tests for cli.semantic_search_cli and semantic search commands."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest import CaptureFixture

from cli.core.semantic_search import ChunkedSemanticSearch, SemanticSearch
from cli.semantic_search_cli import main


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset SemanticSearch and ChunkedSemanticSearch singletons before each test."""
    monkeypatch.setattr(SemanticSearch, "_instance", None)
    monkeypatch.setattr(ChunkedSemanticSearch, "_instance", None)


def test_verify_command_calls_verify_model(capsys: CaptureFixture[str]) -> None:
    """The verify command should call verify_model and print its output."""
    mock_model = MagicMock()
    mock_model.max_seq_length = 128
    with (
        patch("sys.argv", ["cli", "verify"]),
        patch("cli.core.semantic_search.SentenceTransformer", return_value=mock_model),
    ):
        main()

    out = capsys.readouterr().out
    assert "Model loaded:" in out
    assert "Max sequence length:" in out


def test_verify_embeddings_command_prints_shape(capsys: CaptureFixture[str]) -> None:
    """The verify_embeddings command should print document count and embedding shape."""
    mock_embeddings = MagicMock()
    mock_embeddings.shape = (2, 384)
    mock_docs = [
        {"id": 1, "title": "A", "description": "desc A"},
        {"id": 2, "title": "B", "description": "desc B"},
    ]
    mock_model = MagicMock()
    with (
        patch("sys.argv", ["cli", "verify_embeddings"]),
        patch("cli.core.semantic_search.SentenceTransformer", return_value=mock_model),
        patch("cli.core.semantic_search.get_movies", return_value=mock_docs),
        patch.object(
            SemanticSearch,
            "load_or_create_embeddings",
            return_value=mock_embeddings,
        ),
    ):
        main()

    out = capsys.readouterr().out
    assert "Number of docs:" in out
    assert "Embeddings shape:" in out


def test_embed_text_command_prints_embedding_info(capsys: CaptureFixture[str]) -> None:
    """The embed_text command should encode the text and print embedding info."""
    mock_embedding = MagicMock()
    mock_embedding.__getitem__ = MagicMock(return_value="[0.1, 0.2, 0.3]")
    mock_embedding.shape = [384]
    mock_model = MagicMock()
    mock_model.encode.return_value = [mock_embedding]
    with (
        patch("sys.argv", ["cli", "embed_text", "hello world"]),
        patch("cli.core.semantic_search.SentenceTransformer", return_value=mock_model),
    ):
        main()

    out = capsys.readouterr().out
    assert "Text: hello world" in out
    assert "First 3 dimensions:" in out
    assert "Dimensions:" in out


def test_embed_query_command_prints_embedding_info(capsys: CaptureFixture[str]) -> None:
    """The embed_query command should encode the query and print its embedding info."""
    mock_embedding = MagicMock()
    mock_embedding.__getitem__ = MagicMock(return_value="[0.1, 0.2, 0.3]")
    mock_embedding.shape = (384,)
    mock_model = MagicMock()
    mock_model.encode.return_value = [mock_embedding]
    with (
        patch("sys.argv", ["cli", "embed_query", "dark knight"]),
        patch("cli.core.semantic_search.SentenceTransformer", return_value=mock_model),
    ):
        main()

    out = capsys.readouterr().out
    assert "Query: dark knight" in out
    assert "First 3 dimensions:" in out
    assert "Shape:" in out


def test_search_command_prints_ranked_results(capsys: CaptureFixture[str]) -> None:
    """The search command should print ranked results with scores and descriptions."""
    docs = [{"id": 1, "title": "The Dark Knight", "description": "Batman fights Joker"}]
    mock_model = MagicMock()
    query_emb = np.array([1.0, 0.0])
    doc_emb = np.array([1.0, 0.0])
    mock_model.encode.return_value = [query_emb]
    with (
        patch("sys.argv", ["cli", "search", "batman"]),
        patch("cli.core.semantic_search.SentenceTransformer", return_value=mock_model),
        patch(
            "cli.commands.search.semantic_search_command.get_movies", return_value=docs
        ),
        patch.object(
            SemanticSearch,
            "load_or_create_embeddings",
            return_value=np.array([doc_emb]),
        ),
        patch.object(
            SemanticSearch,
            "search",
            return_value=[
                {
                    "title": "The Dark Knight",
                    "score": 1.0,
                    "description": "Batman fights Joker",
                }
            ],
        ),
    ):
        main()

    out = capsys.readouterr().out
    assert "Searching for: batman" in out
    assert "The Dark Knight" in out
    assert "score:" in out


def test_chunk_command_prints_chunks(capsys: CaptureFixture[str]) -> None:
    """The chunk command should split the text and print each chunk with its index."""
    with patch("sys.argv", ["cli", "chunk", "the dark knight rises"]):
        main()

    out = capsys.readouterr().out
    assert "Chunking" in out
    assert "1." in out


def test_chunk_command_with_overlap_prints_multiple_chunks(
    capsys: CaptureFixture[str],
) -> None:
    """The chunk command with --overlap should produce overlapping chunks."""
    with patch(
        "sys.argv",
        [
            "cli",
            "chunk",
            "The bear attack was very terrifying",
            "--chunk-size",
            "4",
            "--overlap",
            "1",
        ],
    ):
        main()

    out = capsys.readouterr().out
    assert "1." in out
    assert "2." in out


def test_semantic_chunk_command_prints_chunks(capsys: CaptureFixture[str]) -> None:
    """The semantic_chunk command should split text into chunks and print each one."""
    text = "This is sentence one. This is sentence two. This is sentence three."
    with patch(
        "sys.argv",
        ["cli", "semantic_chunk", text, "--max-chunk-size", "2"],
    ):
        main()

    out = capsys.readouterr().out
    assert "Semantically chunking" in out
    assert "1." in out
    assert "2." in out


def test_semantic_chunk_command_with_overlap(capsys: CaptureFixture[str]) -> None:
    """The semantic_chunk command with --overlap should produce overlapping chunks."""
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    with patch(
        "sys.argv",
        ["cli", "semantic_chunk", text, "--max-chunk-size", "2", "--overlap", "1"],
    ):
        main()

    out = capsys.readouterr().out
    assert "1." in out
    assert "2." in out
    assert "3." in out


def test_embed_chunks_command_prints_chunk_count(capsys: CaptureFixture[str]) -> None:
    """The embed_chunks command should print the number of chunk embeddings."""
    mock_embeddings = np.zeros((5, 3))
    mock_docs = [{"id": 1, "title": "A", "description": "desc"}]
    mock_model = MagicMock()
    with (
        patch("sys.argv", ["cli", "embed_chunks"]),
        patch("cli.core.semantic_search.SentenceTransformer", return_value=mock_model),
        patch("cli.core.semantic_search.get_movies", return_value=mock_docs),
        patch.object(
            ChunkedSemanticSearch,
            "load_or_create_chunk_embeddings",
            return_value=mock_embeddings,
        ),
    ):
        main()

    out = capsys.readouterr().out
    assert "5" in out
    assert "chunked embeddings" in out


def test_no_command_prints_help(capsys: CaptureFixture[str]) -> None:
    """Running without a subcommand should print the help message."""
    with patch("sys.argv", ["cli"]):
        main()
    assert capsys.readouterr().out != ""
