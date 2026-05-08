"""Tests for cli.semantic_search_cli and semantic search commands."""

from unittest.mock import MagicMock, patch

from pytest import CaptureFixture

from cli.semantic_search_cli import main


def test_verify_command_calls_verify_model(capsys: CaptureFixture[str]) -> None:
    """The verify command should call verify_model and print its output."""
    mock_model = MagicMock()
    mock_model.max_seq_length = 128
    with (
        patch("sys.argv", ["cli", "verify"]),
        patch("sentence_transformers.SentenceTransformer", return_value=mock_model),
    ):
        main()

    out = capsys.readouterr().out
    assert "Model loaded:" in out
    assert "Max sequence length:" in out


def test_embed_text_command_prints_embedding_info(capsys: CaptureFixture[str]) -> None:
    """The embed_text command should encode the text and print embedding info."""
    mock_embedding = MagicMock()
    mock_embedding.__getitem__ = MagicMock(return_value="[0.1, 0.2, 0.3]")
    mock_embedding.shape = [384]
    mock_model = MagicMock()
    mock_model.encode.return_value = [mock_embedding]
    with (
        patch("sys.argv", ["cli", "embed_text", "hello world"]),
        patch("sentence_transformers.SentenceTransformer", return_value=mock_model),
    ):
        main()

    out = capsys.readouterr().out
    assert "Text: hello world" in out
    assert "First 3 dimensions:" in out
    assert "Dimensions:" in out


def test_no_command_prints_help(capsys: CaptureFixture[str]) -> None:
    """Running without a subcommand should print the help message."""
    with patch("sys.argv", ["cli"]):
        main()
    assert capsys.readouterr().out != ""
