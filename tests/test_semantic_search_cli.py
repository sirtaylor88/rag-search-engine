"""Tests for cli.semantic_search_cli and cli.commands.verify_command."""

from unittest.mock import MagicMock, patch

from pytest import CaptureFixture

from cli.semantic_search_cli import main


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


def test_no_command_prints_help(capsys: CaptureFixture[str]) -> None:
    """Running without a subcommand should print the help message."""
    with patch("sys.argv", ["cli"]):
        main()
    assert capsys.readouterr().out != ""
