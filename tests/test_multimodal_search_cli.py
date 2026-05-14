"""Tests for cli.multimodal_search_cli."""

from unittest.mock import MagicMock, patch

import pytest
from pytest import CaptureFixture

from cli.core.multimodal_search import MultimodalSearch
from cli.multimodal_search_cli import main


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the MultimodalSearch singleton before each test."""
    monkeypatch.setattr(MultimodalSearch, "_instance", None)


def _run_main(argv: list[str] | None = None) -> MagicMock:
    """Run main() with I/O mocked; return the verify_image_embedding mock."""
    mock_verify = MagicMock()
    with (
        patch(
            "sys.argv",
            argv or ["multimodal-cli", "verify_image_embedding", "img.jpg"],
        ),
        patch(
            "cli.commands.verify_commands.verify_image_embedding",
            mock_verify,
        ),
    ):
        main()
    return mock_verify


class TestMultimodalSearchCLI:
    """Tests for the multimodal search CLI main function."""

    def test_verify_image_embedding_dispatches_with_path(self) -> None:
        """verify_image_embedding subcommand should forward the image path."""
        mock_verify = _run_main(
            ["multimodal-cli", "verify_image_embedding", "poster.jpg"]
        )
        mock_verify.assert_called_once_with("poster.jpg")

    def test_no_command_prints_help(self, capsys: CaptureFixture[str]) -> None:
        """Running without a subcommand should print the help message."""
        with patch("sys.argv", ["multimodal-cli"]):
            main()
        assert capsys.readouterr().out != ""
