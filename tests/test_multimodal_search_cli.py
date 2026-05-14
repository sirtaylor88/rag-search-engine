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


class TestMultimodalSearchCLI:
    """Tests for the multimodal search CLI main function."""

    def test_verify_image_embedding_dispatches_with_path(self) -> None:
        """verify_image_embedding subcommand should forward the image path."""
        mock_verify = MagicMock()
        with (
            patch(
                "sys.argv",
                ["multimodal-cli", "verify_image_embedding", "poster.jpg"],
            ),
            patch(
                "cli.commands.search.multimodal_search_commands.verify_image_embedding",
                mock_verify,
            ),
        ):
            main()

        mock_verify.assert_called_once_with("poster.jpg")

    def test_image_search_dispatches_with_path(
        self, capsys: CaptureFixture[str]
    ) -> None:
        """image_search subcommand should call search_with_image and print results."""
        results = [{"title": "Movie A", "score": 0.9, "document": "desc"}]
        with (
            patch("sys.argv", ["multimodal-cli", "image_search", "poster.jpg"]),
            patch(
                "cli.commands.search.multimodal_search_commands.load_movies",
                return_value=[],
            ),
            patch(
                "cli.commands.search.multimodal_search_commands.MultimodalSearch"
            ) as mock_ms,
        ):
            mock_ms.return_value.search_with_image.return_value = results
            main()

        out = capsys.readouterr().out
        assert "Movie A" in out

    def test_no_command_prints_help(self, capsys: CaptureFixture[str]) -> None:
        """Running without a subcommand should print the help message."""
        with patch("sys.argv", ["multimodal-cli"]):
            main()
        assert capsys.readouterr().out != ""
