"""Tests for cli.describe_image_cli."""

from unittest.mock import MagicMock, mock_open, patch

from pytest import CaptureFixture

from cli.describe_image_cli import main


def _make_response(text: str | None, total_tokens: int | None = 15) -> MagicMock:
    """Build a mock describe_image response."""
    response = MagicMock()
    response.text = text
    if total_tokens is not None:
        response.usage_metadata.total_token_count = total_tokens
    else:
        response.usage_metadata = None
    return response


def _run_main(
    argv: list[str] | None = None,
    response: MagicMock | None = None,
    mime: str | None = "image/jpeg",
) -> MagicMock:
    """Run main() with all heavy I/O mocked; return the describe_image mock."""
    default_response = response or _make_response("Paddington London")
    mock_describe = MagicMock(return_value=default_response)
    default_argv = ["describe-image", "--image", "img.jpg", "--query", "bear in London"]
    with (
        patch("sys.argv", argv or default_argv),
        patch("mimetypes.guess_type", return_value=(mime, None)),
        patch("builtins.open", mock_open(read_data=b"fake-image-bytes")),
        patch("cli.describe_image_cli.describe_image", mock_describe),
    ):
        main()
    return mock_describe


class TestDescribeImageCLI:
    """Tests for the describe_image CLI main function."""

    def test_prints_rewritten_query(self, capsys: CaptureFixture[str]) -> None:
        """main() should print the rewritten query returned by describe_image."""
        _run_main()
        assert "Rewritten query: Paddington London" in capsys.readouterr().out

    def test_prints_total_tokens(self, capsys: CaptureFixture[str]) -> None:
        """main() should print total token count when usage_metadata is present."""
        _run_main()
        out = capsys.readouterr().out
        assert "Total tokens:" in out
        assert "15" in out

    def test_empty_text_prints_empty_rewritten_query(
        self, capsys: CaptureFixture[str]
    ) -> None:
        """When response.text is None the rewritten query line should be empty."""
        _run_main(response=_make_response(None))
        assert "Rewritten query: " in capsys.readouterr().out

    def test_no_usage_metadata_skips_token_line(
        self, capsys: CaptureFixture[str]
    ) -> None:
        """When usage_metadata is None the total-tokens line should not appear."""
        _run_main(response=_make_response("result", total_tokens=None))
        assert "Total tokens:" not in capsys.readouterr().out

    def test_falls_back_to_jpeg_when_mime_unknown(self) -> None:
        """Should default to 'image/jpeg' when mimetypes cannot detect the type."""
        mock_describe = _run_main(mime=None)
        assert mock_describe.call_args[0][1] == "image/jpeg"

    def test_passes_query_to_describe_image(self) -> None:
        """The --query value should be forwarded as the third arg to describe_image."""
        mock_describe = _run_main(
            argv=["describe-image", "--image", "img.jpg", "--query", "dark knight"]
        )
        assert mock_describe.call_args[0][2] == "dark knight"
