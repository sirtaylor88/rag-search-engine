"""Tests for cli.hybrid_search_cli and the NormalizeCommand."""

from unittest.mock import patch

import pytest
from pytest import CaptureFixture

from cli.hybrid_search_cli import main


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
