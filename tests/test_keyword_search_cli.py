"""Tests for cli.keyword_search_cli."""

# pylint: disable=redefined-outer-name

from unittest.mock import patch

import pytest
from pytest import CaptureFixture

from cli.inverted_index import InvertedIndex
from cli.keyword_search_cli import display_five_best_results, main


@pytest.fixture()
def inverted_index() -> InvertedIndex:
    """InvertedIndex built from a small movie list for search tests."""
    movies = [
        {"id": i, "title": title, "description": ""}
        for i, title in enumerate(
            [
                "Batman Begins",
                "Batman Forever",
                "Batman Returns",
                "Batman & Robin",
                "Batman v Superman",
                "The Dark Knight",
                "Inception",
            ],
            start=1,
        )
    ]
    idx = InvertedIndex()
    idx.build(movies)
    return idx


@pytest.mark.parametrize("query", ["batman", "BATMAN"])
def test_returns_matching_results(
    inverted_index: InvertedIndex, capsys: CaptureFixture[str], query: str
) -> None:
    """Only matching titles should appear, capped at five, case-insensitively."""
    display_five_best_results(query, inverted_index)
    output = capsys.readouterr().out.splitlines()
    assert len(output) == 5
    assert all("batman" in line.lower() for line in output)


def test_no_matches_prints_nothing(
    inverted_index: InvertedIndex, capsys: CaptureFixture[str]
) -> None:
    """No output should be produced when the query matches no titles."""
    display_five_best_results("nomatch", inverted_index)
    output = capsys.readouterr().out
    assert output == ""


def test_fewer_than_five_matches(
    inverted_index: InvertedIndex, capsys: CaptureFixture[str]
) -> None:
    """All matches should be shown when there are fewer than five."""
    display_five_best_results("inception", inverted_index)
    output = capsys.readouterr().out.splitlines()
    assert len(output) == 1
    assert "Inception" in output[0]


def test_result_lines_are_numbered(
    inverted_index: InvertedIndex, capsys: CaptureFixture[str]
) -> None:
    """Each result line should be prefixed with the movie's doc ID."""
    display_five_best_results("batman", inverted_index)
    output = capsys.readouterr().out.splitlines()
    for i, line in enumerate(output, start=1):
        assert line.startswith(f"{i}.")


def test_search_missing_cache_prints_error_and_exits(
    capsys: CaptureFixture[str],
) -> None:
    """A missing cache file should print an error message and exit with code 1."""
    with (
        patch("cli.keyword_search_cli.InvertedIndex.load", side_effect=OSError),
        patch("sys.argv", ["cli", "search", "batman"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    assert capsys.readouterr().out == "Cannot load movies data.\n"
