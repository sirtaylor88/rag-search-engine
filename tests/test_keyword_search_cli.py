"""Tests for cli.keyword_search_cli."""

# pylint: disable=redefined-outer-name

from typing import Any

import pytest
from pytest import CaptureFixture

from cli.keyword_search_cli import display_five_best_results


@pytest.fixture()
def movies() -> list[dict[str, Any]]:
    """Small movie list for search tests."""
    return [
        {"id": i, "title": title, "description": ""}
        for i, title in enumerate(
            [
                "The Dark Knight",
                "Batman Begins",
                "Batman Forever",
                "Batman Returns",
                "Batman & Robin",
                "Batman v Superman",
                "Inception",
            ],
            start=1,
        )
    ]


@pytest.mark.parametrize("query", ["batman", "BATMAN"])
def test_returns_matching_results(
    movies: list[dict[str, Any]], capsys: CaptureFixture[str], query: str
) -> None:
    """Only matching titles should appear, capped at five, case-insensitively."""
    display_five_best_results(query, movies)
    output = capsys.readouterr().out.splitlines()
    assert len(output) == 5
    assert all("batman" in line.lower() for line in output)


def test_no_matches_prints_nothing(
    movies: list[dict[str, Any]], capsys: CaptureFixture[str]
) -> None:
    """No output should be produced when the query matches no titles."""
    display_five_best_results("nomatch", movies)
    output = capsys.readouterr().out
    assert output == ""


def test_fewer_than_five_matches(
    movies: list[dict[str, Any]], capsys: CaptureFixture[str]
) -> None:
    """All matches should be shown when there are fewer than five."""
    display_five_best_results("inception", movies)
    output = capsys.readouterr().out.splitlines()
    assert len(output) == 1
    assert "Inception" in output[0]


def test_result_lines_are_numbered(
    movies: list[dict[str, Any]], capsys: CaptureFixture[str]
) -> None:
    """Each result line should be prefixed with its 1-based position number."""
    display_five_best_results("batman", movies)
    output = capsys.readouterr().out.splitlines()
    for i, line in enumerate(output, start=1):
        assert line.startswith(f"{i}.")
