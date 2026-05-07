"""Tests for cli.utils."""

import json
import logging
from pathlib import Path

import pytest

from cli.utils import (
    get_movies,
    get_stemmed_tokens,
    get_stop_words,
    remove_all_punctuations,
    tokenize_text,
)


def test_get_movies(tmp_path: Path) -> None:
    """get_movies should return the list of movies from a JSON file."""
    movies = [{"id": 1, "title": "Test Movie", "description": ""}]
    path = tmp_path / "movies.json"
    path.write_text(json.dumps({"movies": movies}), encoding="utf-8")
    assert get_movies(str(path)) == movies


def test_get_movies_missing_file_logs_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """get_movies should log an error when the file cannot be opened."""
    with pytest.raises(Exception), caplog.at_level(logging.ERROR, logger="cli.utils"):
        get_movies(str(tmp_path / "nonexistent.json"))
    assert "Failed to read" in caplog.text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello, World!", "Hello World"),
        ("Hello World", "Hello World"),
        ("", ""),
        ("!@#$%^&*()", ""),
    ],
)
def test_remove_all_punctuations(text: str, expected: str) -> None:
    """Punctuation characters should be stripped; non-punctuation left intact."""
    assert remove_all_punctuations(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello, World!", ["hello", "world"]),
        ("Batman & Robin", ["batman", "robin"]),
        ("  extra  spaces  ", ["extra", "spaces"]),
        ("", []),
    ],
)
def test_tokenize_text(text: str, expected: list[str]) -> None:
    """Text should be lowercased, punctuation stripped, and split into tokens."""
    assert tokenize_text(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("running", {"run"}),
        ("Batman & Robin", {"batman", "robin"}),
        ("", set()),
    ],
)
def test_get_stemmed_tokens(text: str, expected: set[str]) -> None:
    """Tokens should be reduced to their Porter stems and returned as a set."""
    assert get_stemmed_tokens(text) == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("a\nthe\nis\n", ["a", "the", "is"]),
        ("", []),
        ("word\n", ["word"]),
    ],
)
def test_get_stop_words(tmp_path: Path, content: str, expected: list[str]) -> None:
    """Stop words file should be read into a list, one word per line."""
    path = tmp_path / "stopwords.txt"
    path.write_text(content, encoding="utf-8")
    assert get_stop_words(str(path)) == expected
