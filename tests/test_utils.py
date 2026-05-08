"""Tests for cli.utils."""

from pathlib import Path

import pytest

from cli.utils import (
    get_stemmed_tokens,
    get_stop_words,
    get_term_token,
    remove_all_punctuations,
    tokenize_text,
)


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
        ("running", ["run"]),
        ("Batman & Robin", ["batman", "robin"]),
        ("", []),
    ],
)
def test_get_stemmed_tokens(text: str, expected: list[str]) -> None:
    """Tokens should be reduced to their Porter stems in input order."""
    assert get_stemmed_tokens(text) == expected


def test_get_term_token_returns_stem() -> None:
    """get_term_token should return the Porter stem of a single-word term."""
    assert get_term_token("running") == "run"


def test_get_term_token_raises_for_multi_word() -> None:
    """get_term_token should raise ValueError for a multi-word term."""
    with pytest.raises(ValueError, match="unique"):
        get_term_token("batman begins")


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
