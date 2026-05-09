"""Tests for cli.utils."""

from pathlib import Path

import pytest

from pytest import CaptureFixture

import numpy as np

from cli.utils import (
    cosine_similarity,
    get_overlapping_chunks,
    get_stemmed_tokens,
    get_stop_words,
    get_term_token,
    remove_all_punctuations,
    timer,
    tokenize_text,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello, World!", "Hello World"),
        ("Hello World", "Hello World"),
        ("", ""),
        ("!@#$%^&*()", ""),
        ("‘it’s a “test”", "its a test"),
        ("en–dash and em—dash", "endash and emdash"),
    ],
)
def test_remove_all_punctuations(text: str, expected: str) -> None:
    """ASCII and Unicode punctuation should be stripped; other chars left intact."""
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


def test_get_stemmed_tokens_filters_stop_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop words should be excluded from the stemmed output."""
    monkeypatch.setattr("cli.utils.STOP_WORDS", ["the", "a"])
    assert get_stemmed_tokens("the batman") == ["batman"]


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


def test_timer_prints_elapsed_to_stderr(capsys: CaptureFixture[str]) -> None:
    """timer() should print elapsed time in seconds to stderr on exit."""
    with timer():
        pass
    err = capsys.readouterr().err
    assert "Completed in" in err
    assert err.strip().endswith("s")


class TestCosineSimilarity:
    """Tests for the cosine_similarity function."""

    def test_identical_vectors_return_one(self) -> None:
        """Identical non-zero vectors should have cosine similarity of 1.0."""
        vec = np.array([1.0, 2.0, 3.0])
        result = cosine_similarity(vec, vec)
        assert abs(result - 1.0) < 1e-6

    def test_orthogonal_vectors_return_zero(self) -> None:
        """Orthogonal vectors should have cosine similarity of 0.0."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        result = cosine_similarity(vec1, vec2)
        assert abs(result) < 1e-6

    def test_opposite_vectors_return_minus_one(self) -> None:
        """Opposite vectors should have cosine similarity of -1.0."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([-1.0, 0.0])
        result = cosine_similarity(vec1, vec2)
        assert abs(result - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self) -> None:
        """A zero vector should yield 0.0 to avoid division by zero."""
        zero = np.array([0.0, 0.0])
        vec = np.array([1.0, 2.0])
        assert cosine_similarity(zero, vec) == 0.0
        assert cosine_similarity(vec, zero) == 0.0


class TestGetOverlappingChunks:
    """Tests for the get_overlapping_chunks function."""

    def test_no_overlap_produces_non_overlapping_chunks(self) -> None:
        """With overlap=0, chunks should not share words."""
        words = ["a", "b", "c", "d"]
        assert get_overlapping_chunks(words, chunk_size=2, overlap=0) == [
            ["a", "b"],
            ["c", "d"],
        ]

    def test_overlap_shares_words_between_chunks(self) -> None:
        """With overlap=1, consecutive chunks should share one word."""
        words = "The bear attack was very terrifying".split()
        result = get_overlapping_chunks(words, chunk_size=4, overlap=1)
        assert result[0] == ["The", "bear", "attack", "was"]
        assert result[1] == ["was", "very", "terrifying"]

    def test_last_chunk_may_be_shorter(self) -> None:
        """The final chunk should include remaining words even if shorter."""
        words = ["a", "b", "c", "d", "e"]
        result = get_overlapping_chunks(words, chunk_size=3, overlap=0)
        assert result[-1] == ["d", "e"]

    def test_overlap_equal_to_chunk_size_raises(self) -> None:
        """overlap >= chunk_size should raise ValueError."""
        with pytest.raises(ValueError, match="Overlap"):
            get_overlapping_chunks(["a", "b"], chunk_size=2, overlap=2)

    def test_empty_words_returns_empty(self) -> None:
        """An empty word list should return an empty list."""
        assert get_overlapping_chunks([], chunk_size=3, overlap=1) == []
