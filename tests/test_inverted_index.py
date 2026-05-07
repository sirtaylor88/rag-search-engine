"""Tests for cli.inverted_index."""

# pylint: disable=redefined-outer-name

import pickle
from pathlib import Path
from typing import Any

import pytest

from cli.inverted_index import InvertedIndex


def _make_movies(*titles: str) -> list[dict[str, Any]]:
    """Build a minimal movie list with sequential IDs and empty descriptions."""
    return [
        {"id": i, "title": title, "description": ""}
        for i, title in enumerate(titles, start=1)
    ]


@pytest.fixture()
def small_index() -> InvertedIndex:
    """InvertedIndex built from three movies."""
    idx = InvertedIndex()
    idx.build(_make_movies("Batman Begins", "Batman Returns", "Inception"))
    return idx


@pytest.mark.parametrize("term", ["batman", "BATMAN"])
def test_get_documents_returns_matching_ids(
    small_index: InvertedIndex, term: str
) -> None:
    """Doc IDs containing the term should be returned sorted, regardless of case."""
    assert small_index.get_documents(term) == [1, 2]


def test_get_documents_no_match_returns_empty(small_index: InvertedIndex) -> None:
    """An unknown term should return an empty list."""
    assert small_index.get_documents("nomatch") == []


def test_build_populates_docmap(small_index: InvertedIndex) -> None:
    """build() should register every movie in the docmap keyed by its ID."""
    assert set(small_index.docmap.keys()) == {1, 2, 3}


def test_build_indexes_description_text() -> None:
    """build() should index tokens from both title and description fields."""
    movies = [{"id": 1, "title": "", "description": "A caped crusader"}]
    idx = InvertedIndex()
    idx.build(movies)
    assert 1 in idx.get_documents("crusad")  # Porter stem of "crusader"


def test_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """load() should restore the index and docmap saved by save()."""
    monkeypatch.chdir(tmp_path)
    idx = InvertedIndex()
    idx.build(_make_movies("Batman Begins"))
    idx.save()

    new_idx = InvertedIndex()
    new_idx.load()
    assert new_idx.index == idx.index
    assert new_idx.docmap == idx.docmap


def test_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """save() should write pickle files that round-trip back to the original index."""
    monkeypatch.chdir(tmp_path)
    idx = InvertedIndex()
    idx.build(_make_movies("Batman Begins"))
    idx.save()
    assert (tmp_path / "cache" / "index.pkl").exists()
    assert (tmp_path / "cache" / "docmap.pkl").exists()
    with open(tmp_path / "cache" / "index.pkl", "rb") as fh:
        assert pickle.load(fh) == idx.index
