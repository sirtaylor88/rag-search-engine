"""Tests for cli.inverted_index."""

# pylint: disable=redefined-outer-name

import math
import pickle
from pathlib import Path

import pytest

from cli.inverted_index import Document, InvertedIndex


def _make_movies(*titles: str) -> list[Document]:
    """Build a minimal movie list with sequential IDs and empty descriptions."""
    return [
        Document(id=i, title=title, description="")
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
    movies = [Document(id=1, title="", description="A caped crusader")]
    idx = InvertedIndex()
    idx.build(movies)
    assert 1 in idx.get_documents("crusad")  # Porter stem of "crusader"


def test_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """load() should restore the index, docmap, and term_frequencies saved by save()."""
    monkeypatch.chdir(tmp_path)
    idx = InvertedIndex()
    idx.build(_make_movies("Batman Begins"))
    idx.save()

    new_idx = InvertedIndex()
    new_idx.load()
    assert new_idx.index == idx.index
    assert new_idx.docmap == idx.docmap
    assert new_idx.term_frequencies == idx.term_frequencies


def test_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """save() should write pickle files that round-trip back to the original index."""
    monkeypatch.chdir(tmp_path)
    idx = InvertedIndex()
    idx.build(_make_movies("Batman Begins"))
    idx.save()
    assert (tmp_path / "cache" / "index.pkl").exists()
    assert (tmp_path / "cache" / "docmap.pkl").exists()
    assert (tmp_path / "cache" / "term_frequencies.pkl").exists()
    with open(tmp_path / "cache" / "index.pkl", "rb") as fh:
        assert pickle.load(fh) == idx.index


def test_get_tf_returns_frequency(small_index: InvertedIndex) -> None:
    """get_tf should return the stem's frequency in the document."""
    assert small_index.get_tf(1, "batman") == 1


def test_get_tf_returns_zero_for_missing_term(small_index: InvertedIndex) -> None:
    """get_tf should return 0 when the term is not present in the document."""
    assert small_index.get_tf(1, "inception") == 0


def test_get_tf_raises_for_multi_word_term(small_index: InvertedIndex) -> None:
    """get_tf should raise ValueError when the term contains more than one word."""
    with pytest.raises(ValueError, match="unique"):
        small_index.get_tf(1, "batman begins")


def test_get_idf_returns_smoothed_value(small_index: InvertedIndex) -> None:
    """get_idf should return log((N+1)/(df+1)) for a known term."""
    # small_index has 3 docs; "batman" appears in 2 of them
    assert small_index.get_idf("batman") == pytest.approx(math.log(4 / 3))


def test_get_idf_higher_for_rare_term(small_index: InvertedIndex) -> None:
    """get_idf should be higher for a term that appears in fewer documents."""
    assert small_index.get_idf("inception") > small_index.get_idf("batman")


def test_get_bm25_idf_returns_value(small_index: InvertedIndex) -> None:
    """get_bm25_idf should return log((N-df+0.5)/(df+0.5)+1) for a known term."""
    # small_index has 3 docs; "batman" appears in 2 of them
    assert small_index.get_bm25_idf("batman") == pytest.approx(
        math.log((3 - 2 + 0.5) / (2 + 0.5) + 1)
    )


def test_get_bm25_idf_higher_for_rare_term(small_index: InvertedIndex) -> None:
    """get_bm25_idf should be higher for a term that appears in fewer documents."""
    assert small_index.get_bm25_idf("inception") > small_index.get_bm25_idf("batman")


def test_get_bm25_tf_returns_saturated_value(small_index: InvertedIndex) -> None:
    """get_bm25_tf should return (tf*(k1+1))/(tf+k1) for a known term."""
    k1 = 1.5
    raw_tf = small_index.get_tf(1, "batman")
    expected = (raw_tf * (k1 + 1)) / (raw_tf + k1)
    assert small_index.get_bm25_tf(1, "batman", k1=k1) == pytest.approx(expected)


def test_get_bm25_tf_zero_for_missing_term(small_index: InvertedIndex) -> None:
    """get_bm25_tf should return 0.0 when the term is absent from the document."""
    assert small_index.get_bm25_tf(1, "inception") == pytest.approx(0.0)
