"""Tests for cli.inverted_index."""

# pylint: disable=redefined-outer-name

import math
import pickle
from pathlib import Path
import pytest

from cli.constants import BM25_B, BM25_K1
from cli.core.keyword_search import Document, InvertedIndex


def _make_movies(*titles: str) -> list[Document]:
    """Build a minimal movie list with sequential IDs and empty descriptions."""
    return [
        Document(id=i, title=title, description="")
        for i, title in enumerate(titles, start=1)
    ]


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the InvertedIndex singleton before each test."""
    monkeypatch.setattr(InvertedIndex, "_instance", None)


@pytest.fixture()
def small_index() -> InvertedIndex:
    """InvertedIndex built from three movies."""
    idx = InvertedIndex()
    idx.build(_make_movies("Batman Begins", "Batman Returns", "Inception"))
    return idx


def test_returns_same_instance_on_multiple_calls() -> None:
    """Two InvertedIndex() calls should return the identical object."""
    assert InvertedIndex() is InvertedIndex()


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
    """load() should restore all five persisted attributes saved by save()."""
    monkeypatch.chdir(tmp_path)
    idx = InvertedIndex()
    idx.build(_make_movies("Batman Begins"))
    idx.save()

    new_idx = InvertedIndex()
    new_idx.load()
    assert new_idx.index == idx.index
    assert new_idx.docmap == idx.docmap
    assert new_idx.term_frequencies == idx.term_frequencies
    assert new_idx.doc_lengths == idx.doc_lengths


def test_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """save() should write pickle files for all four persisted attributes."""
    monkeypatch.chdir(tmp_path)
    idx = InvertedIndex()
    idx.build(_make_movies("Batman Begins"))
    idx.save()
    assert (tmp_path / "cache" / "index.pkl").exists()
    assert (tmp_path / "cache" / "docmap.pkl").exists()
    assert (tmp_path / "cache" / "term_frequencies.pkl").exists()
    assert (tmp_path / "cache" / "doc_lengths.pkl").exists()
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


def test_build_populates_doc_lengths(small_index: InvertedIndex) -> None:
    """build() should record the token count for each indexed document."""
    assert set(small_index.doc_lengths.keys()) == {1, 2, 3}
    assert all(v > 0 for v in small_index.doc_lengths.values())


def test_avg_doc_length_returns_mean(small_index: InvertedIndex) -> None:
    """avg_doc_length should equal the mean of all per-document token counts."""
    expected = sum(small_index.doc_lengths.values()) / len(small_index.doc_lengths)
    assert small_index.avg_doc_length == pytest.approx(expected)


def test_avg_doc_length_empty_returns_zero() -> None:
    """avg_doc_length should be 0.0 when no documents have been indexed."""
    assert InvertedIndex().avg_doc_length == pytest.approx(0.0)


def test_get_bm25_tf_returns_saturated_value(small_index: InvertedIndex) -> None:
    """get_bm25_tf should return the length-normalized BM25 TF for a known term."""
    raw_tf = small_index.get_tf(1, "batman")
    length_norm = (
        1 - BM25_B + BM25_B * (small_index.doc_lengths[1] / small_index.avg_doc_length)
    )
    expected = (raw_tf * (BM25_K1 + 1)) / (raw_tf + BM25_K1 * length_norm)
    assert small_index.get_bm25_tf(1, "batman", k1=BM25_K1, b=BM25_B) == pytest.approx(
        expected
    )


def test_get_bm25_tf_zero_for_missing_term(small_index: InvertedIndex) -> None:
    """get_bm25_tf should return 0.0 when the term is absent from the document."""
    assert small_index.get_bm25_tf(
        1, "inception", k1=BM25_K1, b=BM25_B
    ) == pytest.approx(0.0)


def test_search_returns_matching_ids(small_index: InvertedIndex) -> None:
    """search() should return doc IDs containing at least one query token."""
    results = small_index.search("batman")
    assert 1 in results
    assert 2 in results


def test_search_respects_limit(small_index: InvertedIndex) -> None:
    """search() should return at most `limit` results."""
    assert len(small_index.search("batman", limit=1)) <= 1


def test_search_breaks_early_when_limit_reached(small_index: InvertedIndex) -> None:
    """search() should stop iterating tokens once doc_ids fills the limit."""
    results = small_index.search("batman begins", limit=1)
    assert len(results) == 1


def test_search_no_match_returns_empty(small_index: InvertedIndex) -> None:
    """search() should return an empty list when the query matches nothing."""
    assert small_index.search("nomatch") == []


def test_bm25_returns_positive_score(small_index: InvertedIndex) -> None:
    """bm25() should return a positive score for a term present in the document."""
    assert small_index.bm25(1, "batman") > 0.0


def test_bm25_search_returns_ranked_results(small_index: InvertedIndex) -> None:
    """bm25_search() should return (doc_id, score) pairs in descending score order."""
    results = small_index.bm25_search("batman", limit=2)
    assert len(results) <= 2
    assert all(score > 0 for _, score in results)
    if len(results) > 1:
        assert results[0][1] >= results[1][1]


def test_bm25_search_empty_for_no_match(small_index: InvertedIndex) -> None:
    """bm25_search() should return an empty list when the query matches nothing."""
    assert small_index.bm25_search("nomatch", limit=5) == []
