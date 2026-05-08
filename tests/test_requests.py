"""Tests for request model validation."""

import pytest
from pydantic import ValidationError

from cli.commands.base import (
    BM25Request,
    SearchRequest,
    TermRequest,
    TermWithDocIDRequest,
)
from cli.constants import BM25_B, BM25_K1


class TestSearchRequest:
    """Validation tests for SearchRequest."""

    def test_valid(self) -> None:
        """Valid query and limit should be accepted."""
        req = SearchRequest(query="batman", limit=3)
        assert req.query == "batman"
        assert req.limit == 3

    def test_default_limit(self) -> None:
        """Default limit should be 5."""
        assert SearchRequest(query="batman").limit == 5

    def test_empty_query_raises(self) -> None:
        """An empty query string should raise ValidationError."""
        with pytest.raises(ValidationError):
            SearchRequest(query="")

    def test_zero_limit_raises(self) -> None:
        """A limit of zero should raise ValidationError."""
        with pytest.raises(ValidationError):
            SearchRequest(query="batman", limit=0)

    def test_negative_limit_raises(self) -> None:
        """A negative limit should raise ValidationError."""
        with pytest.raises(ValidationError):
            SearchRequest(query="batman", limit=-1)


class TestTermRequest:
    """Validation tests for TermRequest."""

    def test_valid(self) -> None:
        """A non-empty term should be accepted."""
        assert TermRequest(term="batman").term == "batman"

    def test_empty_term_raises(self) -> None:
        """An empty term string should raise ValidationError."""
        with pytest.raises(ValidationError):
            TermRequest(term="")


class TestTermWithDocIDRequest:
    """Validation tests for TermWithDocIDRequest."""

    def test_valid(self) -> None:
        """A positive doc_id and non-empty term should be accepted."""
        req = TermWithDocIDRequest(term="batman", doc_id=1)
        assert req.doc_id == 1

    def test_zero_doc_id_raises(self) -> None:
        """A doc_id of zero should raise ValidationError."""
        with pytest.raises(ValidationError):
            TermWithDocIDRequest(term="batman", doc_id=0)

    def test_negative_doc_id_raises(self) -> None:
        """A negative doc_id should raise ValidationError."""
        with pytest.raises(ValidationError):
            TermWithDocIDRequest(term="batman", doc_id=-1)


class TestBM25Request:
    """Validation tests for BM25Request."""

    def test_valid_defaults(self) -> None:
        """Default k1 and b values should be accepted."""
        req = BM25Request(term="batman", doc_id=1)
        assert req.k1 == BM25_K1
        assert req.b == BM25_B

    def test_zero_k1_raises(self) -> None:
        """A k1 of zero should raise ValidationError."""
        with pytest.raises(ValidationError):
            BM25Request(term="batman", doc_id=1, k1=0)

    def test_negative_k1_raises(self) -> None:
        """A negative k1 should raise ValidationError."""
        with pytest.raises(ValidationError):
            BM25Request(term="batman", doc_id=1, k1=-1.0)

    def test_b_above_one_raises(self) -> None:
        """A b greater than 1 should raise ValidationError."""
        with pytest.raises(ValidationError):
            BM25Request(term="batman", doc_id=1, b=1.1)

    def test_negative_b_raises(self) -> None:
        """A negative b should raise ValidationError."""
        with pytest.raises(ValidationError):
            BM25Request(term="batman", doc_id=1, b=-0.1)

    def test_b_boundary_values(self) -> None:
        """b=0 and b=1 should both be accepted."""
        assert BM25Request(term="batman", doc_id=1, b=0).b == 0
        assert BM25Request(term="batman", doc_id=1, b=1).b == 1
