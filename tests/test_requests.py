"""Tests for payload model validation and request construction."""

import pytest
from pydantic import ValidationError

from cli.commands.base import (
    BM25Payload,
    BM25Request,
    SearchPayload,
    SearchRequest,
    TermPayload,
    TermRequest,
    TermWithDocIDPayload,
    TermWithDocIDRequest,
)
from cli.constants import BM25_B, BM25_K1


class TestSearchPayload:
    """Validation tests for SearchPayload."""

    def test_valid(self) -> None:
        """Valid query and limit should be accepted."""
        payload = SearchPayload(query="batman", limit=3)
        assert payload.query == "batman"
        assert payload.limit == 3

    def test_default_limit(self) -> None:
        """Default limit should be 5."""
        assert SearchPayload(query="batman").limit == 5

    def test_empty_query_raises(self) -> None:
        """An empty query string should raise ValidationError."""
        with pytest.raises(ValidationError):
            SearchPayload(query="")

    def test_zero_limit_raises(self) -> None:
        """A limit of zero should raise ValidationError."""
        with pytest.raises(ValidationError):
            SearchPayload(query="batman", limit=0)

    def test_negative_limit_raises(self) -> None:
        """A negative limit should raise ValidationError."""
        with pytest.raises(ValidationError):
            SearchPayload(query="batman", limit=-1)


class TestTermPayload:
    """Validation tests for TermPayload."""

    def test_valid(self) -> None:
        """A non-empty term should be accepted."""
        assert TermPayload(term="batman").term == "batman"

    def test_empty_term_raises(self) -> None:
        """An empty term string should raise ValidationError."""
        with pytest.raises(ValidationError):
            TermPayload(term="")


class TestTermWithDocIDPayload:
    """Validation tests for TermWithDocIDPayload."""

    def test_valid(self) -> None:
        """A positive doc_id and non-empty term should be accepted."""
        payload = TermWithDocIDPayload(term="batman", doc_id=1)
        assert payload.doc_id == 1

    def test_zero_doc_id_raises(self) -> None:
        """A doc_id of zero should raise ValidationError."""
        with pytest.raises(ValidationError):
            TermWithDocIDPayload(term="batman", doc_id=0)

    def test_negative_doc_id_raises(self) -> None:
        """A negative doc_id should raise ValidationError."""
        with pytest.raises(ValidationError):
            TermWithDocIDPayload(term="batman", doc_id=-1)


class TestBM25Payload:
    """Validation tests for BM25Payload."""

    def test_valid_defaults(self) -> None:
        """Default k1 and b values should be accepted."""
        payload = BM25Payload(term="batman", doc_id=1)
        assert payload.k1 == BM25_K1
        assert payload.b == BM25_B

    def test_zero_k1_raises(self) -> None:
        """A k1 of zero should raise ValidationError."""
        with pytest.raises(ValidationError):
            BM25Payload(term="batman", doc_id=1, k1=0)

    def test_negative_k1_raises(self) -> None:
        """A negative k1 should raise ValidationError."""
        with pytest.raises(ValidationError):
            BM25Payload(term="batman", doc_id=1, k1=-1.0)

    def test_b_above_one_raises(self) -> None:
        """A b greater than 1 should raise ValidationError."""
        with pytest.raises(ValidationError):
            BM25Payload(term="batman", doc_id=1, b=1.1)

    def test_negative_b_raises(self) -> None:
        """A negative b should raise ValidationError."""
        with pytest.raises(ValidationError):
            BM25Payload(term="batman", doc_id=1, b=-0.1)

    def test_b_boundary_values(self) -> None:
        """b=0 and b=1 should both be accepted."""
        assert BM25Payload(term="batman", doc_id=1, b=0).b == 0
        assert BM25Payload(term="batman", doc_id=1, b=1).b == 1


class TestRequestConstruction:
    """Tests that Request wrappers correctly carry their payloads."""

    def test_search_request(self) -> None:
        """SearchRequest should expose its payload fields."""
        req = SearchRequest(payload=SearchPayload(query="batman", limit=3))
        assert req.payload.query == "batman"
        assert req.payload.limit == 3

    def test_term_request(self) -> None:
        """TermRequest should expose its payload term."""
        req = TermRequest(payload=TermPayload(term="batman"))
        assert req.payload.term == "batman"

    def test_term_with_doc_id_request(self) -> None:
        """TermWithDocIDRequest should expose its payload fields."""
        req = TermWithDocIDRequest(
            payload=TermWithDocIDPayload(term="batman", doc_id=1)
        )
        assert req.payload.doc_id == 1
        assert req.payload.term == "batman"

    def test_bm25_request(self) -> None:
        """BM25Request should expose its payload fields with defaults."""
        req = BM25Request(payload=BM25Payload(term="batman", doc_id=1))
        assert req.payload.k1 == BM25_K1
        assert req.payload.b == BM25_B
