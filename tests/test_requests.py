"""Tests for payload model validation and request construction."""

from pydantic import ValidationError
import pytest

from cli.constants import BM25_B, BM25_K1, CHUNK_SIZE, DEFAULT_K, SEMANTIC_CHUNK_SIZE
from cli.schemas import (
    BM25Payload,
    BM25Request,
    ChunkPayload,
    ChunkRequest,
    EmptyPayload,
    EmptyRequest,
    OverlapPayload,
    RRFSearchPayload,
    RRFSearchRequest,
    SearchPayload,
    SearchRequest,
    SemanticChunkPayload,
    SemanticChunkRequest,
    TermPayload,
    TermRequest,
    TermWithDocIDPayload,
    TermWithDocIDRequest,
)


class TestOverlapPayload:
    """Validation tests for OverlapPayload."""

    def test_valid_defaults(self) -> None:
        """Default overlap should be 0."""
        payload = OverlapPayload(term="hello world")
        assert payload.overlap == 0

    def test_custom_overlap(self) -> None:
        """A non-negative overlap should be accepted."""
        assert OverlapPayload(term="hello world", overlap=2).overlap == 2

    def test_negative_overlap_raises(self) -> None:
        """A negative overlap should raise ValidationError."""
        with pytest.raises(ValidationError):
            OverlapPayload(term="hello", overlap=-1)

    def test_empty_term_raises(self) -> None:
        """An empty term should raise ValidationError."""
        with pytest.raises(ValidationError):
            OverlapPayload(term="")


class TestSemanticChunkPayload:
    """Validation tests for SemanticChunkPayload."""

    def test_valid_defaults(self) -> None:
        """Default max_chunk_size should equal SEMANTIC_CHUNK_SIZE and overlap 0."""
        payload = SemanticChunkPayload(term="Hello world.")
        assert payload.max_chunk_size == SEMANTIC_CHUNK_SIZE
        assert payload.overlap == 0

    def test_custom_max_chunk_size(self) -> None:
        """A positive max_chunk_size should be accepted."""
        assert SemanticChunkPayload(term="Hello.", max_chunk_size=3).max_chunk_size == 3

    def test_custom_overlap(self) -> None:
        """A non-negative overlap should be accepted."""
        assert (
            SemanticChunkPayload(term="Hello.", max_chunk_size=3, overlap=1).overlap
            == 1
        )

    def test_zero_max_chunk_size_raises(self) -> None:
        """A max_chunk_size of zero should raise ValidationError."""
        with pytest.raises(ValidationError):
            SemanticChunkPayload(term="Hello.", max_chunk_size=0)

    def test_negative_overlap_raises(self) -> None:
        """A negative overlap should raise ValidationError."""
        with pytest.raises(ValidationError):
            SemanticChunkPayload(term="Hello.", overlap=-1)

    def test_empty_term_raises(self) -> None:
        """An empty term should raise ValidationError."""
        with pytest.raises(ValidationError):
            SemanticChunkPayload(term="")


class TestRRFSearchPayload:
    """Validation tests for RRFSearchPayload."""

    def test_valid_defaults(self) -> None:
        """Default k should be DEFAULT_K and enhance should be None."""
        payload = RRFSearchPayload(query="batman")
        assert payload.k == DEFAULT_K
        assert payload.enhance is None

    def test_spell_enhance(self) -> None:
        """'spell' should be accepted as an enhance value."""
        payload = RRFSearchPayload(query="batman", enhance="spell")
        assert payload.enhance == "spell"

    def test_rewrite_enhance(self) -> None:
        """'rewrite' should be accepted as an enhance value."""
        payload = RRFSearchPayload(query="batman", enhance="rewrite")
        assert payload.enhance == "rewrite"

    def test_expand_enhance(self) -> None:
        """'expand' should be accepted as an enhance value."""
        payload = RRFSearchPayload(query="batman", enhance="expand")
        assert payload.enhance == "expand"

    def test_invalid_enhance_raises(self) -> None:
        """An unrecognised enhance method should raise ValidationError."""
        with pytest.raises(ValidationError):
            RRFSearchPayload(
                query="batman",
                enhance="unknown",  # type: ignore[arg-type]
            )

    def test_rerank_method_defaults_to_none(self) -> None:
        """Default rerank_method should be None."""
        assert RRFSearchPayload(query="batman").rerank_method is None

    def test_rerank_method_individual(self) -> None:
        """'individual' should be accepted as a rerank_method value."""
        payload = RRFSearchPayload(query="batman", rerank_method="individual")
        assert payload.rerank_method == "individual"

    def test_rerank_method_batch(self) -> None:
        """'batch' should be accepted as a rerank_method value."""
        payload = RRFSearchPayload(query="batman", rerank_method="batch")
        assert payload.rerank_method == "batch"

    def test_rerank_method_cross_encoder(self) -> None:
        """'cross_encoder' should be accepted as a rerank_method value."""
        payload = RRFSearchPayload(query="batman", rerank_method="cross_encoder")
        assert payload.rerank_method == "cross_encoder"

    def test_invalid_rerank_method_raises(self) -> None:
        """An unrecognised rerank_method should raise ValidationError."""
        with pytest.raises(ValidationError):
            RRFSearchPayload(
                query="batman",
                rerank_method="unknown",  # type: ignore[arg-type]
            )

    def test_zero_k_raises(self) -> None:
        """A k of zero should raise ValidationError."""
        with pytest.raises(ValidationError):
            RRFSearchPayload(query="batman", k=0)


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


class TestChunkPayload:
    """Validation tests for ChunkPayload."""

    def test_valid_defaults(self) -> None:
        """Default chunk_size should equal CHUNK_SIZE and overlap should be 0."""
        payload = ChunkPayload(term="hello world")
        assert payload.chunk_size == CHUNK_SIZE
        assert payload.overlap == 0

    def test_custom_chunk_size(self) -> None:
        """A positive chunk_size should be accepted."""
        assert ChunkPayload(term="hello", chunk_size=10).chunk_size == 10

    def test_custom_overlap(self) -> None:
        """A non-negative overlap should be accepted."""
        assert ChunkPayload(term="hello world", chunk_size=5, overlap=2).overlap == 2

    def test_negative_overlap_raises(self) -> None:
        """A negative overlap should raise ValidationError."""
        with pytest.raises(ValidationError):
            ChunkPayload(term="hello", overlap=-1)

    def test_zero_chunk_size_raises(self) -> None:
        """A chunk_size of zero should raise ValidationError."""
        with pytest.raises(ValidationError):
            ChunkPayload(term="hello", chunk_size=0)

    def test_empty_term_raises(self) -> None:
        """An empty term should raise ValidationError."""
        with pytest.raises(ValidationError):
            ChunkPayload(term="")


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

    def test_chunk_request(self) -> None:
        """ChunkRequest should expose its text and chunk_size."""
        req = ChunkRequest(payload=ChunkPayload(term="hello world", chunk_size=5))
        assert req.payload.term == "hello world"
        assert req.payload.chunk_size == 5

    def test_empty_request(self) -> None:
        """EmptyRequest should be constructable with an EmptyPayload."""
        req = EmptyRequest(payload=EmptyPayload())
        assert isinstance(req.payload, EmptyPayload)

    def test_semantic_chunk_request(self) -> None:
        """SemanticChunkRequest should expose its text and max_chunk_size."""
        req = SemanticChunkRequest(
            payload=SemanticChunkPayload(term="Hello world.", max_chunk_size=3)
        )
        assert req.payload.term == "Hello world."
        assert req.payload.max_chunk_size == 3

    def test_rrf_search_request(self) -> None:
        """RRFSearchRequest should expose query, k, and enhance fields."""
        req = RRFSearchRequest(
            payload=RRFSearchPayload(query="batman", k=30, enhance="rewrite")
        )
        assert req.payload.query == "batman"
        assert req.payload.k == 30
        assert req.payload.enhance == "rewrite"
