"""Pydantic payload models for CLI commands."""

from typing import Annotated, Any, Literal, Optional, TypeAlias, Union

from pydantic import BaseModel, Field, field_validator

from cli.constants import (
    BM25_B,
    BM25_K1,
    CHUNK_SIZE,
    DEFAULT_ALPHA,
    DEFAULT_K,
    SEARCH_LIMIT,
    SEMANTIC_CHUNK_SIZE,
)

PositiveFloat = Annotated[float, Field(gt=0)]
EnhanceMethod: TypeAlias = Literal["spell", "rewrite"]


class EmptyPayload(BaseModel):
    """Payload with no fields, used for commands that need no input data."""


class SearchPayload(BaseModel):
    """Payload for search commands: non-empty query string and positive result limit."""

    query: str = Field(min_length=1)
    limit: int = Field(default=SEARCH_LIMIT, ge=1)


class WeightedSearchPayload(SearchPayload):
    """Payload for weighted search: query, limit, and BM25/semantic alpha weight."""

    alpha: float = Field(default=DEFAULT_ALPHA, ge=0, le=1)


class RRFSearchPayload(SearchPayload):
    """Payload for RRF search: query, limit, k parameter, and optional enhancement."""

    k: int = Field(default=DEFAULT_K, ge=1)
    enhance: Optional[EnhanceMethod] = None

    @field_validator("enhance", mode="before")
    @classmethod
    def lowercase_name(cls, value: Any) -> Union[str, Any]:
        """Normalise the enhance value to lowercase before validation."""
        if isinstance(value, str):
            return value.lower()
        return value


class TermPayload(BaseModel):
    """Payload carrying a single non-empty term string."""

    term: str = Field(min_length=1)


class OverlapPayload(TermPayload):
    """Payload carrying a term and a non-negative overlap count."""

    overlap: int = Field(default=0, ge=0)


class ChunkPayload(OverlapPayload):
    """Payload for the chunk command: text, words per chunk, and overlap count."""

    chunk_size: int = Field(default=CHUNK_SIZE, ge=1)


class SemanticChunkPayload(OverlapPayload):
    """Payload for the semantic chunk command: text, sentences per chunk, overlap."""

    max_chunk_size: int = Field(default=SEMANTIC_CHUNK_SIZE, ge=1)


class TermWithDocIDPayload(TermPayload):
    """Payload carrying a non-empty term and a positive document ID."""

    doc_id: int = Field(ge=1)


class BM25Payload(TermWithDocIDPayload):
    """Payload for BM25 TF commands: term, document ID, and tuning parameters."""

    k1: float = Field(default=BM25_K1, gt=0)
    b: float = Field(default=BM25_B, ge=0, le=1)


class ScoreListPayload(BaseModel):
    """Payload carrying a list of positive float scores for normalisation."""

    scores: list[PositiveFloat] = Field(default_factory=list)
