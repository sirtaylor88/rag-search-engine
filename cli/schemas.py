"""Pydantic payload and request models shared across CLI commands."""

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from cli.constants import BM25_B, BM25_K1, CHUNK_SIZE, SEARCH_LIMIT

T = TypeVar("T")


class EmptyPayload(BaseModel):
    """Payload with no fields, used for commands that need no input data."""


class SearchPayload(BaseModel):
    """Payload for search commands: non-empty query string and positive result limit."""

    query: str = Field(min_length=1)
    limit: int = Field(default=SEARCH_LIMIT, ge=1)


class TermPayload(BaseModel):
    """Payload carrying a single non-empty term string."""

    term: str = Field(min_length=1)


class ChunkPayload(TermPayload):
    """Payload for the chunk command: text, words per chunk, and overlap count."""

    chunk_size: int = Field(default=CHUNK_SIZE, ge=1)
    overlap: int = Field(default=0, ge=0)


class TermWithDocIDPayload(TermPayload):
    """Payload carrying a non-empty term and a positive document ID."""

    doc_id: int = Field(ge=1)


class BM25Payload(TermWithDocIDPayload):
    """Payload for BM25 TF commands: term, document ID, and tuning parameters."""

    k1: float = Field(default=BM25_K1, gt=0)
    b: float = Field(default=BM25_B, ge=0, le=1)


class Request(BaseModel, Generic[T]):
    """Generic typed base for all CLI request models.

    ``T`` is the primary data type carried by the request (e.g. ``str`` for
    text-based requests). Concrete request models inherit from ``Request[T]``
    with a bound type and add their own validated fields.
    """

    payload: T


class EmptyRequest(Request[EmptyPayload]):
    """Request with no payload fields, for commands that require no input."""

    payload: EmptyPayload


class SearchRequest(Request[SearchPayload]):
    """Request carrying a non-empty search query and a positive result limit."""

    payload: SearchPayload


class TermRequest(Request[TermPayload]):
    """Request carrying a single non-empty term."""

    payload: TermPayload


class ChunkRequest(Request[ChunkPayload]):
    """Request carrying a text and the number of words per chunk."""

    payload: ChunkPayload


class TermWithDocIDRequest(Request[TermWithDocIDPayload]):
    """Request carrying a non-empty term and a positive document ID."""

    payload: TermWithDocIDPayload


class BM25Request(Request[BM25Payload]):
    """Request carrying a term, a document ID, and BM25 tuning parameters.

    k1 must be positive; b must be in [0, 1].
    """

    payload: BM25Payload
