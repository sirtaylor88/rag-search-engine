"""Pydantic request envelope models for CLI commands."""

from typing import Generic, TypeVar

from pydantic import BaseModel

from cli.schemas.payloads import (
    BM25Payload,
    ChunkPayload,
    EmptyPayload,
    ScoreListPayload,
    SearchPayload,
    SemanticChunkPayload,
    TermPayload,
    TermWithDocIDPayload,
    WeightedSearchPayload,
)

T = TypeVar("T")


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


class WeightedSearchRequest(Request[WeightedSearchPayload]):
    """Request carrying a search query, result limit, and alpha weight."""

    payload: WeightedSearchPayload


class TermRequest(Request[TermPayload]):
    """Request carrying a single non-empty term."""

    payload: TermPayload


class ChunkRequest(Request[ChunkPayload]):
    """Request carrying a text and the number of words per chunk."""

    payload: ChunkPayload


class SemanticChunkRequest(Request[SemanticChunkPayload]):
    """Request carrying a text, max sentences per chunk, and overlap count."""

    payload: SemanticChunkPayload


class TermWithDocIDRequest(Request[TermWithDocIDPayload]):
    """Request carrying a non-empty term and a positive document ID."""

    payload: TermWithDocIDPayload


class BM25Request(Request[BM25Payload]):
    """Request carrying a term, a document ID, and BM25 tuning parameters.

    k1 must be positive; b must be in [0, 1].
    """

    payload: BM25Payload


class ScoreListRequest(Request[ScoreListPayload]):
    """Request carrying a list of positive float scores."""

    payload: ScoreListPayload
