"""Pydantic payload models for CLI commands."""

from pydantic import BaseModel, Field

from cli.constants import BM25_B, BM25_K1, CHUNK_SIZE, SEARCH_LIMIT, SEMANTIC_CHUNK_SIZE


class EmptyPayload(BaseModel):
    """Payload with no fields, used for commands that need no input data."""


class SearchPayload(BaseModel):
    """Payload for search commands: non-empty query string and positive result limit."""

    query: str = Field(min_length=1)
    limit: int = Field(default=SEARCH_LIMIT, ge=1)


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
