"""Semantic search commands: full-doc and chunked embedding search."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, override

from cli.commands.base import BaseSearchCommand
from cli.constants import CHUNKED_SEARCH_LIMIT
from cli.core.semantic_search import ChunkedSemanticSearch, SemanticSearch
from cli.schemas import Request, SearchPayload
from cli.utils import load_movies

if TYPE_CHECKING:
    from cli.core.keyword_search import Document


class BaseSemanticSearchCommand(BaseSearchCommand):
    """Abstract base for semantic search commands; owns the load/search/print loop."""

    @override
    def run(self, request: Request[SearchPayload]) -> None:
        """Load movies, prepare embeddings, then print ranked results.

        Args:
            request (Request[SearchPayload]): Contains the search query and limit.
        """
        documents = load_movies()
        self._prepare(documents)
        print("Searching for:", request.payload.query)
        results = self._search(request.payload.query, request.payload.limit)
        for idx, result in enumerate(results, start=1):
            print(f"\n{idx}. {result['title']} (score: {result['score']:.4f})")
            print(f"   {self._get_excerpt(result)[:100]}...")

    @abstractmethod
    def _prepare(self, documents: list[Document]) -> None:
        """Load or create embeddings for the given documents."""

    @abstractmethod
    def _search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Run the search and return ranked results."""

    def _get_excerpt(self, result: dict[str, Any]) -> str:
        """Return the text excerpt for a result."""
        return result["document"]


class SemanticSearchCommand(BaseSemanticSearchCommand):
    """Ranks movies by cosine similarity to the full-document query embedding."""

    def _prepare(self, documents: list[Document]) -> None:
        SemanticSearch().load_or_create_embeddings(documents)

    def _search(self, query: str, limit: int) -> list[dict[str, Any]]:
        return SemanticSearch().search(query, limit)


class SearchChunkedCommand(BaseSemanticSearchCommand):
    """Ranks movies by max chunk cosine similarity to the query embedding."""

    search_limit = CHUNKED_SEARCH_LIMIT

    def _prepare(self, documents: list[Document]) -> None:
        ChunkedSemanticSearch().load_or_create_chunk_embeddings(documents)

    def _search(self, query: str, limit: int) -> list[dict[str, Any]]:
        return ChunkedSemanticSearch().search_chunks(query, limit)
