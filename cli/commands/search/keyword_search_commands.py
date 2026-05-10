"""Keyword search commands: unranked and BM25-ranked."""

from abc import abstractmethod
from typing import override

from cli.commands.base import BaseSearchCommand
from cli.schemas import Request, SearchPayload


class BaseKeywordSearchCommand(BaseSearchCommand):
    """Abstract base for keyword search commands; owns the cache-load preamble."""

    @override
    def run(self, request: Request[SearchPayload]) -> None:
        """Load the index, print the query banner, and delegate to _run_search.

        Args:
            request (Request[SearchPayload]): Contains the search query and limit.
        """
        self.load_cache()
        print("Searching for:", request.payload.query)
        self._run_search(request.payload.query, request.payload.limit)

    @abstractmethod
    def _run_search(self, query: str, limit: int) -> None:
        """Execute the search and print the results."""


class SearchCommand(BaseKeywordSearchCommand):
    """Command that queries the inverted index and prints the top matching movies."""

    def _run_search(self, query: str, limit: int) -> None:
        results = self.inverted_index.search(query, limit)
        for doc_id in results:
            title = self.inverted_index.docmap[doc_id]["title"]
            print(f"{doc_id}. {title}")


class BM25SearchCommand(BaseKeywordSearchCommand):
    """Command that ranks and displays movies by full BM25 score."""

    def _run_search(self, query: str, limit: int) -> None:
        results = self.inverted_index.bm25_search(query, limit)
        for idx, (doc_id, score) in enumerate(results, start=1):
            title = self.inverted_index.docmap[doc_id]["title"]
            print(f"{idx}. ({doc_id}) {title} - Score: {score:.2f}")
