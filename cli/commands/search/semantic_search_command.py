"""Semantic search command."""

from typing import override

from cli.commands.base import BaseSearchCommand
from cli.utils import get_movies
from cli.core.semantic_search import SemanticSearch
from cli.schemas import Request, SearchPayload


class SemanticSearchCommand(BaseSearchCommand):
    """Ranks movies by cosine similarity to the query embedding."""

    @override
    def run(self, request: Request[SearchPayload]) -> None:
        """Load or create corpus embeddings and display the best matching results.

        Args:
            request (Request[SearchPayload]): Contains the search query string.
        """

        sem_search = SemanticSearch()
        documents = get_movies()
        sem_search.load_or_create_embeddings(documents)

        print("Searching for:", request.payload.query)

        results = sem_search.search(request.payload.query, request.payload.limit)

        for idx, result in enumerate(results, start=1):
            print(f"{idx}. {result['title']} (score: {result['score']:.4f})")
            print(f"  {result['description'][:100]}...")
