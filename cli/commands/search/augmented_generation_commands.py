"""RAG command: retrieve documents via RRF search then generate a grounded answer."""

from cli.api.gemini_agent import augment_query
from cli.commands.base import BaseSearchCommand
from cli.constants import DEFAULT_K
from cli.core.hybrid_search import HybridSearch
from cli.schemas.requests import SearchRequest
from cli.utils import load_movies


class RagCommand(BaseSearchCommand):
    """Retrieves top results via RRF search then generates a grounded answer."""

    def run(self, request: SearchRequest) -> None:  # type: ignore[override]
        """Run RRF retrieval and print a Gemini-generated answer.

        Args:
            request (SearchRequest): The parsed request containing query and limit.
        """
        payload = request.payload
        query = payload.query
        documents = load_movies()
        hs = HybridSearch(documents)
        top_results = hs.rrf_search(query, k=DEFAULT_K, limit=payload.limit)

        print("Search results:")
        formatted_results = []
        for result in top_results:
            print(f"- {result['title']}")
            formatted_results.append(f"- {result['title']} - {result['document']}")

        rag_answer = augment_query(query, formatted_results)
        print(f"RAG Response:\n{rag_answer or ''}")
