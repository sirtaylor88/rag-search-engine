"""RAG command: retrieve documents via RRF search then generate a grounded answer."""

from typing import override

from cli.api.gemini_agent import augment_result
from cli.commands.base import BaseSearchCommand
from cli.constants import DEFAULT_K
from cli.core.hybrid_search import HybridSearch
from cli.schemas import Request, SearchPayload
from cli.utils import load_movies


class BaseAugmentedCommand(BaseSearchCommand):
    """Base class for augmented-generation commands."""

    _method: str
    _label: str

    @override
    def run(self, request: Request[SearchPayload]) -> None:
        """Run RRF retrieval and print a Gemini-generated answer.

        Args:
            request (Request[SearchPayload]): The parsed request containing
                query and limit.
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

        answer = augment_result(query, formatted_results, method=self._method)
        print(f"{self._label}\n{answer or ''}")


class RagCommand(BaseAugmentedCommand):
    """Retrieves top results via RRF search then generates a grounded answer."""

    _method = "rag"
    _label = "RAG Response:"


class SummarizeCommand(BaseAugmentedCommand):
    """Retrieves top results via RRF search then generates a synthesized summary."""

    _method = "summarize"
    _label = "LLM Summary:"


class CitationsCommand(BaseAugmentedCommand):
    """Retrieves top results via RRF search then generates a cited LLM answer."""

    _method = "citations"
    _label = "LLM Answer:"


class QuestionCommand(BaseAugmentedCommand):
    """Retrieves top results via RRF search then answers a conversational question."""

    _method = "question"
    _label = "Answer:"
