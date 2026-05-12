"""Hybrid search commands: weighted combination and Reciprocal Rank Fusion."""

from abc import abstractmethod
from argparse import ArgumentParser
from time import sleep
from typing import Any, Generic, TypeVar, get_args, override

from cli.api.gemini_agent import enhance_query, rerank_query
from cli.commands.base import BaseSearchCommand
from cli.constants import DEFAULT_ALPHA, DEFAULT_K
from cli.core.hybrid_search import HybridSearch
from cli.schemas import Request
from cli.schemas.payloads import (
    EnhanceMethod,
    ReRankeMethod,
    RRFSearchPayload,
    SearchPayload,
    WeightedSearchPayload,
)
from cli.utils import load_movies

P = TypeVar("P", bound=SearchPayload)


class BaseHybridSearchCommand(BaseSearchCommand, Generic[P]):
    """Shared load-search-print flow for hybrid retriever commands."""

    def _get_query(self, payload: P) -> str:
        """Return the query string to search with and display in the banner.

        Args:
            payload (P): The parsed request payload.

        Returns:
            str: The query string.
        """
        return payload.query

    @abstractmethod
    def _search(self, hs: HybridSearch, payload: P, query: str) -> list[dict[str, Any]]:
        """Run the retrieval and return ranked results.

        Args:
            hs (HybridSearch): The hybrid search instance to query.
            payload (P): The parsed request payload.
            query (str): The final query string to search with.

        Returns:
            list[dict[str, Any]]: Ranked result dicts.
        """

    @abstractmethod
    def _format_scores(self, result: dict[str, Any]) -> str:
        """Format the score line for a single result.

        Args:
            result (dict[str, Any]): A result dict from the search method.

        Returns:
            str: The formatted score string to print.
        """

    @override
    def run(self, request: Request[P]) -> None:
        """Load movies, run search, and print ranked results.

        Args:
            request (Request[P]): The parsed request containing the payload.
        """
        payload = request.payload
        query = self._get_query(payload)
        documents = load_movies()
        results = self._search(HybridSearch(documents), payload, query)
        print(f'\nResults for: "{query}"\n')
        for idx, result in enumerate(results, start=1):
            print(f"{idx}. {result['title']}")
            if getattr(payload, "rerank_method", None) == "individual":
                print(f"   Re-rank Score: {result['new_score']:.3f}/10")
            print(f"   {self._format_scores(result)}")
            print(f"   {result['document'][:100]}...")


class WeightedSearchCommand(BaseHybridSearchCommand[WeightedSearchPayload]):
    """Ranks movies by a weighted combination of BM25 and semantic scores."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register query, --limit, and --alpha arguments with the subparser.

        Args:
            parser (ArgumentParser): The subparser for this command.
        """
        super().add_arguments(parser)
        parser.add_argument(
            "--alpha",
            type=float,
            default=DEFAULT_ALPHA,
            help="Alpha coefficient to control the weighting between the two scores",
        )

    def _search(
        self,
        hs: HybridSearch,
        payload: WeightedSearchPayload,
        query: str,
    ) -> list[dict[str, Any]]:
        """Run weighted hybrid search.

        Args:
            hs (HybridSearch): The hybrid search instance to query.
            payload (WeightedSearchPayload): Contains alpha and limit.
            query (str): The query string to search with.

        Returns:
            list[dict[str, Any]]: Results ranked by hybrid score.
        """
        return hs.weighted_search(query, payload.alpha, payload.limit)

    def _format_scores(self, result: dict[str, Any]) -> str:
        """Format hybrid, BM25, and semantic scores for display.

        Args:
            result (dict[str, Any]): A result dict from weighted_search.

        Returns:
            str: Formatted score string.
        """
        return (
            f"Hybrid: {result['hybrid_score']:.4f}"
            f"  BM25: {result['bm25_score']:.4f}"
            f"  Semantic: {result['semantic_score']:.4f}"
        )


class RRFSearchCommand(BaseHybridSearchCommand[RRFSearchPayload]):
    """Ranks movies using Reciprocal Rank Fusion of BM25 and semantic rankings."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register query, --limit, and --k arguments with the subparser.

        Args:
            parser (ArgumentParser): The subparser for this command.
        """
        super().add_arguments(parser)
        parser.add_argument(
            "--k",
            type=int,
            default=DEFAULT_K,
            help="K coefficient to control the weighting between the two scores",
        )
        parser.add_argument(
            "--enhance",
            type=str.lower,
            choices=get_args(EnhanceMethod),
            help="Query enhancement method",
        )
        parser.add_argument(
            "--rerank-method",
            type=str.lower,
            choices=get_args(ReRankeMethod),
            help="Query re-ranking method",
        )

    def _get_query(self, payload: RRFSearchPayload) -> str:
        """Return the enhanced query when ``--enhance`` is set, else the original.

        Args:
            payload (RRFSearchPayload): The parsed RRF search payload.

        Returns:
            str: The final query string to search with and display.
        """
        return enhance_query(payload.query, method=payload.enhance)

    def _search(
        self,
        hs: HybridSearch,
        payload: RRFSearchPayload,
        query: str,
    ) -> list[dict[str, Any]]:
        """Run Reciprocal Rank Fusion search.

        Args:
            hs (HybridSearch): The hybrid search instance to query.
            payload (RRFSearchPayload): Contains k and limit.
            query (str): The (possibly enhanced) query string to search with.

        Returns:
            list[dict[str, Any]]: Results ranked by RRF score.
        """
        limit = payload.limit
        if payload.rerank_method == "individual":
            top_results = hs.rrf_search(query, payload.k, 5 * limit)
            for result in top_results:
                doc_input = f"{result.get('title', '')} - {result.get('document', '')}"
                result["new_score"] = rerank_query(
                    query,
                    doc_input,
                    payload.rerank_method,
                )
                sleep(3)

            return sorted(
                top_results,
                key=lambda r: r["new_score"],
                reverse=True,
            )[:limit]

        return hs.rrf_search(query, payload.k, limit)

    def _format_scores(self, result: dict[str, Any]) -> str:
        """Format RRF score, BM25 rank, and semantic rank for display.

        Args:
            result (dict[str, Any]): A result dict from rrf_search.

        Returns:
            str: Formatted score string.
        """
        return (
            f"RRF Score: {result['rrf_score']:.3f}"
            f"  BM25 Rank: {result['bm25_rank']}"
            f"  Semantic Rank: {result['semantic_rank']}"
        )
