"""WeightedSearchCommand: hybrid BM25 + semantic search over movies."""

from argparse import ArgumentParser
from typing import override

from cli.commands.base import BaseSearchCommand
from cli.constants import DEFAULT_ALPHA
from cli.core.hybrid_search import HybridSearch
from cli.schemas import Request
from cli.schemas.payloads import WeightedSearchPayload
from cli.utils import load_movies


class WeightedSearchCommand(BaseSearchCommand):
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

    @override
    def run(self, request: Request[WeightedSearchPayload]) -> None:
        """Print ranked results with hybrid, BM25, and semantic scores.

        Args:
            request (Request[WeightedSearchPayload]): Contains the search query,
                result limit, and alpha weighting coefficient.
        """
        documents = load_movies()
        results = HybridSearch(documents).weighted_search(
            request.payload.query,
            request.payload.alpha,
            request.payload.limit,
        )

        print(f'\nResults for: "{request.payload.query}"\n')
        for idx, result in enumerate(results, start=1):
            print(f"{idx}. {result['title']}")
            print(
                f"   Hybrid: {result['hybrid_score']:.4f}"
                f"  BM25: {result['bm25_score']:.4f}"
                f"  Semantic: {result['semantic_score']:.4f}"
            )
            print(f"   {result['document'][:100]}...")
