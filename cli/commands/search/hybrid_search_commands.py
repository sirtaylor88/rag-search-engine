"""Hybrid search commands: weighted combination and Reciprocal Rank Fusion."""

from abc import abstractmethod
from argparse import ArgumentParser
import json
import logging
from time import sleep
from typing import Any, Generic, TypeVar, get_args, override

import numpy.typing as npt
from sentence_transformers import CrossEncoder

from cli.api.gemini_agent import enhance_query, evaluate_query, rerank_query
from cli.commands.base import BaseSearchCommand
from cli.constants import DEFAULT_ALPHA, DEFAULT_CROSS_ENCODER_MODEL, DEFAULT_K
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

logger = logging.getLogger(__name__)

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

    def _format_result(
        self,
        idx: int,
        result: dict[str, Any],
        rerank_method: str | None,
    ) -> str:
        """Build the display block for a single ranked result.

        Args:
            idx (int): 1-based rank of this result.
            result (dict[str, Any]): Result dict from the search method.
            rerank_method (str | None): Active re-ranking method, or ``None``.

        Returns:
            str: Multi-line display string for this result.
        """
        rows = [(f"{idx}. {result['title']}")]

        if rerank_method == "individual":
            rows.append(f"   Re-rank Score: {result['new_score']:.3f}/10")
        elif rerank_method == "batch":
            rows.append(f"   Re-rank Rank: {idx}")
        elif rerank_method == "cross_encoder":
            rows.append(f"   Cross Encoder Score: {result['cross_encoder_score']}")

        rows.append(f"   {self._format_scores(result)}")
        rows.append(f"   {result['document'][:100]}...")

        return "\n".join(rows)

    @override
    def run(self, request: Request[P]) -> None:
        """Load movies, run search, print results, and optionally evaluate.

        Prints a re-ranking banner when ``payload.rerank_method`` is set,
        otherwise a plain results banner. When ``payload.evaluate`` is ``True``,
        calls ``evaluate_query`` after printing and shows a 0–3 relevance score
        per result.

        Args:
            request (Request[P]): The parsed request containing the payload.
        """
        payload = request.payload
        query = self._get_query(payload)
        documents = load_movies()
        results = self._search(HybridSearch(documents), payload, query)
        rerank_method = getattr(payload, "rerank_method", None)

        if rerank_method:
            k = getattr(payload, "k", "")
            print(
                f"\nRe-ranking top {payload.limit} results using "
                f"{rerank_method} method...\n"
                f"Reciprocal Rank Fusion Results for '{query}' (k={k})\n"
            )
        else:
            print(f'\nResults for: "{query}"\n')

        formatted_results: list[str] = []
        for idx, result in enumerate(results, start=1):
            fmt_result = self._format_result(idx, result, rerank_method)
            print(fmt_result)
            formatted_results.append(fmt_result)

        if getattr(payload, "evaluate", False):
            eval_result = evaluate_query(query, formatted_results)
            if eval_result:
                eval_scores: list[int] = json.loads(eval_result)
                for idx, (r, score) in enumerate(zip(results, eval_scores), start=1):
                    print(f"{idx}. {r['title']}: {score}/3")


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
        """Register --k, --enhance, --rerank-method, --evaluate, and --verbose.

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
        parser.add_argument(
            "--evaluate",
            action="store_true",
            help="Evaluate how relevant each result is to this query",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Enable DEBUG logging for each pipeline stage",
        )

    def _get_query(self, payload: RRFSearchPayload) -> str:
        """Return the enhanced query when ``--enhance`` is set, else the original.

        Args:
            payload (RRFSearchPayload): The parsed RRF search payload.

        Returns:
            str: The final query string to search with and display.
        """
        logger.debug("Original query: %s", payload.query)
        enhanced = enhance_query(payload.query, method=payload.enhance)
        if payload.enhance:
            logger.debug("Enhanced query (%s): %s", payload.enhance, enhanced)
        return enhanced

    def _search(
        self,
        hs: HybridSearch,
        payload: RRFSearchPayload,
        query: str,
    ) -> list[dict[str, Any]]:
        """Run Reciprocal Rank Fusion search, with optional re-ranking.

        Without ``--rerank-method``, returns ``rrf_search`` results directly.
        With a rerank method, fetches ``5 × limit`` candidates and re-ranks:

        - ``individual``: scores each candidate via ``rerank_query``
          (one call per result with a 3 s delay), then sorts by score.
        - ``batch``: sends all candidates in one ``rerank_query`` call,
          parses the JSON ID list, sorts by that order (falls back to
          original RRF order on empty/None response).
        - ``cross_encoder``: scores all pairs via a local ``CrossEncoder``
          model and sorts by score.

        Args:
            hs (HybridSearch): The hybrid search instance to query.
            payload (RRFSearchPayload): Contains k, limit, and rerank_method.
            query (str): The (possibly enhanced) query string to search with.

        Returns:
            list[dict[str, Any]]: Results ranked by RRF score or re-rank score.
        """
        limit = payload.limit

        if not payload.rerank_method:
            results = hs.rrf_search(query, payload.k, limit)
            logger.debug(
                "RRF results (%d): %s",
                len(results),
                [r["title"] for r in results],
            )
            return results

        top_results = hs.rrf_search(query, payload.k, 5 * limit)
        logger.debug(
            "RRF candidates (%d): %s",
            len(top_results),
            [r["title"] for r in top_results],
        )

        if payload.rerank_method == "individual":
            for result in top_results:
                doc_input = f"{result.get('title', '')} - {result.get('document', '')}"
                res = rerank_query(query, doc_input, payload.rerank_method)
                result["new_score"] = float(res) if res is not None else 0.0
                sleep(3)

            final = sorted(
                top_results,
                key=lambda r: r["new_score"],
                reverse=True,
            )[:limit]
            logger.debug(
                "Re-ranked results (individual, %d): %s",
                len(final),
                [r["title"] for r in final],
            )
            return final

        if payload.rerank_method == "batch":
            doc_input = "\n".join(
                f"{r['id']} - {r.get('title', '')} - {r.get('document', '')}"
                for r in top_results
            )
            res = rerank_query(query, doc_input, payload.rerank_method)
            ordered_doc_ids = json.loads(res) if res else []
            if ordered_doc_ids:
                order_map = {doc_id: idx for idx, doc_id in enumerate(ordered_doc_ids)}
                top_results.sort(key=lambda r: order_map[r["id"]])
            final = top_results[:limit]
            logger.debug(
                "Re-ranked results (batch, %d): %s",
                len(final),
                [r["title"] for r in final],
            )
            return final

        # * Cross_encoder is the only remaining ReRankeMethod value
        cross_encoder = CrossEncoder(DEFAULT_CROSS_ENCODER_MODEL)
        pairs = [
            [query, f"{r.get('title', '')} - {r.get('document', '')}"]
            for r in top_results
        ]
        scores: npt.NDArray = cross_encoder.predict(pairs)
        for result, score in zip(top_results, scores):
            result["cross_encoder_score"] = score

        final = sorted(
            top_results,
            key=lambda r: r["cross_encoder_score"],
            reverse=True,
        )[:limit]
        logger.debug(
            "Re-ranked results (cross_encoder, %d): %s",
            len(final),
            [r["title"] for r in final],
        )
        return final

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
