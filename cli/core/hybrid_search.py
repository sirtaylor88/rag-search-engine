"""Hybrid search: combines BM25 keyword scoring and chunked semantic scoring."""

from collections import defaultdict
from typing import Any

from cli.constants import DEFAULT_ALPHA, SEARCH_LIMIT
from cli.core.keyword_search import Document, InvertedIndex
from cli.core.semantic_search import ChunkedSemanticSearch
from cli.singleton import Singleton


def normalize_scores(scores: list[float]) -> list[float]:
    """Apply min-max normalisation to scale a list of scores to [0, 1].

    Args:
        scores (list[float]): The list of scores to normalise.

    Returns:
        list[float]: Normalised scores in [0, 1]; returns 1.0 for each score
            when all values are equal (ZeroDivision guard).
    """
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    def _normalize(score: float) -> float:
        try:
            return (score - min_score) / (max_score - min_score)
        except ZeroDivisionError:
            return 1.0

    return [_normalize(score) for score in scores]


def hybrid_score(
    *,
    bm25_score: float,
    semantic_score: float,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """Compute a weighted combination of BM25 and semantic scores.

    Args:
        bm25_score (float): The BM25 retrieval score.
        semantic_score (float): The cosine similarity score from semantic search.
        alpha (float): Weight for the BM25 score; ``1 - alpha`` weights semantics.

    Returns:
        float: The combined hybrid score.
    """
    return alpha * bm25_score + (1 - alpha) * semantic_score


class HybridSearch(Singleton):
    """Combines BM25 keyword search and chunked semantic search into a single ranking.

    Implemented as a singleton so the index and embeddings are loaded only once
    per process.
    """

    def __init__(self, documents: list[Document]) -> None:
        """Load the inverted index and chunk embeddings for the given corpus.

        Args:
            documents (list[Document]): The movie corpus to search over.
        """
        if self._initialized:
            return

        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not self.idx.index_path.exists():
            self.idx.build(documents)
            self.idx.save()

        self._initialized = True

    def _bm25_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        """Return top BM25-ranked (doc_id, score) pairs for the query.

        Args:
            query (str): The search query string.
            limit (int): Maximum number of results to return.

        Returns:
            list[tuple[int, float]]: Descending-score (doc_id, bm25_score) pairs.
        """
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(
        self,
        query: str,
        alpha: float,
        limit: int = SEARCH_LIMIT,
    ) -> list[dict[str, Any]]:
        """Rank documents by a weighted combination of BM25 and semantic scores.

        Retrieves a large candidate set from both retrievers, merges scores per
        document, computes a hybrid score, and returns the top ``limit`` results.

        Args:
            query (str): The search query string.
            alpha (float): Weight for BM25 (``1 - alpha`` weights semantics).
            limit (int): Maximum number of results to return.

        Returns:
            list[dict[str, Any]]: Top results sorted by descending hybrid score, each
                with ``id``, ``title``, ``document``, ``bm25_score``,
                ``semantic_score``, and ``hybrid_score`` keys.
        """
        sample_limit = 500 * limit

        bm25_results = self._bm25_search(query, sample_limit)
        semantic_results = self.semantic_search.search_chunks(query, sample_limit)

        doc_scores: defaultdict[int, dict[str, Any]] = defaultdict(
            lambda: {"bm25_score": 0.0, "semantic_score": 0.0}
        )

        for doc_id, bm25 in bm25_results:
            doc_scores[doc_id]["bm25_score"] = bm25

        for result in semantic_results:
            doc_scores[result["id"]]["semantic_score"] = result["score"]

        for doc_id, values in doc_scores.items():
            values["hybrid_score"] = hybrid_score(
                bm25_score=values["bm25_score"],
                semantic_score=values["semantic_score"],
                alpha=alpha,
            )
            doc = self.document_map[doc_id]
            values["title"] = doc["title"]
            values["document"] = doc["description"][:100]

        top_results = sorted(
            doc_scores.items(),
            key=lambda item: item[1]["hybrid_score"],
            reverse=True,
        )[:limit]

        return [
            {
                "id": doc_id,
                "title": values["title"],
                "document": values["document"],
                "bm25_score": values["bm25_score"],
                "semantic_score": values["semantic_score"],
                "hybrid_score": values["hybrid_score"],
            }
            for doc_id, values in top_results
        ]
