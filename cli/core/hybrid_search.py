"""Hybrid search: combines BM25 keyword scoring and chunked semantic scoring."""

from typing import Any

from cli.constants import CHUNKED_SEARCH_LIMIT, DEFAULT_ALPHA, DEFAULT_K, SEARCH_LIMIT
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


def rrf_score(rank: int, k: int = DEFAULT_K) -> float:
    """Compute the Reciprocal Rank Fusion score for a given rank.

    Args:
        rank (int): The 1-based rank position of the document.
        k (int): Smoothing constant that prevents high scores for top-ranked items.

    Returns:
        float: The RRF score ``1 / (k + rank)``.
    """
    return 1 / (k + rank)


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

    def _fetch_candidates(
        self, query: str, sample_limit: int
    ) -> tuple[dict[int, float], dict[int, float], list[int]]:
        """Retrieve raw candidate sets from both retrievers.

        Args:
            query (str): The search query string.
            sample_limit (int): Maximum candidates to fetch from each retriever.

        Returns:
            tuple[dict[int, float], dict[int, float], list[int]]: A triple of
                ``(raw_bm25, raw_semantic, doc_ids)`` where ``raw_bm25`` and
                ``raw_semantic`` map doc_id to raw score, and ``doc_ids`` is the
                sorted union of both candidate sets.
        """
        raw_bm25: dict[int, float] = dict(self._bm25_search(query, sample_limit))
        raw_semantic: dict[int, float] = {
            r["id"]: r["score"]
            for r in self.semantic_search.search_chunks(query, sample_limit)
        }
        return raw_bm25, raw_semantic, sorted(set(raw_bm25) | set(raw_semantic))

    def weighted_search(
        self,
        query: str,
        alpha: float,
        limit: int = SEARCH_LIMIT,
    ) -> list[dict[str, Any]]:
        """Rank documents by a weighted combination of BM25 and semantic scores.

        Retrieves a large candidate set from both retrievers, normalises each
        score list to [0, 1] via min-max scaling so the two signals contribute
        equally at the chosen alpha, then returns the top ``limit`` results.

        Args:
            query (str): The search query string.
            alpha (float): Weight for normalised BM25 (``1 - alpha`` weights semantics).
            limit (int): Maximum number of results to return.

        Returns:
            list[dict[str, Any]]: Top results sorted by descending hybrid score, each
                with ``id``, ``title``, ``document`` (full description),
                ``bm25_score``, ``semantic_score``, and ``hybrid_score`` keys
                (scores are min-max normalised to [0, 1]).
        """
        raw_bm25, raw_semantic, doc_ids = self._fetch_candidates(query, 500 * limit)

        norm_bm25 = normalize_scores([raw_bm25.get(doc_id, 0.0) for doc_id in doc_ids])
        norm_semantic = normalize_scores(
            [raw_semantic.get(doc_id, 0.0) for doc_id in doc_ids]
        )

        results: list[dict[str, Any]] = []
        for idx, doc_id in enumerate(doc_ids):
            doc = self.document_map[doc_id]
            results.append(
                {
                    "id": doc_id,
                    "title": doc["title"],
                    "document": doc["description"],
                    "bm25_score": norm_bm25[idx],
                    "semantic_score": norm_semantic[idx],
                    "hybrid_score": hybrid_score(
                        bm25_score=norm_bm25[idx],
                        semantic_score=norm_semantic[idx],
                        alpha=alpha,
                    ),
                }
            )

        return sorted(results, key=lambda r: r["hybrid_score"], reverse=True)[:limit]

    def rrf_search(
        self,
        query: str,
        k: int,
        limit: int = CHUNKED_SEARCH_LIMIT,
    ) -> list[dict[str, Any]]:
        """Rank documents by Reciprocal Rank Fusion of BM25 and semantic rankings.

        Retrieves a large candidate set from both retrievers, assigns each document
        an RRF score based on its rank in each list, then returns the top ``limit``
        results sorted by combined RRF score.

        Args:
            query (str): The search query string.
            k (int): Smoothing constant for RRF scoring (higher values reduce the
                impact of top-ranked items).
            limit (int): Maximum number of results to return.

        Returns:
            list[dict[str, Any]]: Top results sorted by descending RRF score, each
                with ``id``, ``title``, ``document`` (full description),
                ``bm25_rank``, ``semantic_rank``, and ``rrf_score`` keys.
        """
        raw_bm25, raw_semantic, doc_ids = self._fetch_candidates(query, 500 * limit)

        bm25_ranks: dict[int, dict[str, Any]] = {
            doc_id: {"rank": rank, "rrf_score": rrf_score(rank, k=k)}
            for rank, doc_id in enumerate(raw_bm25.keys(), start=1)
        }
        semantic_ranks: dict[int, dict[str, Any]] = {
            doc_id: {"rank": rank, "rrf_score": rrf_score(rank, k=k)}
            for rank, doc_id in enumerate(raw_semantic.keys(), start=1)
        }

        results: list[dict[str, Any]] = []
        for doc_id in doc_ids:
            doc = self.document_map[doc_id]
            bm25_entry = bm25_ranks.get(doc_id)
            semantic_entry = semantic_ranks.get(doc_id)
            results.append(
                {
                    "id": doc_id,
                    "title": doc["title"],
                    "document": doc["description"],
                    "bm25_rank": bm25_entry["rank"] if bm25_entry else None,
                    "semantic_rank": semantic_entry["rank"] if semantic_entry else None,
                    "rrf_score": (bm25_entry["rrf_score"] if bm25_entry else 0)
                    + (semantic_entry["rrf_score"] if semantic_entry else 0),
                }
            )

        return sorted(results, key=lambda r: r["rrf_score"], reverse=True)[:limit]
