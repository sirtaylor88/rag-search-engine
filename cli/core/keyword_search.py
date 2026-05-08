"""Inverted index for fast token-based document lookup."""

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pickle  # nosec B403
from collections import defaultdict
from typing import Counter, TypedDict

import progressbar

from cli.constants import BM25_B, BM25_K1, CACHE_DIR
from cli.utils import get_stemmed_tokens, get_term_token, timer


CACHE_DIR_PATH = Path(CACHE_DIR)


class Document(TypedDict):
    """Typed structure for a movie document stored in the index."""

    id: int
    title: str
    description: str


class InvertedIndex:
    """An inverted index, like a SQL database index, helps to accelerate text search."""

    def __init__(self) -> None:
        # * Map token to a set of document IDs.
        self.index: defaultdict[str, set[int]] = defaultdict(set)

        # * Map document ID to Document information.
        self.docmap: dict[int, Document] = {}

        # * Map document ID to a Counter with frequencies of its token.
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)

        # * Map document ID to its number of tokens.
        self.doc_lengths: dict[int, int] = {}

        # * The path to the cache file for document lengths.
        self.doc_lengths_path = CACHE_DIR_PATH / "doc_lengths.pkl"

        self._lock = threading.Lock()

    @property
    def total_doc_count(self) -> int:
        """Return the total number of indexed documents."""
        return len(self.docmap)

    @property
    def avg_doc_length(self) -> float:
        """Return the mean document length across all indexed documents."""
        return self.__get_avg_doc_length()

    def __add_document(self, doc_id: int, text: str) -> None:
        """Index a document by stemming its text and mapping each token to the doc ID.

        Args:
            doc_id (int): The document's unique identifier.
            text (str): The text content to tokenize and index.
        """
        tokens = get_stemmed_tokens(text)
        counter = Counter(tokens)
        doc_length = sum(counter.values())

        with self._lock:
            self.doc_lengths[doc_id] = doc_length
            for token in set(tokens):
                self.index[token].add(doc_id)
                self.term_frequencies[doc_id][token] += counter[token]

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0

        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        """Return sorted document IDs that contain the given term.

        Args:
            term (str): The search term (case-insensitive).

        Returns:
            list[int]: Sorted document IDs matching the term, or an empty list.
        """
        return sorted(self.index[term.lower()])

    def search(self, query: str, limit: int = 5) -> list[int]:
        """Return up to `limit` sorted document IDs whose tokens overlap with the query.

        Args:
            query (str): The search query string.
            limit (int): Maximum number of document IDs to return.

        Returns:
            list[int]: Sorted document IDs of matching documents.
        """
        query_tokens = set(get_stemmed_tokens(query))
        doc_ids: set[int] = set()
        for query_token in query_tokens:
            if len(doc_ids) >= limit:
                break
            doc_ids.update(self.get_documents(query_token))

        return sorted(doc_ids)[:limit]

    def get_tf(self, doc_id: int, term: str) -> int:
        """Return the frequency of a single-token term in the given document.

        Args:
            doc_id (int): The document's unique identifier.
            term (str): A single-word term whose stem is looked up.

        Returns:
            int: Number of times the term's stem appears in the document, or 0.
        """
        term_token = get_term_token(term)

        return self.term_frequencies[doc_id].get(term_token, 0)

    def get_df(self, term: str) -> int:
        """Return the number of documents containing the given term.

        Args:
            term (str): A single-word term whose stem is looked up.

        Returns:
            int: Count of documents that contain the term's stem.
        """
        term_token = get_term_token(term)

        return len(self.index[term_token])

    def get_idf(self, term: str) -> float:
        """Compute smoothed IDF for a single-word term.

        Args:
            term (str): A single-word term to look up.

        Returns:
            float: log((N + 1) / (df + 1)) where N is total docs and df is match count.
        """

        df = self.get_df(term)

        return math.log((self.total_doc_count + 1) / (df + 1))

    def get_bm25_idf(self, term: str) -> float:
        """Compute Okapi BM25 IDF for a single-word term.

        Args:
            term (str): A single-word term to look up.

        Returns:
            float: log((N - df + 0.5) / (df + 0.5) + 1) where N is total docs
                and df is match count.
        """
        df = self.get_df(term)

        return math.log((self.total_doc_count - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(
        self,
        doc_id: int,
        term: str,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> float:
        """Compute saturated Okapi BM25 TF for a single-token term in a document.

        Args:
            doc_id (int): The document's unique identifier.
            term (str): A single-word term whose stem is looked up.
            k1 (float): BM25 saturation parameter controlling TF saturation speed.
            b (float): BM25 length normalization parameter (0 = no normalization).

        Returns:
            float: (tf * (k1 + 1)) / (tf + k1 * length_norm) where
                length_norm = 1 - b + b * (doc_length / avg_doc_length).
        """

        # * Get the length normalization factor
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.avg_doc_length)

        # * Get the raw TF score
        raw_tf = self.get_tf(doc_id, term)

        return (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        """Compute the full BM25 score for a single term in a document.

        Args:
            doc_id (int): The document's unique identifier.
            term (str): A single-word term.

        Returns:
            float: Product of BM25 TF and BM25 IDF for the term.
        """
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        """Return the top documents ranked by cumulative BM25 score.

        Args:
            query (str): The search query string.
            limit (int): Maximum number of results to return.

        Returns:
            list[tuple[int, float]]: Descending-score list of (doc_id, score) pairs.
        """
        query_tokens = set(get_stemmed_tokens(query))
        scores: defaultdict[int, float] = defaultdict(lambda: 0)

        for query_token in query_tokens:
            doc_ids = self.index.get(query_token, set())
            for doc_id in doc_ids:
                scores[doc_id] += self.bm25(doc_id, query_token)

        scores_dsc = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return scores_dsc[:limit]

    def build(self, movies: list[Document]) -> None:
        """Build the index and document map from a list of movie dicts using threads.

        Tokenization and stemming run in parallel across a thread pool; index
        writes are serialized via an internal lock.

        Args:
            movies (list[Document]): List of movie documents.
        """
        progress = progressbar.ProgressBar(
            maxval=len(movies) + 1,
            widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
        )

        def _process(movie: Document) -> None:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            with self._lock:
                self.docmap[movie["id"]] = movie

        with timer():
            progress.start()
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_process, movie) for movie in movies]
                for idx, _ in enumerate(as_completed(futures), start=1):
                    progress.update(idx)

            progress.finish()

    def save(self) -> None:
        """Write the index, document map, term frequencies, and doc lengths to cache."""

        os.makedirs(CACHE_DIR_PATH, exist_ok=True)
        for attr_name in ["index", "docmap", "term_frequencies"]:
            with open(CACHE_DIR_PATH / f"{attr_name}.pkl", "wb") as fh:
                pickle.dump(getattr(self, attr_name), fh)

        with open(self.doc_lengths_path, "wb") as fh:
            pickle.dump(self.doc_lengths, fh)

    def load(self) -> None:
        """Load index, document map, term frequencies, and doc lengths from cache."""

        for attr_name in ["index", "docmap", "term_frequencies"]:
            with open(CACHE_DIR_PATH / f"{attr_name}.pkl", "rb") as fh:
                setattr(self, attr_name, pickle.load(fh))  # nosec B301

        with open(self.doc_lengths_path, "rb") as fh:
            self.doc_lengths = pickle.load(fh)  # nosec B301
