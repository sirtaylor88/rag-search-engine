"""Inverted index for fast token-based document lookup."""

import math
import os
import pickle  # nosec B403
from collections import defaultdict
from typing import Counter, TypedDict

import progressbar

from cli.constants import BM25_K1
from cli.utils import get_stemmed_tokens, get_term_token


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

    @property
    def total_doc_count(self) -> int:
        """Return the total number of indexed documents."""
        return len(self.docmap)

    def __add_document(self, doc_id: int, text: str) -> None:
        """Index a document by stemming its text and mapping each token to the doc ID.

        Args:
            doc_id (int): The document's unique identifier.
            text (str): The text content to tokenize and index.
        """
        tokens = get_stemmed_tokens(text)
        counter = Counter(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += counter[token]

    def get_documents(self, term: str) -> list[int]:
        """Return sorted document IDs that contain the given term.

        Args:
            term (str): The search term (case-insensitive).

        Returns:
            list[int]: Sorted document IDs matching the term, or an empty list.
        """
        return sorted(self.index[term.lower()])

    def get_tf(self, doc_id: int, term: str) -> int:
        """Return the frequency of a single-token term in the given document.

        Args:
            doc_id (int): The document's unique identifier.
            term (str): A single-word term whose stem is looked up.

        Returns:
            int: Number of times the term's stem appears in the document, or 0.
        """
        term_token = get_stemmed_tokens(term)
        if len(term_token) != 1:
            raise ValueError("The term must be unique.")

        return self.term_frequencies[doc_id].get(term_token[0], 0)

    def get_df(self, term: str) -> int:
        """Get document frequency with the term."""

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
        """Compute `Okapi BM25` IDF for a single-word term."""
        df = self.get_df(term)

        return math.log((self.total_doc_count - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1) -> float:
        """Compute saturated Okapi BM25 TF for a single-token term in a document.

        Args:
            doc_id (int): The document's unique identifier.
            term (str): A single-word term whose stem is looked up.
            k1 (float): BM25 saturation parameter. Defaults to BM25_K1.

        Returns:
            float: (tf * (k1 + 1)) / (tf + k1) where tf is the raw term frequency.
        """
        raw_tf = self.get_tf(doc_id, term)

        return (raw_tf * (k1 + 1)) / (raw_tf + k1)

    def build(self, movies: list[Document]) -> None:
        """Build the index and document map from a list of movie dicts.

        Args:
            movies (list[Document]): List of movie documents.
        """
        progress = progressbar.ProgressBar(
            maxval=len(movies) + 1,
            widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
        )
        progress.start()

        for idx, movie in enumerate(movies):
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

            progress.update(idx + 1)

        progress.finish()

    def save(self) -> None:
        """Write the index, document map, and term frequencies to cache/."""
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as fh:
            pickle.dump(self.index, fh)

        with open("cache/docmap.pkl", "wb") as fh:
            pickle.dump(self.docmap, fh)

        with open("cache/term_frequencies.pkl", "wb") as fh:
            pickle.dump(self.term_frequencies, fh)

    def load(self) -> None:
        """Load the index, document map, and term frequencies from cache/."""
        with open("cache/index.pkl", "rb") as fh:
            self.index = pickle.load(fh)  # nosec B301

        with open("cache/docmap.pkl", "rb") as fh:
            self.docmap = pickle.load(fh)  # nosec B301

        with open("cache/term_frequencies.pkl", "rb") as fh:
            self.term_frequencies = pickle.load(fh)  # nosec B301
