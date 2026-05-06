"""Inverted index for fast token-based document lookup."""

import os
import pickle
from collections import defaultdict
from typing import Any, DefaultDict

import progressbar

from cli.utils import get_stemmed_tokens


class InvertedIndex:
    """An inverted index, like a SQL database index, helps to accelerate text search."""

    def __init__(self) -> None:
        self.index: DefaultDict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict[str, Any]] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        """Index a document by stemming its text and mapping each token to the doc ID.

        Args:
            doc_id (int): The document's unique identifier.
            text (str): The text content to tokenize and index.
        """
        tokens = get_stemmed_tokens(text)
        for token in tokens:
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        """Return sorted document IDs that contain the given term.

        Args:
            term (str): The search term (case-insensitive).

        Returns:
            list[int]: Sorted document IDs matching the term, or an empty list.
        """
        return sorted(self.index[term.lower()])

    def build(self, movies: list[dict[str, Any]]) -> None:
        """Build the index and document map from a list of movie dicts.

        Args:
            movies (list[dict[str, Any]]): Movies with 'id', 'title', and 'description' keys.
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
        """Write the index and document map to cache/index.pkl and cache/docmap.pkl."""
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as fh:
            pickle.dump(self.index, fh)

        with open("cache/docmap.pkl", "wb") as fh:
            pickle.dump(self.docmap, fh)
