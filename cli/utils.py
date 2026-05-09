"""Utility functions for text processing and data loading."""

from __future__ import annotations

import json
import string
import logging
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
import numpy as np

from nltk.stem import PorterStemmer

if TYPE_CHECKING:
    import numpy.typing as npt
    from cli.core.keyword_search import Document
    from torch import Tensor

STEMMER = PorterStemmer()
logger = logging.getLogger(__name__)


@contextmanager
def timer() -> Generator[None, None, None]:
    """Context manager that prints elapsed wall-clock time on exit.

    Yields:
        None
    """
    start = time.perf_counter()
    yield
    print(f"\nCompleted in {time.perf_counter() - start:.2f}s", file=sys.stderr)


def get_stop_words(data_path: str = "data/stopwords.txt") -> list[str]:
    """Load stop words from a plain-text file, one word per line.

    Args:
        data_path (str): Path to the stop words file.

    Returns:
        list[str]: Stop words read from the file.
    """
    with open(data_path, encoding="utf-8") as fh:
        words = fh.read().splitlines()
    return words


STOP_WORDS = get_stop_words()


def remove_all_punctuations(text: str) -> str:
    """Remove ASCII punctuation and common Unicode punctuation from text.

    Args:
        text (str): The input string to process.

    Returns:
        str: The input string with all punctuation characters removed.
    """
    # ! Handle also non-ASCII punctuation from Unicode
    extra = "\u2018\u2019\u201c\u201d\u2013\u2014"
    trans_table = str.maketrans("", "", string.punctuation + extra)
    return text.translate(trans_table)


def tokenize_text(text: str) -> list[str]:
    """Lowercase, strip punctuation, and split text into tokens.

    Args:
        text (str): The input string to tokenize.

    Returns:
        list[str]: Non-empty tokens after removing punctuation and lowercasing.
    """
    formatted_text = remove_all_punctuations(text).lower()
    return formatted_text.split()


def get_stemmed_tokens(text: str) -> list[str]:
    """Tokenize text, filter stop words, and reduce each token to its Porter stem.

    Args:
        text (str): The input string to process.

    Returns:
        list[str]: Stemmed non-stop-word tokens in input order; duplicates preserved.
    """
    return [
        STEMMER.stem(token) for token in tokenize_text(text) if token not in STOP_WORDS
    ]


def get_term_token(term: str) -> str:
    """Stem a single-word term and return its token.

    Args:
        term (str): A single-word term to stem.

    Returns:
        str: The Porter stem of the term.

    Raises:
        ValueError: If the term produces more or fewer than one token.
    """
    term_tokens = get_stemmed_tokens(term)
    if len(term_tokens) != 1:
        raise ValueError("The term must be unique.")

    return term_tokens[0]


def get_movies(data_path: str = "data/movies.json") -> list[Document]:
    """Load and return the list of movies from a JSON file.

    Args:
        data_path (str): Path to the JSON file containing movie data.

    Returns:
        list[Document]: List of movie documents from the 'movies' key.
    """
    with open(data_path, encoding="utf-8") as fh:
        data = json.load(fh)

    return data["movies"]


def cosine_similarity(
    vec1: npt.NDArray[Any] | Tensor,
    vec2: npt.NDArray[Any] | Tensor,
) -> float:
    """Return the cosine similarity between two vectors, or 0.0 if either is zero.

    Args:
        vec1 (Tensor): First embedding vector.
        vec2 (Tensor): Second embedding vector.

    Returns:
        float: Cosine similarity in [-1, 1], or 0.0 when either norm is zero.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def get_overlapping_chunks(
    words: list[str],
    chunk_size: int,
    overlap: int,
) -> list[list[str]]:
    """Split a word list into overlapping fixed-size chunks.

    Args:
        words (list[str]): The list of words to chunk.
        chunk_size (int): Maximum number of words per chunk.
        overlap (int): Number of words shared between consecutive chunks.

    Returns:
        list[list[str]]: List of word-list chunks; the last chunk may be shorter.

    Raises:
        ValueError: If overlap is greater than or equal to chunk_size.
    """
    if overlap >= chunk_size:
        raise ValueError("Overlap value must be smaller than chunk size.")

    step = chunk_size - overlap
    return [words[i : i + chunk_size] for i in range(0, len(words), step)]
