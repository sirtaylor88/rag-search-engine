"""Utility functions for text processing."""

import string
import logging

from nltk.stem import PorterStemmer

STEMMER = PorterStemmer()
logger = logging.getLogger(__name__)


def remove_all_punctuations(text: str) -> str:
    """Remove all punctuations.

    Args:
        text (str): The input string to process.

    Returns:
        str: The input string with all punctuation characters removed.
    """

    trans_table = str.maketrans("", "", string.punctuation)
    return text.translate(trans_table)


def tokenize_text(text: str) -> list[str]:
    """Lowercase, strip punctuation, and split text into tokens.

    Args:
        text (str): The input string to tokenize.

    Returns:
        list[str]: Non-empty tokens after removing punctuation and lowercasing.
    """
    formatted_text = remove_all_punctuations(text).lower()
    tokens = formatted_text.split(" ")
    return list(filter(None, tokens))


def get_stemmed_tokens(text: str) -> list[str]:
    """Tokenize text and reduce each token to its Porter stem.

    Args:
        text (str): The input string to process.

    Returns:
        list[str]: Stemmed tokens in input order; duplicates preserved.
    """
    return [STEMMER.stem(token) for token in tokenize_text(text)]


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
