"""Semantic search: loads a SentenceTransformer model for text encoding."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from torch import Tensor


def verify_model() -> None:
    """Instantiate SemanticSearch and print model info to verify it loaded correctly."""
    sem_search = SemanticSearch()
    print(f"Model loaded: {sem_search.model}")
    print(f"Max sequence length: {sem_search.model.max_seq_length}")


def embed_text(text: str) -> None:
    """Instantiate SemanticSearch, encode text, and print embedding info.

    Args:
        text (str): The text to encode.
    """
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


class SemanticSearch:
    """Wraps SentenceTransformer to encode text into dense embedding vectors."""

    def __init__(self) -> None:
        """Load the all-MiniLM-L6-v2 model (downloads automatically on first use)."""
        from sentence_transformers import SentenceTransformer  # pylint: disable=import-outside-toplevel

        self.model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embedding(self, text: str) -> Tensor:
        """Encode text into a dense embedding vector.

        Args:
            text (str): The text to encode. Must not be empty or whitespace-only.

        Returns:
            Tensor: A 1-D tensor of shape (embedding_dim,).

        Raises:
            ValueError: If text is empty or contains only whitespace.
        """
        if not text.strip():
            raise ValueError("The text cannot be empty or contains only whitespaces.")

        return self.model.encode([text])[0]
