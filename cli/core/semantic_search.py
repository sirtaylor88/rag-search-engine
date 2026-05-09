"""Semantic search: loads a SentenceTransformer model for text encoding."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from torch import Tensor

from cli.commands.build_command import get_movies
from cli.core.keyword_search import CACHE_DIR_PATH, Document


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


def verify_embeddings() -> None:
    """Load or create embeddings for the full movie corpus and print their shape."""
    sem_search = SemanticSearch()
    documents = get_movies()
    embeddings = sem_search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors "
        f"in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str) -> None:
    """Instantiate SemanticSearch, encode a query string, and print its embedding info.

    Args:
        query (str): The query text to encode.
    """
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


class SemanticSearch:
    """Wraps SentenceTransformer to encode text into dense embedding vectors."""

    EMBEDDINGS_FILE_PATH = CACHE_DIR_PATH / "movie_embeddings.np"

    def __init__(self) -> None:
        """Load the all-MiniLM-L6-v2 model (downloads automatically on first use)."""
        self.model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings: Optional[Tensor] = None
        self.documents: Optional[list[Document]] = None
        self.document_map: dict[int, Document] = {}

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

    def _populate_docs(self, documents: list[Document]) -> None:
        """Store documents and build a lookup map keyed by document ID.

        Args:
            documents (list[Document]): The list of documents to store.
        """
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

    def build_embeddings(self, documents: list[Document]) -> Tensor:
        """Encode all documents and persist the embedding matrix to disk.

        Args:
            documents (list[Document]): Documents to encode.

        Returns:
            Tensor: 2-D tensor of shape (num_docs, embedding_dim).
        """
        self._populate_docs(documents)

        doc_reprs = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(doc_reprs, show_progress_bar=True)

        np.save(self.EMBEDDINGS_FILE_PATH, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Document]) -> Tensor:
        """Load embeddings from disk when count matches, else build them.

        Args:
            documents (list[Document]): Documents to encode if building is needed.

        Returns:
            Tensor: 2-D tensor of shape (num_docs, embedding_dim).
        """
        self._populate_docs(documents)
        if self.EMBEDDINGS_FILE_PATH.is_file():
            self.embeddings = np.load(self.EMBEDDINGS_FILE_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)
