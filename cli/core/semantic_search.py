"""Semantic search: loads a SentenceTransformer model for text encoding."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from cli.constants import CACHE_DIR_PATH, DEFAULT_EMBEDDING_MODEL
from cli.core.keyword_search import Document
from cli.singleton import Singleton
from cli.utils import cosine_similarity, get_movies


def verify_model() -> None:
    """Access the SemanticSearch singleton and print model info to verify it loaded."""
    sem_search = SemanticSearch()
    print(f"Model loaded: {sem_search.model}")
    print(f"Max sequence length: {sem_search.model.max_seq_length}")


def embed_text(text: str) -> None:
    """Encode text using the SemanticSearch singleton and print embedding info.

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
    """Encode a query using the SemanticSearch singleton and print its embedding info.

    Args:
        query (str): The query text to encode.
    """
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


class SemanticSearch(Singleton):
    """Wraps SentenceTransformer to encode text into dense embedding vectors."""

    EMBEDDINGS_FILE_PATH = CACHE_DIR_PATH / "movie_embeddings.np"

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        """Load the all-MiniLM-L6-v2 model (downloads automatically on first use)."""
        if self._initialized:
            return
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.embeddings: Optional[npt.NDArray[Any]] = None
        self.documents: Optional[list[Document]] = None
        self.document_map: dict[int, Document] = {}
        self._initialized = True

    def generate_embedding(self, text: str) -> npt.NDArray[Any]:
        """Encode text into a dense embedding vector.

        Args:
            text (str): The text to encode. Must not be empty or whitespace-only.

        Returns:
            npt.NDArray[Any]: A 1-D array of shape (embedding_dim,).

        Raises:
            ValueError: If text is empty or contains only whitespace.
        """
        if not text.strip():
            raise ValueError("The text cannot be empty or contains only whitespaces.")

        return np.asarray(self.model.encode([text])[0])

    def _populate_docs(self, documents: list[Document]) -> None:
        """Store documents and build a lookup map keyed by document ID.

        Args:
            documents (list[Document]): The list of documents to store.
        """
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

    def build_embeddings(self, documents: list[Document]) -> npt.NDArray[Any]:
        """Encode all documents and persist the embedding matrix to disk.

        Args:
            documents (list[Document]): Documents to encode.

        Returns:
            npt.NDArray[Any]: 2-D array of shape (num_docs, embedding_dim).
        """
        self._populate_docs(documents)

        doc_reprs = [f"{doc['title']}: {doc['description']}" for doc in documents]
        embeddings = np.asarray(self.model.encode(doc_reprs, show_progress_bar=True))
        self.embeddings = embeddings

        np.save(self.EMBEDDINGS_FILE_PATH, embeddings)

        return embeddings

    def load_or_create_embeddings(self, documents: list[Document]) -> npt.NDArray[Any]:
        """Load embeddings from disk when count matches, else build them.

        Args:
            documents (list[Document]): Documents to encode if building is needed.

        Returns:
            npt.NDArray[Any]: 2-D array of shape (num_docs, embedding_dim).
        """
        self._populate_docs(documents)
        if self.EMBEDDINGS_FILE_PATH.is_file():
            self.embeddings = np.load(self.EMBEDDINGS_FILE_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Rank documents by cosine similarity to the query and return the top results.

        Args:
            query (str): The search query string.
            limit (int): Maximum number of results to return.

        Returns:
            list[dict[str, Any]]: Top-ranked results, each with ``score``, ``title``,
                and ``description`` keys, sorted by descending similarity.

        Raises:
            ValueError: If embeddings have not been loaded yet.
        """
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        score_doc_pair: list[tuple[float, Document]] = []
        for document, embedding in zip(self.documents, self.embeddings):
            score = cosine_similarity(query_embedding, embedding)
            score_doc_pair.append((score, document))

        sorted_score_doc_pair = sorted(
            score_doc_pair,
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            {
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            }
            for score, doc in sorted_score_doc_pair[:limit]
        ]
