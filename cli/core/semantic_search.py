"""Semantic search: loads a SentenceTransformer model for text encoding."""

from __future__ import annotations

from collections import defaultdict
import json
from typing import Any, Optional, TypedDict

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from cli.constants import (
    CACHE_DIR_PATH,
    CHUNKED_SEARCH_LIMIT,
    DEFAULT_EMBEDDING_MODEL,
    SCORE_PRECISION,
)
from cli.core.keyword_search import Document
from cli.singleton import Singleton
from cli.utils import (
    cosine_similarity,
    get_overlapping_chunks,
    get_sentences,
    load_movies,
)


class ChunkMetadata(TypedDict):
    """Metadata entry linking a chunk embedding to its source document."""

    doc_id: int
    chunk_idx: int
    total_chunks: int


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
    documents = load_movies()
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


def embed_chunks() -> None:
    """Load or create chunk embeddings for the full corpus and print the count."""
    chunked_sem_search = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = chunked_sem_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


class SemanticSearch(Singleton):
    """Wraps SentenceTransformer to encode text into dense embedding vectors."""

    EMBEDDINGS_FILE_PATH = CACHE_DIR_PATH / "movie_embeddings.npy"

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        """Load the all-MiniLM-L6-v2 model (downloads automatically on first use)."""
        if self._initialized:
            return

        # * The embedding model
        self.model: SentenceTransformer = SentenceTransformer(model_name)

        # * List of embeddings
        self.embeddings: Optional[npt.NDArray[Any]] = None

        # * List of documents
        self.documents: Optional[list[Document]] = None

        # * Map doc ID to its corresponding document
        self.document_map: dict[int, Document] = {}

        # * For singleton
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

        score_doc_pairs: list[tuple[float, Document]] = []
        for document, embedding in zip(self.documents, self.embeddings):
            score = cosine_similarity(query_embedding, embedding)
            score_doc_pairs.append((score, document))

        top_results: list[tuple[float, Document]] = sorted(
            score_doc_pairs,
            key=lambda x: x[0],
            reverse=True,
        )[:limit]

        return [
            {
                "id": doc["id"],
                "score": round(score, SCORE_PRECISION),
                "title": doc["title"],
                "document": doc["description"][:100],
            }
            for score, doc in top_results[:limit]
        ]


class ChunkedSemanticSearch(SemanticSearch):
    """Extends SemanticSearch to encode sentence-level chunks instead of full docs."""

    EMBEDDINGS_FILE_PATH = CACHE_DIR_PATH / "chunk_embeddings.npy"
    CHUNK_METADATA_FILE_PATH = CACHE_DIR_PATH / "chunk_metadata.json"

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        """Load the embedding model and initialise chunk-level state.

        Args:
            model_name (str): Sentence-transformers model identifier.
        """
        if self._initialized:
            return
        super().__init__(model_name)
        self.chunk_embeddings: Optional[npt.NDArray[Any]] = None
        self.chunk_metadata: Optional[list[ChunkMetadata]] = None

    def build_chunk_embeddings(self, documents: list[Document]) -> npt.NDArray[Any]:
        """Encode sentence chunks for all documents and persist to disk.

        Args:
            documents (list[Document]): Documents whose descriptions are chunked.

        Returns:
            npt.NDArray[Any]: 2-D array of shape (num_chunks, embedding_dim).
        """
        self._populate_docs(documents)

        # * Gather all the chunks and their chunk metadata.
        all_docs_chunks: list[str] = []
        chunk_metadata: list[ChunkMetadata] = []
        for doc in documents:
            if not doc["description"]:
                continue

            sentences = get_sentences(doc["description"])
            overlapping_chunks = get_overlapping_chunks(
                sentences,
                chunk_size=4,
                overlap=1,
            )
            individual_doc_chunks = len(overlapping_chunks)

            for idx, chunks in enumerate(overlapping_chunks):
                all_docs_chunks.append(" ".join(chunks))
                chunk_metadata.append(
                    {
                        "doc_id": doc["id"],
                        "chunk_idx": idx,
                        "total_chunks": individual_doc_chunks,
                    }
                )

        # * Embed the chunks.
        chunk_embeddings = np.asarray(
            self.model.encode(
                all_docs_chunks,
                show_progress_bar=True,
            )
        )

        # * Update the state and save.
        self.chunk_embeddings = chunk_embeddings
        self.chunk_metadata = chunk_metadata

        np.save(self.EMBEDDINGS_FILE_PATH, chunk_embeddings)
        with open(self.CHUNK_METADATA_FILE_PATH, "w", encoding="utf-8") as fh:
            data = {
                "chunks": chunk_metadata,
                "total_chunks": len(all_docs_chunks),
            }
            json.dump(data, fh, indent=2)

        return chunk_embeddings

    def load_or_create_chunk_embeddings(
        self, documents: list[Document]
    ) -> npt.NDArray[Any]:
        """Load chunk embeddings from disk when available, else build them.

        Args:
            documents (list[Document]): Documents to encode if building is needed.

        Returns:
            npt.NDArray[Any]: 2-D array of shape (num_chunks, embedding_dim).
        """
        self._populate_docs(documents)

        both_cached = (
            self.EMBEDDINGS_FILE_PATH.is_file()
            and self.CHUNK_METADATA_FILE_PATH.is_file()
        )
        if both_cached:
            chunk_embeddings = np.load(self.EMBEDDINGS_FILE_PATH)
            self.chunk_embeddings = chunk_embeddings
            with open(self.CHUNK_METADATA_FILE_PATH, encoding="utf-8") as fh:
                self.chunk_metadata = json.load(fh)["chunks"]
            return chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def _load_movies_idx_from_chunk_idx(self, chunk_idx: int) -> int:
        if self.chunk_metadata is None or self.documents is None:
            raise ValueError("Chunk metadata not loaded.")
        doc_id = self.chunk_metadata[chunk_idx]["doc_id"]
        return self.documents.index(self.document_map[doc_id])

    def search_chunks(
        self,
        query: str,
        limit: int = CHUNKED_SEARCH_LIMIT,
    ) -> list[dict[str, Any]]:
        """Rank documents by max chunk similarity to the query.

        Args:
            query (str): The search query string.
            limit (int): Maximum number of results to return.

        Returns:
            list[dict[str, Any]]: Top-ranked results with ``id``, ``title``,
                ``document``, ``score``, and ``metadata`` keys.

        Raises:
            ValueError: If chunk embeddings have not been loaded yet.
        """
        if self.chunk_embeddings is None or self.documents is None:
            raise ValueError(
                "No chunked embeddings loaded. "
                "Call `load_or_create_chunk_embeddings` first."
            )

        # * Generate query embedding and gather the chunk scores.
        query_embedding = self.generate_embedding(query)
        chunk_scores: list[dict[str, Any]] = [
            {
                "chunk_idx": idx,
                "movies_idx": self._load_movies_idx_from_chunk_idx(idx),
                "score": cosine_similarity(query_embedding, chunk_embedding),
            }
            for idx, chunk_embedding in enumerate(self.chunk_embeddings)
        ]

        # * Map movies index to max score from the chunks.
        movies_idx_to_score_mapping: defaultdict[int, float] = defaultdict(lambda: 0)
        for chunk_score in chunk_scores:
            movies_idx = chunk_score["movies_idx"]
            movies_idx_to_score_mapping[movies_idx] = max(
                chunk_score["score"],
                movies_idx_to_score_mapping[movies_idx],
            )

        # * Build the top results.
        top_results: list[tuple[int, float]] = sorted(
            movies_idx_to_score_mapping.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]

        return [
            {
                "id": self.documents[movies_idx]["id"],
                "title": self.documents[movies_idx]["title"],
                "document": self.documents[movies_idx]["description"][:100],
                "score": round(score, SCORE_PRECISION),
                "metadata": self.chunk_metadata or {},
            }
            for movies_idx, score in top_results
        ]
