"""Multimodal search: image embedding via a CLIP sentence-transformer model."""

from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageFile
from sentence_transformers import SentenceTransformer

from cli.constants import (
    DEFAULT_MULTIMODAL_EMBEDDIND_MODEL,
    SCORE_PRECISION,
    SEARCH_LIMIT,
)
from cli.core.keyword_search import Document
from cli.singleton import Singleton
from cli.utils import cosine_similarity, load_movies


class MultimodalSearch(Singleton):
    """Singleton wrapper around a CLIP-based multimodal sentence-transformer model."""

    def __init__(
        self,
        documents: list[Document],
        model_name: str = DEFAULT_MULTIMODAL_EMBEDDIND_MODEL,
    ) -> None:
        """Load the multimodal model and pre-encode all document texts on first use.

        Args:
            documents (list[Document]): Corpus of movie documents to search over.
            model_name (str): Sentence-transformers model identifier.
                Defaults to ``DEFAULT_MULTIMODAL_EMBEDDIND_MODEL``.
        """
        if self._initialized:
            return

        self.model = SentenceTransformer(model_name)

        self.documents = documents

        self.texts: list[str] = [
            f"{doc['title']}: {doc['description']}" for doc in documents
        ]

        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

        self._initialized = True

    def embed_image(self, image_path: str) -> npt.NDArray[Any]:
        """Encode an image file into a dense embedding vector.

        Args:
            image_path (str): Path to the image file.

        Returns:
            npt.NDArray[Any]: 1-D embedding array produced by the CLIP model.
        """
        image_file: ImageFile.ImageFile = Image.open(image_path)

        return np.asarray(self.model.encode([image_file])[0])

    def search_with_image(
        self,
        image_path: str,
        limit: int = SEARCH_LIMIT,
    ) -> list[dict[str, Any]]:
        """Rank documents by cosine similarity between an image and text embeddings.

        Args:
            image_path (str): Path to the query image file.
            limit (int): Maximum number of results to return.
                Defaults to ``SEARCH_LIMIT``.

        Returns:
            list[dict[str, Any]]: Top results with ``id``, ``score``, ``title``,
                and ``document`` (full description) keys, sorted by descending score.
        """
        img_embedding = self.embed_image(image_path)

        score_doc_pairs: list[tuple[float, Document]] = []
        for document, text_embedding in zip(self.documents, self.text_embeddings):
            score = cosine_similarity(img_embedding, text_embedding)
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
                "document": doc["description"],
            }
            for score, doc in top_results
        ]


def verify_image_embedding(image_path: str) -> None:
    """Load the multimodal model, embed an image, and print the embedding size.

    Args:
        image_path (str): Path to the image file to embed.
    """
    documents = load_movies()
    ms = MultimodalSearch(documents)
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
