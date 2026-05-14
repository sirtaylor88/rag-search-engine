"""Multimodal search: image embedding via a CLIP sentence-transformer model."""

from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageFile
from sentence_transformers import SentenceTransformer

from cli.constants import DEFAULT_MULTIMODAL_EMBEDDIND_MODEL
from cli.singleton import Singleton


class MultimodalSearch(Singleton):
    """Singleton wrapper around a CLIP-based multimodal sentence-transformer model."""

    def __init__(self, model_name: str = DEFAULT_MULTIMODAL_EMBEDDIND_MODEL) -> None:
        """Load the multimodal model on first instantiation.

        Args:
            model_name (str): Sentence-transformers model identifier.
                Defaults to ``DEFAULT_MULTIMODAL_EMBEDDIND_MODEL``.
        """
        if self._initialized:
            return

        self.model = SentenceTransformer(model_name)

        self._initialized = True

    def embed_image(self, file_path: str) -> npt.NDArray[Any]:
        """Encode an image file into a dense embedding vector.

        Args:
            file_path (str): Path to the image file.

        Returns:
            npt.NDArray[Any]: 1-D embedding array produced by the CLIP model.
        """
        image_file: ImageFile.ImageFile = Image.open(file_path)

        return np.asarray(self.model.encode([image_file])[0])


def verify_image_embedding(file_path: str) -> None:
    """Load the multimodal model, embed an image, and print the embedding size.

    Args:
        file_path (str): Path to the image file to embed.
    """
    ms = MultimodalSearch()
    embedding = ms.embed_image(file_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
