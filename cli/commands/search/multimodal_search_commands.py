"""Multimodal search commands: verify image embedding and image-similarity search."""

from abc import abstractmethod
from argparse import ArgumentParser
from typing import Any, override

from cli.commands.base import TermCommand
from cli.core.multimodal_search import MultimodalSearch, verify_image_embedding
from cli.schemas.payloads import TermPayload
from cli.schemas.requests import Request
from cli.utils import load_movies


class BaseMultimodalCommand(TermCommand):
    """Abstract base for commands that accept an ``image_path`` positional argument."""

    @override
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register the ``image_path`` positional argument.

        Args:
            parser (ArgumentParser): The multimodal subparser.
        """
        parser.add_argument(
            "term",
            type=str,
            metavar="image_path",
            help="Path to the image file.",
        )

    @abstractmethod
    @override
    def run(self, request: Request[TermPayload]) -> None:
        """Dispatch to the specific multimodal action.

        Args:
            request (Request[TermPayload]): Payload containing the image path
                as ``term``.
        """


class VerifyImageEmbeddingCommand(BaseMultimodalCommand):
    """Command that embeds an image and prints the resulting embedding size."""

    @override
    def run(self, request: Request[TermPayload]) -> None:
        """Embed the image at the given path and print its embedding size.

        Args:
            request (Request[TermPayload]): Payload containing the image path
                as ``term``.
        """
        verify_image_embedding(request.payload.term)


class ImageSearchCommand(BaseMultimodalCommand):
    """Command that searches the corpus by image-to-text cosine similarity."""

    @override
    def run(self, request: Request[TermPayload]) -> None:
        """Search for movies most visually similar to the given image.

        Args:
            request (Request[TermPayload]): Payload containing the image path
                as ``term``.
        """
        documents = load_movies()
        ms = MultimodalSearch(documents)
        top_results: list[dict[str, Any]] = ms.search_with_image(request.payload.term)
        for idx, result in enumerate(top_results, start=1):
            print(f"\n{idx}. {result['title']} (similarity: {result['score']:.3f})")
            print(f"   {result['document'][:100]}...")
