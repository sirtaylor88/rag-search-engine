"""Verify commands: load the semantic search model or corpus embeddings."""

from abc import abstractmethod
from argparse import ArgumentParser
from typing import override

from cli.commands.base import BaseCommand
from cli.core.multimodal_search import verify_image_embedding
from cli.core.semantic_search import verify_embeddings, verify_model
from cli.schemas import EmptyPayload, Request
from cli.schemas.payloads import TermPayload


class BaseVerifyCommand(BaseCommand[EmptyPayload]):
    """Abstract base for verify commands; subclasses implement _verify."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        """No arguments are registered for verify commands.

        Args:
            parser (ArgumentParser): The verify subparser.
        """

    @override
    def run(self, request: Request[EmptyPayload]) -> None:
        """Execute verification and print the results.

        Args:
            request (Request[EmptyPayload]): No payload fields required.
        """
        self._verify()

    @abstractmethod
    def _verify(self) -> None:
        """Run the specific verification."""


class VerifyCommand(BaseVerifyCommand):
    """Command that loads the semantic search model and prints its properties."""

    @override
    def _verify(self) -> None:
        verify_model()


class VerifyEmbeddingsCommand(BaseVerifyCommand):
    """Command that loads or creates corpus embeddings and prints their shape."""

    @override
    def _verify(self) -> None:
        verify_embeddings()


class VerifyImageEmbeddingCommand(BaseVerifyCommand):
    """Command that embeds an image and prints the resulting embedding size."""

    @override
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register the ``image_path`` positional argument.

        Args:
            parser (ArgumentParser): The verify subparser.
        """
        parser.add_argument(
            "term",
            type=str,
            metavar="image_path",
            help="Path to the image file.",
        )

    @override
    def run(self, request: Request[TermPayload]) -> None:  # type: ignore[override]
        """Embed the image at the given path and print its embedding size.

        Args:
            request (Request[TermPayload]): Payload containing the image path
                as ``term``.
        """
        verify_image_embedding(request.payload.term)

    @override
    def _verify(self) -> None:  # pragma: no cover
        pass
