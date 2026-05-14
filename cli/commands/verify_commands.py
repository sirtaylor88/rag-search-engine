"""Verify commands: load the semantic search model or corpus embeddings."""

from abc import abstractmethod
from argparse import ArgumentParser
from typing import override

from cli.commands.base import BaseCommand
from cli.core.semantic_search import verify_embeddings, verify_model
from cli.schemas import EmptyPayload, Request


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
