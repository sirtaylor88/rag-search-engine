"""Abstract base command."""

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass
import sys
from typing import Generic, TypeVar

from cli.constants import BM25_K1
from cli.inverted_index import InvertedIndex


RequestT = TypeVar("RequestT")


@dataclass
class TermRequest:
    """Request carrying a single term."""

    term: str


@dataclass
class TermWithDocIDRequest(TermRequest):
    """Request carrying a term and a document ID."""

    doc_id: int


@dataclass
class ExtendedTermWithDocIDRequest(TermWithDocIDRequest):
    """Request carrying a term, a document ID, and a BM25 k1 parameter."""

    k1: float = BM25_K1


class BaseCommand(ABC, Generic[RequestT]):
    """Abstract base class for CLI commands."""

    def __init__(self, parser: ArgumentParser, inverted_index: InvertedIndex) -> None:
        """Register command arguments and store the shared index.

        Args:
            parser (ArgumentParser): The subparser for this command.
            inverted_index (InvertedIndex): The shared index instance.
        """
        self.inverted_index = inverted_index
        self.add_arguments(parser)

    @abstractmethod
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register the command's CLI arguments with its subparser.

        Args:
            parser (ArgumentParser): The subparser for this command.
        """

    @abstractmethod
    def run(self, request: RequestT) -> None:
        """Execute the command with the given parsed argument values."""

    def load_cache(self) -> None:
        """Load the inverted index from cache, exiting with code 1 on failure."""
        try:
            self.inverted_index.load()
        except OSError:
            print("Cannot load cache data. Please run build command first.")
            sys.exit(1)


class TermCommand(BaseCommand):
    """Base command that registers a single `term` positional argument."""

    term_help = "The term"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register the term positional argument with the subparser.

        Args:
            parser (ArgumentParser): The subparser for this command.
        """
        parser.add_argument("term", type=str, help=self.term_help)
