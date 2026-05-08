"""Abstract base command."""

from abc import ABC, abstractmethod
from argparse import ArgumentParser
import sys
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from cli.constants import BM25_B, BM25_K1
from cli.core.keyword_search import InvertedIndex


RequestT = TypeVar("RequestT")


class SearchRequest(BaseModel):
    """Request carrying a non-empty search query and a positive result limit."""

    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1)


class TermRequest(BaseModel):
    """Request carrying a single non-empty term."""

    term: str = Field(min_length=1)


class TermWithDocIDRequest(TermRequest):
    """Request carrying a non-empty term and a positive document ID."""

    doc_id: int = Field(ge=1)


class BM25Request(TermWithDocIDRequest):
    """Request carrying a term, a document ID, and BM25 tuning parameters.

    k1 must be positive; b must be in [0, 1].
    """

    k1: float = Field(default=BM25_K1, gt=0)
    b: float = Field(default=BM25_B, ge=0, le=1)


class BaseCommand(ABC, Generic[RequestT]):
    """Abstract base class for CLI commands."""

    def __init__(self, parser: ArgumentParser) -> None:
        """Register command arguments with the shared subparser.

        Args:
            parser (ArgumentParser): The subparser for this command.
        """
        self.add_arguments(parser)

    @property
    def inverted_index(self) -> InvertedIndex:
        """Return the shared InvertedIndex singleton."""
        return InvertedIndex()

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


class BaseSearchCommand(BaseCommand):
    """Base command that registers `query` and `--limit` arguments."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register query and --limit arguments with the search subparser.

        Args:
            parser (ArgumentParser): The subparser for this command.
        """
        parser.add_argument(
            "query",
            type=str,
            help="Search query",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=5,
            help="Top N documents.",
        )
