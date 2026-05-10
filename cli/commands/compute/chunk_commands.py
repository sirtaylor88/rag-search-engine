"""Chunk commands: split text into fixed-size word or sentence chunks."""

from abc import abstractmethod
from argparse import ArgumentParser
from typing import Generic, TypeVar, override

from cli.commands.base import TermCommand
from cli.constants import CHUNK_SIZE, SEMANTIC_CHUNK_SIZE
from cli.schemas import ChunkPayload, OverlapPayload, Request, SemanticChunkPayload
from cli.utils import get_overlapping_chunks, get_sentences

P = TypeVar("P", bound=OverlapPayload)


class BaseChunkCommand(TermCommand, Generic[P]):
    """Abstract base for chunk commands; registers --overlap and owns the run loop."""

    _label: str
    overlap_help = "Number of overlapping text"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register the --overlap argument with the chunk subparser.

        Args:
            parser (ArgumentParser): The chunk subparser.
        """
        parser.add_argument(
            "--overlap",
            type=int,
            default=0,
            help=self.overlap_help,
        )
        super().add_arguments(parser)

    @abstractmethod
    def _split(self, term: str) -> list[str]:
        """Split term into parts for chunking."""

    @abstractmethod
    def _get_chunk_size(self, payload: P) -> int:
        """Return the chunk size from payload."""

    @override
    def run(self, request: Request[P]) -> None:
        """Split the text into chunks and print each one with its index.

        Args:
            request (Request[P]): Contains text, chunk size, and overlap.
        """
        payload = request.payload
        print(f"{self._label} {len(payload.term)} characters.")
        parts = self._split(payload.term)
        chunks = get_overlapping_chunks(
            parts,
            chunk_size=self._get_chunk_size(payload),
            overlap=payload.overlap,
        )
        for idx, chunk in enumerate(chunks, start=1):
            print(f"{idx}. {' '.join(chunk)}")


class ChunkCommand(BaseChunkCommand[ChunkPayload]):
    """Command that splits a text into fixed-size word chunks and prints each chunk."""

    _label = "Chunking"
    term_help = "Text to split into fixed-size word chunks"
    overlap_help = "Number of overlap words per chunk."

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register --chunk-size and --overlap arguments with the chunk subparser.

        Args:
            parser (ArgumentParser): The chunk subparser.
        """
        parser.add_argument(
            "--chunk-size",
            type=int,
            dest="chunk_size",
            default=CHUNK_SIZE,
            help=f"Number of words per chunk (default: {CHUNK_SIZE}).",
        )
        super().add_arguments(parser)

    @override
    def _split(self, term: str) -> list[str]:
        return term.split()

    @override
    def _get_chunk_size(self, payload: ChunkPayload) -> int:
        return payload.chunk_size


class SemanticChunkCommand(BaseChunkCommand[SemanticChunkPayload]):
    """Command that splits a text into sentence chunks and prints each chunk."""

    _label = "Semantically chunking"
    term_help = "Text to split into fixed-size sentence chunks"
    overlap_help = "Number of overlap sentences per chunk."

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register --max-chunk-size and --overlap arguments with the chunk subparser.

        Args:
            parser (ArgumentParser): The chunk subparser.
        """
        parser.add_argument(
            "--max-chunk-size",
            type=int,
            dest="max_chunk_size",
            default=SEMANTIC_CHUNK_SIZE,
            help=f"Number of sentences per chunk (default: {SEMANTIC_CHUNK_SIZE}).",
        )
        super().add_arguments(parser)

    @override
    def _split(self, term: str) -> list[str]:
        return get_sentences(term)

    @override
    def _get_chunk_size(self, payload: SemanticChunkPayload) -> int:
        return payload.max_chunk_size
