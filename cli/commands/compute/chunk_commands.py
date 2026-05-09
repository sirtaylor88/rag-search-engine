"""Chunk commands: split text into fixed-size word or sentence chunks."""

from argparse import ArgumentParser
from typing import override
import re

from cli.commands.base import TermCommand
from cli.constants import CHUNK_SIZE, SEMANTIC_CHUNK_SIZE, SENTENCE_SPLIT_PATTERN
from cli.schemas import ChunkPayload, Request, SemanticChunkPayload
from cli.utils import get_overlapping_chunks


class BaseChunkCommand(TermCommand):
    """Abstract base for chunk commands; registers the shared --overlap argument."""

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


class ChunkCommand(BaseChunkCommand):
    """Command that splits a text into fixed-size word chunks and prints each chunk."""

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
    def run(self, request: Request[ChunkPayload]) -> None:
        """Split the text into word chunks and print each one with its index.

        Args:
            request (Request[ChunkPayload]): Contains the text, chunk size, and overlap.
        """
        print(f"Chunking {len(request.payload.term)} characters.")

        words = request.payload.term.split()

        chunks = get_overlapping_chunks(
            words,
            chunk_size=request.payload.chunk_size,
            overlap=request.payload.overlap,
        )

        for idx, chunk in enumerate(chunks, start=1):
            print(f"{idx}. {' '.join(chunk)}")


class SemanticChunkCommand(BaseChunkCommand):
    """Command that splits a text into sentence chunks and prints each chunk."""

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
    def run(self, request: Request[SemanticChunkPayload]) -> None:
        """Split the text into sentence chunks and print each one with its index.

        Args:
            request (Request[SemanticChunkPayload]): Contains the text, max sentences
                per chunk, and overlap.
        """
        print(f"Semantically chunking {len(request.payload.term)} characters.")

        sentences = re.split(SENTENCE_SPLIT_PATTERN, request.payload.term)

        chunks = get_overlapping_chunks(
            sentences,
            chunk_size=request.payload.max_chunk_size,
            overlap=request.payload.overlap,
        )

        for idx, chunk in enumerate(chunks, start=1):
            print(f"{idx}. {' '.join(chunk)}")
