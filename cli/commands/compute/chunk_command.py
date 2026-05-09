"""Chunk command: splits text into fixed-size word chunks."""

from argparse import ArgumentParser
from typing import override

from cli.commands.base import TermCommand
from cli.constants import CHUNK_SIZE
from cli.schemas import ChunkPayload, Request
from cli.utils import get_overlapping_chunks


class ChunkCommand(TermCommand):
    """Command that splits a text into fixed-size word chunks and prints each one."""

    term_help = "Text to split into fixed-size word chunks"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register --chunk-size and --overlap arguments with the chunk subparser.

        Args:
            parser (ArgumentParser): The chunk subparser.
        """
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=CHUNK_SIZE,
            help=f"Number of words per chunk (default: {CHUNK_SIZE}).",
        )
        parser.add_argument(
            "--overlap",
            type=int,
            default=0,
            help="Number of overlap words per chunk.",
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
