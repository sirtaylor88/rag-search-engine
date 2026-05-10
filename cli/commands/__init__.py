"""CLI command classes."""

from cli.commands.compute.chunk_commands import ChunkCommand, SemanticChunkCommand
from cli.commands.search.search_command import SearchCommand
from cli.commands.build_command import BuildCommand
from cli.commands.compute.tf_commands import (
    ComputeTFCommand,
    ComputeTFIDFCommand,
    ComputeBM25TFCommand,
)
from cli.commands.compute.idf_commands import ComputeIDFCommand, ComputeBM25IDFCommand
from cli.commands.search.bm25_search_command import BM25SearchCommand
from cli.commands.embed_commands import (
    EmbedChunksCommand,
    EmbedQueryCommand,
    EmbedTextCommand,
)
from cli.commands.search.semantic_search_command import SemanticSearchCommand
from cli.commands.verify_commands import VerifyCommand, VerifyEmbeddingsCommand

__all__ = [
    "BM25SearchCommand",
    "BuildCommand",
    "ChunkCommand",
    "ComputeBM25IDFCommand",
    "ComputeBM25TFCommand",
    "ComputeIDFCommand",
    "ComputeTFIDFCommand",
    "ComputeTFCommand",
    "EmbedChunksCommand",
    "EmbedQueryCommand",
    "EmbedTextCommand",
    "SearchCommand",
    "SemanticChunkCommand",
    "SemanticSearchCommand",
    "VerifyCommand",
    "VerifyEmbeddingsCommand",
]
