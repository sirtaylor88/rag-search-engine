"""CLI command classes."""

from cli.commands.build_command import BuildCommand
from cli.commands.compute.chunk_commands import ChunkCommand, SemanticChunkCommand
from cli.commands.compute.idf_commands import ComputeBM25IDFCommand, ComputeIDFCommand
from cli.commands.compute.normalize_command import NormalizeCommand
from cli.commands.compute.tf_commands import (
    ComputeBM25TFCommand,
    ComputeTFCommand,
    ComputeTFIDFCommand,
)
from cli.commands.embed_commands import (
    EmbedChunksCommand,
    EmbedQueryCommand,
    EmbedTextCommand,
)
from cli.commands.search.keyword_search_commands import BM25SearchCommand, SearchCommand
from cli.commands.search.semantic_search_command import (
    SearchChunkedCommand,
    SemanticSearchCommand,
)
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
    "NormalizeCommand",
    "SearchChunkedCommand",
    "SearchCommand",
    "SemanticChunkCommand",
    "SemanticSearchCommand",
    "VerifyCommand",
    "VerifyEmbeddingsCommand",
]
