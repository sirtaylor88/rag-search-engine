"""CLI for semantic movie search using sentence-transformers."""

import argparse

from cli.commands.compute.chunk_commands import SemanticChunkCommand
from cli.commands.search.semantic_search_command import SearchChunkedCommand
from cli.schemas import (
    ChunkPayload,
    ChunkRequest,
    EmptyPayload,
    EmptyRequest,
    SearchPayload,
    SearchRequest,
    SemanticChunkPayload,
    SemanticChunkRequest,
    TermPayload,
    TermRequest,
)
from cli.commands import (
    ChunkCommand,
    EmbedChunksCommand,
    EmbedQueryCommand,
    EmbedTextCommand,
    SemanticSearchCommand,
    VerifyCommand,
    VerifyEmbeddingsCommand,
)


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate semantic search command."""
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_cmd = VerifyCommand(
        subparsers.add_parser("verify", help="Verify semantic model")
    )
    verify_embeddings_cmd = VerifyEmbeddingsCommand(
        subparsers.add_parser("verify_embeddings", help="Verify embeddings")
    )
    embed_text_cmd = EmbedTextCommand(
        subparsers.add_parser("embed_text", help="Embed a text string")
    )
    embed_query_cmd = EmbedQueryCommand(
        subparsers.add_parser("embed_query", help="Embed a query string")
    )
    embed_chunks_cmd = EmbedChunksCommand(
        subparsers.add_parser("embed_chunks", help="Embed chunks")
    )
    search_cmd = SemanticSearchCommand(
        subparsers.add_parser("search", help="Search string using sematic search")
    )
    chunk_cmd = ChunkCommand(subparsers.add_parser("chunk", help="Chunk a text"))
    sem_chunk_cmd = SemanticChunkCommand(
        subparsers.add_parser(
            "semantic_chunk", help="Chunk a text using semantic chunk"
        )
    )
    search_chunked_cmd = SearchChunkedCommand(
        subparsers.add_parser(
            "search_chunked", help="Search string using chunked sematic search"
        ),
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_cmd.run(EmptyRequest(payload=EmptyPayload()))
        case "verify_embeddings":
            verify_embeddings_cmd.run(EmptyRequest(payload=EmptyPayload()))
        case "embed_text":
            embed_text_cmd.run(TermRequest(payload=TermPayload(term=args.term)))
        case "embed_query":
            embed_query_cmd.run(TermRequest(payload=TermPayload(term=args.term)))
        case "embed_chunks":
            embed_chunks_cmd.run(EmptyRequest(payload=EmptyPayload()))
        case "search":
            search_cmd.run(
                SearchRequest(payload=SearchPayload(query=args.query, limit=args.limit))
            )
        case "search_chunked":
            search_chunked_cmd.run(
                SearchRequest(payload=SearchPayload(query=args.query, limit=args.limit))
            )
        case "chunk":
            chunk_cmd.run(
                ChunkRequest(
                    payload=ChunkPayload(
                        term=args.term,
                        chunk_size=args.chunk_size,
                        overlap=args.overlap,
                    )
                )
            )
        case "semantic_chunk":
            sem_chunk_cmd.run(
                SemanticChunkRequest(
                    payload=SemanticChunkPayload(
                        term=args.term,
                        max_chunk_size=args.max_chunk_size,
                        overlap=args.overlap,
                    )
                )
            )
        case _:
            parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
