"""CLI for semantic movie search using sentence-transformers."""

import argparse

from cli.schemas import (
    ChunkPayload,
    ChunkRequest,
    EmptyPayload,
    EmptyRequest,
    SearchPayload,
    SearchRequest,
    TermPayload,
    TermRequest,
)
from cli.commands import (
    ChunkCommand,
    SemanticSearchCommand,
    EmbedQueryCommand,
    EmbedTextCommand,
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
    search_cmd = SemanticSearchCommand(
        subparsers.add_parser("search", help="Search string")
    )
    chunk_cmd = ChunkCommand(subparsers.add_parser("chunk", help="Chunk a text"))

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
        case "search":
            search_cmd.run(
                SearchRequest(payload=SearchPayload(query=args.query, limit=args.limit))
            )
        case "chunk":
            chunk_cmd.run(
                ChunkRequest(
                    payload=ChunkPayload(term=args.term, chunk_size=args.chunk_size)
                )
            )
        case _:
            parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
