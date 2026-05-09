"""CLI for semantic movie search using sentence-transformers."""

import argparse

from cli.schemas import (
    EmptyPayload,
    EmptyRequest,
    SearchPayload,
    SearchRequest,
    TermPayload,
    TermRequest,
)
from cli.commands import (
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
        case _:
            parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
