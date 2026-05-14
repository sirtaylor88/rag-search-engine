"""CLI for Retrieval Augmented Generation: search then generate an answer."""

import argparse

from cli.commands import CitationsCommand, RagCommand, SummarizeCommand
from cli.schemas.payloads import SearchPayload
from cli.schemas.requests import SearchRequest


def main() -> None:
    """Parse CLI arguments and dispatch to the RAG command."""
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_cmd = RagCommand(
        subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    )
    summarize_cmd = SummarizeCommand(
        subparsers.add_parser("summarize", help="Summarize results using LLM model")
    )
    citations_cmd = CitationsCommand(
        subparsers.add_parser("citations", help="Answer with citations using LLM model")
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            rag_cmd.run(
                SearchRequest(payload=SearchPayload(query=args.query, limit=args.limit))
            )
        case "summarize":
            summarize_cmd.run(
                SearchRequest(payload=SearchPayload(query=args.query, limit=args.limit))
            )
        case "citations":
            citations_cmd.run(
                SearchRequest(payload=SearchPayload(query=args.query, limit=args.limit))
            )
        case _:
            parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
