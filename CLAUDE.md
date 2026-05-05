# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A search engine built with Retrieval Augmented Generation (RAG). The current focus is keyword search (BM25) over a movie dataset, with RAG-based semantic search planned as the next phase.

## Setup

This project uses `uv` for dependency and environment management. Python 3.12.9 is required (see `.python-version`).

```bash
uv sync                   # install dependencies
uv run pre-commit install  # set up git hooks
```

## Commands

```bash
# Lint and format
uv run ruff check .
uv run ruff format .
uv run pylint <file_or_dir>
uv run mypy .

# Test
uv run pytest

# Run the keyword search CLI
uv run python cli/keyword_search_cli.py search "<query>"
```

Pre-commit hooks run `ruff check`, `ruff format`, `pylint`, and `mypy` automatically on each commit.

## Architecture

The project is in early development. Current structure:

- `cli/keyword_search_cli.py` — CLI entry point using `argparse` with a `search` subcommand. Runs stemmed token search against `data/movies.json`.
- `cli/utils.py` — Text processing helpers: `remove_all_punctuations`, `tokenize_text`, `get_stemmed_tokens` (Porter stemmer via NLTK), and `get_stop_words`.
- `cli/constants.py` — Project-wide constants; loads `STOP_WORDS` from `data/stopwords.txt` at import time.
- `data/movies.json` — Movie dataset (~25 MB) with fields: `id`, `title`, `description`, and more. Used as the corpus for search.
- `data/stopwords.txt` — Plain-text list of stop words (one per line) excluded from query tokens.
- `examples/movies.json` — 20-movie sample of `data/movies.json` for quick testing without the full dataset.
- `tests/` — Pytest test suite mirroring the `cli/` package structure.

The planned architecture is a RAG pipeline: keyword retrieval (BM25) as the first stage, followed by embedding-based semantic retrieval or re-ranking, with an LLM generating the final answer.
