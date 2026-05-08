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

# Security scan
uv run bandit -r cli/

# Test
uv run pytest
uv run pytest --cov=cli --cov-report=term-missing   # with coverage report

# Build the inverted index (run once before searching)
uv run python cli/keyword_search_cli.py build

# Run the keyword search CLI
uv run python cli/keyword_search_cli.py search "<query>"

# Get term frequency for a term in a document
uv run python cli/keyword_search_cli.py tf <doc_id> <term>
```

Pre-commit hooks run `ruff check`, `ruff format`, `pylint`, `mypy`, `bandit`, and `pytest` (enforcing 100% coverage) automatically on each commit.

## Architecture

The project is in early development. Current structure:

- `cli/keyword_search_cli.py` — CLI entry point: builds the `ArgumentParser`, instantiates each command with its subparser (registering arguments), then parses args and dispatches to `search`, `build`, or `tf`.
- `cli/commands/` — Command classes following an instance-based pattern.
  - `base.py` — `BaseCommand` abstract base class with `__init__(parser, inverted_index)`, abstract `add_arguments(parser)`, and abstract `run(*args)`.
  - `build_command.py` — `get_movies()` (loads JSON) and `BuildCommand`: registers `--data-path` and builds/saves the index.
  - `search_command.py` — `display_best_results()` and `SearchCommand`: registers the `query` positional arg and runs the search.
  - `term_frequency_command.py` — `TermFrequecyCommand`: registers `doc_id` and `term` positional args and prints the term frequency via `InvertedIndex.get_tf()`.
- `cli/inverted_index.py` — `InvertedIndex` class: builds a token→doc-ID index, tracks per-document term frequencies (`term_frequencies`), supports `get_documents(term)` and `get_tf(doc_id, term)`, and persists to/loads from `cache/` via pickle. `Document` is a `TypedDict` for movie records.
- `cli/utils.py` — Text processing helpers: `remove_all_punctuations`, `tokenize_text`, `get_stemmed_tokens` (Porter stemmer via NLTK, returns an ordered list with duplicates), `get_stop_words`, and the shared `STEMMER` instance.
- `cli/constants.py` — Project-wide constants: re-exports `STEMMER` from `cli.utils` and loads `STOP_WORDS` from `data/stopwords.txt` at import time.
- `cache/` — Pickle files (`index.pkl`, `docmap.pkl`, `term_frequencies.pkl`) written by the `build` command. Excluded from git.
- `data/movies.json` — Movie dataset (~25 MB) with fields: `id`, `title`, `description`, and more. Used as the corpus for search.
- `data/stopwords.txt` — Plain-text list of stop words (one per line) excluded from query tokens.
- `examples/movies.json` — 20-movie sample of `data/movies.json` for quick testing without the full dataset.
- `tests/` — Pytest test suite mirroring the `cli/` package structure. 100% coverage is enforced by the pre-commit hook.

The planned architecture is a RAG pipeline: keyword retrieval (BM25) as the first stage, followed by embedding-based semantic retrieval or re-ranking, with an LLM generating the final answer.
