# RAG Search Engine

A search engine built with Retrieval Augmented Generation (RAG). The current focus is keyword search over a movie dataset using stemmed token matching, with RAG-based semantic search planned as the next phase.

## Setup

Python 3.12.9 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run pre-commit install
```

## Usage

### Build the inverted index

Before searching, build and cache the index from the movie dataset:

```bash
uv run python cli/keyword_search_cli.py build
```

To use the sample dataset instead:

```bash
uv run python cli/keyword_search_cli.py build --data-path examples/movies.json
```

### Search

```bash
uv run python cli/keyword_search_cli.py search "<query>"
```

Pass `--data-path` to use a different dataset:

```bash
uv run python cli/keyword_search_cli.py search "<query>" \
  --data-path examples/movies.json
```

Example:

```
$ uv run python cli/keyword_search_cli.py search "the dark knight"
Searching for: the dark knight
1. The Dark Knight
2. The Dark Knight Rises
```

## Development

```bash
uv run pytest          # run tests
uv run mypy .          # type check
uv run ruff check .    # lint
uv run ruff format .   # format
```

Pre-commit hooks run all of the above automatically on each commit.
