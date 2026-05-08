# RAG Search Engine

A search engine built with Retrieval Augmented Generation (RAG). The current focus is keyword search over a movie dataset using an inverted index with term-frequency tracking, with BM25 scoring and RAG-based semantic search planned as the next phase.

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

Example:

```
$ uv run python cli/keyword_search_cli.py search "the dark knight"
Searching for: the dark knight
1. The Dark Knight
2. The Dark Knight Rises
```

### Term frequency

Look up how many times a term appears in a specific document:

```bash
uv run python cli/keyword_search_cli.py tf <doc_id> <term>
```

Example:

```
$ uv run python cli/keyword_search_cli.py tf 1 knight
The term frequency of ``knight`` in document with ID 1 is 2
```

## Development

```bash
uv run pytest                                                    # run tests
uv run pytest --cov=cli --cov-report=term-missing               # run tests with coverage report
uv run mypy .                                                    # type check
uv run ruff check .                                              # lint
uv run ruff format .                                             # format
uv run pylint <file_or_dir>                                      # lint with pylint
uv run bandit -r cli/                                            # security scan
```

Pre-commit hooks run `ruff check`, `ruff format`, `pylint`, `mypy`, `bandit`, and `pytest` (with 100% coverage enforcement) automatically on each commit.
