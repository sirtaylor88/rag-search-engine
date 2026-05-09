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

# Docstring style check (Google convention)
uv run pydocstyle cli/ --convention=google

# Security scan
uv run bandit -r cli/

# Test
uv run pytest
uv run pytest --cov=cli --cov-report=term-missing   # with coverage report

# Build the inverted index (run once before searching)
uv run python cli/keyword_search_cli.py build

# Keyword search (unranked token overlap)
uv run python cli/keyword_search_cli.py search "<query>"

# BM25 search (ranked by cumulative BM25 score)
uv run python cli/keyword_search_cli.py bm25search "<query>"

# Get term frequency for a term in a document
uv run python cli/keyword_search_cli.py tf <doc_id> <term>

# Compute inverse document frequency for a term
uv run python cli/keyword_search_cli.py idf <term>

# Compute TF-IDF score for a term in a document
uv run python cli/keyword_search_cli.py tfidf <doc_id> <term>

# Compute BM25 IDF score for a term
uv run python cli/keyword_search_cli.py bm25idf <term>

# Compute BM25 TF score for a term in a document (k1 default 1.5, b default 0.75)
uv run python cli/keyword_search_cli.py bm25tf <doc_id> <term> [k1] [b]

# Verify that the semantic search model loads correctly
uv run python cli/semantic_search_cli.py verify

# Load or create corpus embeddings and print their shape (cached to cache/)
uv run python cli/semantic_search_cli.py verify_embeddings

# Encode a text string and print its embedding info
uv run python cli/semantic_search_cli.py embed_text "<text>"

# Encode a query string and print its full embedding shape
uv run python cli/semantic_search_cli.py embed_query "<query>"

# Semantic search: rank movies by cosine similarity to the query
uv run python cli/semantic_search_cli.py search "<query>"

# Chunk text into fixed-size word groups (chunk-size default 200)
uv run python cli/semantic_search_cli.py chunk "<text>" [--chunk-size N] [--overlap N]

# Chunk text into fixed-size sentence groups (max-chunk-size default 4)
uv run python cli/semantic_search_cli.py semantic_chunk "<text>" [--max-chunk-size N] [--overlap N]
```

Pre-commit hooks run `ruff check`, `ruff format`, `pylint`, `mypy`, `pydocstyle`, `bandit`, and `pytest` (enforcing 100% coverage) automatically on each commit.

```bash
# Build HTML documentation
uv run sphinx-build -b html docs docs/_build/html

# Serve docs locally with live reload (http://127.0.0.1:8000)
uv run sphinx-autobuild docs docs/_build/html
```

Dev dependencies: `bandit`, `furo`, `mypy`, `myst-parser`, `pre-commit`, `pydocstyle`, `pylint`, `pytest`, `pytest-cov`, `ruff`, `sphinx`, `sphinx-autobuild`, `sphinx-autodoc-typehints`, `sphinx-copybutton`.

## Architecture

The project is in early development. Current structure:

- `cli/keyword_search_cli.py` — CLI entry point: builds the `ArgumentParser`, instantiates each command with its subparser (registering arguments), then parses args, wraps them in the appropriate `Request[XPayload]` instance, and dispatches to `build`, `search`, `bm25search`, `tf`, `idf`, `tfidf`, `bm25idf`, or `bm25tf`.
- `cli/semantic_search_cli.py` — Semantic search CLI entry point: registers `verify`, `verify_embeddings`, `embed_text`, `embed_query`, `search`, `chunk`, and `semantic_chunk` subcommands; dispatches to `VerifyCommand`, `VerifyEmbeddingsCommand` (both via `EmptyRequest`), `EmbedTextCommand`, `EmbedQueryCommand` (both via `TermRequest`), `SemanticSearchCommand` (via `SearchRequest`), `ChunkCommand` (via `ChunkRequest`), and `SemanticChunkCommand` (via `SemanticChunkRequest`).
- `cli/singleton.py` — `Singleton` base class. `__new__` ensures only one instance exists per subclass (per-class `_instance` ClassVar). Sets `_initialized = False` on the fresh instance; subclasses guard `__init__` with `if self._initialized: return` and set `self._initialized = True` at the end of first-time setup.
- `cli/schemas/` — Pydantic `BaseModel` payload and request models, split into two submodules; `cli/schemas/__init__.py` re-exports everything so all `from cli.schemas import …` calls work unchanged.
  - `payloads.py` — **Payload models**: `EmptyPayload` (no fields), `SearchPayload` (`query` min_length=1, `limit` ge=1 default `SEARCH_LIMIT`), `TermPayload` (`term` min_length=1), `OverlapPayload` (extends `TermPayload`, adds `overlap` ge=0 default `0`), `ChunkPayload` (extends `OverlapPayload`, adds `chunk_size` ge=1 default `CHUNK_SIZE`), `SemanticChunkPayload` (extends `OverlapPayload`, adds `max_chunk_size` ge=1 default `SEMANTIC_CHUNK_SIZE`), `TermWithDocIDPayload` (adds `doc_id` ge=1), `BM25Payload` (adds `k1` gt=0, `b` ge=0 le=1).
  - `requests.py` — **Request envelope**: `Request[T]` generic `BaseModel` with a single `payload: T` field; concrete aliases `EmptyRequest`, `SearchRequest`, `TermRequest`, `ChunkRequest`, `SemanticChunkRequest`, `TermWithDocIDRequest`, `BM25Request` narrow the type.
- `cli/commands/` — Command classes following an instance-based pattern.
  - `base.py` — Abstract command infrastructure. **`BaseCommand[PayloadT]`**: abstract base with `__init__(parser)`, abstract `add_arguments(parser)`, abstract `run(request: Request[PayloadT])`, concrete `load_cache()` (shared OSError handling), and `inverted_index` property returning the `InvertedIndex` singleton. Concrete bases: `TermCommand` (registers `term` positional arg), `BaseSearchCommand` (registers `query` positional arg and `--limit` optional arg).
  - `build_command.py` — `BuildCommand`: registers `--data-path` and builds/saves the index.
  - `embed_commands.py` — `BaseEmbedCommand(TermCommand)`: abstract base whose `run()` delegates to `_embed(term)`; subclasses implement `_embed`. `EmbedTextCommand(BaseEmbedCommand)`: calls `embed_text()` to encode the input term and print its embedding info. `EmbedQueryCommand(BaseEmbedCommand)`: calls `embed_query_text()` to encode a query string and print its full embedding shape.
  - `verify_commands.py` — `BaseVerifyCommand(BaseCommand[EmptyPayload])`: abstract base whose `run()` delegates to `_verify()`; `add_arguments` is a no-op (no args needed). `VerifyCommand(BaseVerifyCommand)`: calls `verify_model()`. `VerifyEmbeddingsCommand(BaseVerifyCommand)`: calls `verify_embeddings()` to load or build corpus embeddings and print their shape.
  - `compute/chunk_commands.py` — `BaseChunkCommand(TermCommand, Generic[P])`: generic abstract base (`P bound OverlapPayload`) that registers `--overlap` (default `0`) and owns the full `run()` loop: prints label, delegates splitting to abstract `_split(term)`, delegates chunk-size lookup to abstract `_get_chunk_size(payload: P)`, then prints each chunk. `ChunkCommand(BaseChunkCommand[ChunkPayload])`: registers `--chunk-size` (default `CHUNK_SIZE`); `_split` returns `term.split()`; `_get_chunk_size` returns `payload.chunk_size`. `SemanticChunkCommand(BaseChunkCommand[SemanticChunkPayload])`: registers `--max-chunk-size` (default `SEMANTIC_CHUNK_SIZE`); `_split` splits by `SENTENCE_SPLIT_PATTERN`; `_get_chunk_size` returns `payload.max_chunk_size`.
  - `search/` — Subpackage for keyword, BM25, and semantic search commands.
    - `search_command.py` — `SearchCommand(BaseSearchCommand)`: calls `InvertedIndex.search()` and prints matching doc titles (unranked).
    - `bm25_search_command.py` — `BM25SearchCommand(SearchCommand)`: calls `InvertedIndex.bm25_search()` and prints results ranked by BM25 score.
    - `semantic_search_command.py` — `SemanticSearchCommand(BaseSearchCommand)`: loads or creates corpus embeddings via `SemanticSearch.load_or_create_embeddings()`, then calls `SemanticSearch.search()` and prints results ranked by cosine similarity.
  - `compute/` — Subpackage for all scoring/frequency commands.
    - `tf_commands.py` — `ComputeTFCommand`: registers `doc_id` and `term` positional args and prints the raw term frequency via `InvertedIndex.get_tf()`. `ComputeTFIDFCommand`: extends `ComputeTFCommand` and prints the TF-IDF score (product of `get_tf()` and `get_idf()`). `ComputeBM25TFCommand`: extends `ComputeTFCommand`, registers optional `k1` and `b` args, and prints the BM25 saturated TF via `InvertedIndex.get_bm25_tf()`.
    - `idf_commands.py` — `ComputeIDFCommand`: extends `TermCommand` and prints the smoothed IDF via `InvertedIndex.get_idf()`. `ComputeBM25IDFCommand`: extends `ComputeIDFCommand` and prints the Okapi BM25 IDF via `InvertedIndex.get_bm25_idf()`.
- `cli/core/semantic_search.py` — `SemanticSearch(Singleton)` class wrapping a `SentenceTransformer` model. `__init__(model_name=DEFAULT_EMBEDDING_MODEL)` loads the model (defaults to `"all-MiniLM-L6-v2"`). Methods: `generate_embedding(text)` → 1-D `Tensor` (raises `ValueError` on empty input); `build_embeddings(documents)` → encodes all docs and saves to `cache/movie_embeddings.np`; `load_or_create_embeddings(documents)` → loads from disk if count matches, otherwise rebuilds; `search(query, limit)` → encodes the query, scores all documents by cosine similarity, and returns the top `limit` results as `list[dict]` with `score`, `title`, and `description` keys. Module-level helpers that access the singleton: `verify_model()` prints model info; `embed_text(text)` prints embedding dimensions; `embed_query_text(query)` prints query embedding shape; `verify_embeddings()` loads or builds corpus embeddings and prints their shape.
- `cli/core/keyword_search.py` — `InvertedIndex(Singleton)`: inherits singleton behaviour from `Singleton`; `__init__` is guarded by `_initialized` so it runs only on the first instantiation. Builds a token→doc-ID index using `ThreadPoolExecutor` (tokenization runs in parallel; a `threading.Lock` serializes index writes), tracks per-document term frequencies and lengths (`doc_lengths`, `avg_doc_length`), supports `get_documents(term)`, `search(query, limit)` (unranked token-overlap), `get_tf(doc_id, term)`, `get_idf(term)` (smoothed log IDF), `get_bm25_idf(term)` (Okapi BM25 IDF), `get_bm25_tf(doc_id, term, k1, b)` (length-normalized BM25 TF), `bm25(doc_id, term)` (full BM25 score = BM25 TF × BM25 IDF), `bm25_search(query, limit)` (ranked by cumulative BM25), and persists to/loads from `cache/` via pickle. `Document` is a `TypedDict` for movie records.
- `cli/utils.py` — Text processing and data loading helpers: `get_movies(data_path)` (loads the movies JSON and returns `list[Document]`); `remove_all_punctuations` (strips ASCII and common Unicode punctuation: curly quotes, en/em dashes), `tokenize_text`, `get_stemmed_tokens` (Porter stemmer via NLTK; filters stop words; returns ordered list with duplicates), `get_stop_words` (loads from `data/stopwords.txt`), `get_term_token` (validates and stems a single-word term), `cosine_similarity(vec1, vec2)` (returns cosine similarity in [-1, 1] between two `Tensor` vectors; returns 0.0 if either norm is zero), `get_overlapping_chunks(text_parts, chunk_size, overlap)` (splits a list of text parts into overlapping fixed-size chunks; raises `ValueError` if `overlap >= chunk_size`), `timer` (context manager that prints elapsed wall-clock time to stderr), the shared `STEMMER` instance, and `STOP_WORDS` (loaded once at import time).
- `cli/constants.py` — Project-wide constants: `BM25_K1` (default BM25 saturation parameter, `1.5`), `BM25_B` (default length normalization parameter, `0.75`), `CACHE_DIR` (cache directory name, `"cache"`), `CACHE_DIR_PATH` (`Path(CACHE_DIR)`), `SEARCH_LIMIT` (default result count, `5`), `CHUNKED_SEARCH_LIMIT` (result count for chunked searches, `10`), `CHUNK_SIZE` (default words per chunk, `200`), `SEMANTIC_CHUNK_SIZE` (default sentences per semantic chunk, `4`), `SENTENCE_SPLIT_PATTERN` (regex for sentence boundary splitting, `r"(?<=[.!?])\s+"`), `DEFAULT_EMBEDDING_MODEL` (sentence-transformers model name, `"all-MiniLM-L6-v2"`).
- `cache/` — Pickle files (`index.pkl`, `docmap.pkl`, `term_frequencies.pkl`, `doc_lengths.pkl`) written by the `build` command. Excluded from git.
- `data/movies.json` — Movie dataset (~25 MB) with fields: `id`, `title`, `description`, and more. Used as the corpus for search.
- `data/stopwords.txt` — Plain-text list of stop words (one per line) excluded from query tokens.
- `examples/movies.json` — 20-movie sample of `data/movies.json` for quick testing without the full dataset.
- `examples/stopwords.txt` — Stop words file; copy to `data/stopwords.txt` before running the CLI.
- `tests/` — Pytest test suite mirroring the `cli/` package structure. 100% coverage is enforced by the pre-commit hook.
- `docs/` — Sphinx documentation. `conf.py` configures Furo theme, MyST-Parser (Markdown), autodoc, and typehints. `index.md` is the root; `usage.md` covers setup and CLI usage; `api/` contains per-module autodoc pages. Build output goes to `docs/_build/` (git-ignored).

The planned architecture is a RAG pipeline: keyword retrieval (BM25) as the first stage, followed by embedding-based semantic retrieval or re-ranking, with an LLM generating the final answer.
