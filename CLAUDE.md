# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A search engine built with Retrieval Augmented Generation (RAG) over a movie dataset. Implements keyword search via an inverted index with Okapi BM25 scoring and semantic search via dense sentence embeddings, including chunked retrieval that scores documents by their most relevant sentence.

## Setup

This project uses `uv` for dependency and environment management. Python 3.12.9 is required (see `.python-version`).

```bash
uv sync                   # install dependencies
uv run pre-commit install  # set up git hooks
```

## Commands

```bash
# Lint and format
uv run ruff check .          # lint (includes isort; use --fix to auto-sort imports)
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

# Embed corpus descriptions into sentence-level chunks (cached to cache/)
uv run python cli/semantic_search_cli.py embed_chunks

# Chunked semantic search: rank movies by max chunk cosine similarity
uv run python cli/semantic_search_cli.py search_chunked "<query>" [--limit N]

# Normalize a list of scores to [0, 1] using min-max scaling
uv run python cli/hybrid_search_cli.py normalize <score1> <score2> ...

# Hybrid weighted search: rank movies by alpha*BM25 + (1-alpha)*semantic (alpha default 0.5)
uv run python cli/hybrid_search_cli.py weighted-search "<query>" [--alpha A] [--limit N]

# Hybrid RRF search: rank movies by Reciprocal Rank Fusion of BM25 and semantic rankings (k default 60)
uv run python cli/hybrid_search_cli.py rrf-search "<query>" [--k K] [--limit N]

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

- `cli/hybrid_search_cli.py` — Hybrid search CLI entry point: registers `normalize`, `weighted-search`, and `rrf-search`; dispatches to `NormalizeCommand` (via `ScoreListRequest`), `WeightedSearchCommand` (via `WeightedSearchRequest`), and `RRFSearchCommand` (via `RRFSearchRequest`).
- `cli/keyword_search_cli.py` — CLI entry point: builds the `ArgumentParser`, instantiates each command with its subparser (registering arguments), then parses args, wraps them in the appropriate `Request[XPayload]` instance, and dispatches to `build`, `search`, `bm25search`, `tf`, `idf`, `tfidf`, `bm25idf`, or `bm25tf`.
- `cli/semantic_search_cli.py` — Semantic search CLI entry point: registers `verify`, `verify_embeddings`, `embed_text`, `embed_query`, `embed_chunks`, `search`, `search_chunked`, `chunk`, and `semantic_chunk` subcommands; dispatches to `VerifyCommand`, `VerifyEmbeddingsCommand`, `EmbedChunksCommand` (all via `EmptyRequest`), `EmbedTextCommand`, `EmbedQueryCommand` (both via `TermRequest`), `SemanticSearchCommand`, `SearchChunkedCommand` (both via `SearchRequest`), `ChunkCommand` (via `ChunkRequest`), and `SemanticChunkCommand` (via `SemanticChunkRequest`).
- `cli/singleton.py` — `Singleton` base class. `__new__` ensures only one instance exists per subclass (per-class `_instance` ClassVar). Sets `_initialized = False` on the fresh instance; subclasses guard `__init__` with `if self._initialized: return` and set `self._initialized = True` at the end of first-time setup.
- `cli/schemas/` — Pydantic `BaseModel` payload and request models, split into two submodules; `cli/schemas/__init__.py` re-exports everything so all `from cli.schemas import …` calls work unchanged.
  - `payloads.py` — **Payload models**: `EmptyPayload` (no fields), `SearchPayload` (`query` min_length=1, `limit` ge=1 default `SEARCH_LIMIT`), `WeightedSearchPayload` (extends `SearchPayload`, adds `alpha` ge=0 le=1 default `DEFAULT_ALPHA`), `RRFSearchPayload` (extends `SearchPayload`, adds `k` ge=1 default `DEFAULT_K`), `TermPayload` (`term` min_length=1), `OverlapPayload` (extends `TermPayload`, adds `overlap` ge=0 default `0`), `ChunkPayload` (extends `OverlapPayload`, adds `chunk_size` ge=1 default `CHUNK_SIZE`), `SemanticChunkPayload` (extends `OverlapPayload`, adds `max_chunk_size` ge=1 default `SEMANTIC_CHUNK_SIZE`), `TermWithDocIDPayload` (adds `doc_id` ge=1), `BM25Payload` (adds `k1` gt=0, `b` ge=0 le=1), `ScoreListPayload` (`scores: list[PositiveFloat]` where `PositiveFloat = Annotated[float, Field(gt=0)]`).
  - `requests.py` — **Request envelope**: `Request[T]` generic `BaseModel` with a single `payload: T` field; concrete aliases `EmptyRequest`, `SearchRequest`, `WeightedSearchRequest`, `RRFSearchRequest`, `TermRequest`, `ChunkRequest`, `SemanticChunkRequest`, `TermWithDocIDRequest`, `BM25Request`, `ScoreListRequest` narrow the type.
- `cli/commands/` — Command classes following an instance-based pattern.
  - `base.py` — Abstract command infrastructure. **`BaseCommand[PayloadT]`**: abstract base with `__init__(parser)`, abstract `add_arguments(parser)`, abstract `run(request: Request[PayloadT])`, concrete `load_cache()` (shared OSError handling), and `inverted_index` property returning the `InvertedIndex` singleton. Concrete bases: `TermCommand` (registers `term` positional arg), `BaseSearchCommand` (registers `query` positional arg and `--limit` optional arg), `BaseListCommand` (registers variadic `args` positional arg; `args_type` and `args_help` class attrs configure the argument).
  - `build_command.py` — `BuildCommand`: registers `--data-path` and builds/saves the index.
  - `embed_commands.py` — `BaseEmbedCommand(TermCommand)`: abstract base whose `run()` delegates to `_embed(term)`; subclasses implement `_embed`. `EmbedTextCommand(BaseEmbedCommand)`: calls `embed_text()` to encode the input term and print its embedding info. `EmbedQueryCommand(BaseEmbedCommand)`: calls `embed_query_text()` to encode a query string and print its full embedding shape. `EmbedChunksCommand(BaseCommand[EmptyPayload])`: no-arg command; `run()` calls `embed_chunks()` to load or create chunk embeddings for the corpus and print the count.
  - `verify_commands.py` — `BaseVerifyCommand(BaseCommand[EmptyPayload])`: abstract base whose `run()` delegates to `_verify()`; `add_arguments` is a no-op (no args needed). `VerifyCommand(BaseVerifyCommand)`: calls `verify_model()`. `VerifyEmbeddingsCommand(BaseVerifyCommand)`: calls `verify_embeddings()` to load or build corpus embeddings and print their shape.
  - `compute/normalize_command.py` — `NormalizeCommand(BaseListCommand)`: accepts a variadic list of positive floats; `run()` applies min-max scaling to each score and prints it as `* {value:.4f}`; when all scores are equal, prints `1.0000` (ZeroDivision guard); empty list produces no output.
  - `compute/chunk_commands.py` — `BaseChunkCommand(TermCommand, Generic[P])`: generic abstract base (`P bound OverlapPayload`) that registers `--overlap` (default `0`) and owns the full `run()` loop: prints label, delegates splitting to abstract `_split(term)`, delegates chunk-size lookup to abstract `_get_chunk_size(payload: P)`, then prints each chunk. `ChunkCommand(BaseChunkCommand[ChunkPayload])`: registers `--chunk-size` (default `CHUNK_SIZE`); `_split` returns `term.split()`; `_get_chunk_size` returns `payload.chunk_size`. `SemanticChunkCommand(BaseChunkCommand[SemanticChunkPayload])`: registers `--max-chunk-size` (default `SEMANTIC_CHUNK_SIZE`); `_split` splits by `SENTENCE_SPLIT_PATTERN`; `_get_chunk_size` returns `payload.max_chunk_size`.
  - `search/` — Subpackage for keyword, BM25, and semantic search commands.
    - `keyword_search_commands.py` — `BaseKeywordSearchCommand(BaseSearchCommand)`: abstract base whose `run()` loads the cache, prints the query banner, and delegates to abstract `_run_search(query, limit)`. `SearchCommand(BaseKeywordSearchCommand)`: `_run_search` calls `InvertedIndex.search()` and prints matching doc titles (unranked). `BM25SearchCommand(BaseKeywordSearchCommand)`: `_run_search` calls `InvertedIndex.bm25_search()` and prints results ranked by BM25 score.
    - `semantic_search_command.py` — `BaseSemanticSearchCommand(BaseSearchCommand)`: abstract base whose `run()` loads movies, calls abstract `_prepare(documents)`, prints the query banner, calls abstract `_search(query, limit)`, and prints results via `_get_excerpt(result)` (returns `result["document"]`). `SemanticSearchCommand(BaseSemanticSearchCommand)`: `_prepare` loads corpus embeddings; `_search` calls `SemanticSearch.search()`. `SearchChunkedCommand(BaseSemanticSearchCommand)`: `_prepare` loads chunk embeddings; `_search` calls `ChunkedSemanticSearch.search_chunks()`.
    - `hybrid_search_commands.py` — `BaseHybridSearchCommand(BaseSearchCommand, Generic[P])`: generic abstract base (`P bound SearchPayload`) whose `run()` loads movies, instantiates `HybridSearch`, calls abstract `_search(hs, payload)`, prints the query banner, and iterates results via abstract `_format_scores(result)`. `WeightedSearchCommand(BaseHybridSearchCommand[WeightedSearchPayload])`: registers `--alpha`; `_search` calls `weighted_search`; `_format_scores` returns hybrid, BM25, and semantic scores. `RRFSearchCommand(BaseHybridSearchCommand[RRFSearchPayload])`: registers `--k`; `_search` calls `rrf_search`; `_format_scores` returns RRF score, BM25 rank, and semantic rank.
  - `compute/` — Subpackage for all scoring/frequency commands.
    - `tf_commands.py` — `BaseComputeTFCommand(TermCommand)`: abstract base that registers `doc_id` and `term` positional args. `ComputeTFCommand(BaseComputeTFCommand)`: prints raw term frequency via `get_tf()` with the format `"The term frequency of ``{term}`` in document {doc_id} is {tf}"`. `BaseTFScoreCommand[P](BaseComputeTFCommand, Generic[P])`: generic abstract base (`P bound TermWithDocIDPayload`) whose `run()` loads cache, calls abstract `_score(payload: P) -> float`, and prints `"{_label} of '{term}' in document '{doc_id}': {score:.2f}"`. `ComputeTFIDFCommand(BaseTFScoreCommand[TermWithDocIDPayload])`: `_score` returns `get_tf() * get_idf()`. `ComputeBM25TFCommand(BaseTFScoreCommand[BM25Payload])`: registers optional `k1` and `b` args; `_score` returns `get_bm25_tf()`.
    - `idf_commands.py` — `BaseComputeIDFCommand(TermCommand)`: abstract base whose `run()` loads the cache, calls abstract `_compute_idf(term) -> float`, and prints `"{_label} of '{term}': {value:.2f}"`. `ComputeIDFCommand(BaseComputeIDFCommand)`: `_compute_idf` returns `get_idf(term)`. `ComputeBM25IDFCommand(BaseComputeIDFCommand)`: `_compute_idf` returns `get_bm25_idf(term)`.
- `cli/core/semantic_search.py` — `SemanticSearch(Singleton)` class wrapping a `SentenceTransformer` model. `__init__(model_name=DEFAULT_EMBEDDING_MODEL)` loads the model (defaults to `"all-MiniLM-L6-v2"`). Methods: `generate_embedding(text)` → 1-D `npt.NDArray[Any]` via `np.asarray(encode()[0])` (raises `ValueError` on empty input); `build_embeddings(documents)` → encodes all docs via `np.asarray(encode(...))` and saves to `cache/movie_embeddings.npy`; `load_or_create_embeddings(documents)` → loads from disk if count matches, otherwise rebuilds; `search(query, limit)` → encodes the query, scores all documents by cosine similarity, and returns the top `limit` results as `list[dict]` with `id`, `score`, `title`, and `document` (first 100 chars of description) keys. Module-level helpers: `verify_model()` prints model info; `embed_text(text)` prints embedding dimensions; `embed_query_text(query)` prints query embedding shape; `verify_embeddings()` loads or builds corpus embeddings and prints their shape. `ChunkedSemanticSearch(SemanticSearch)`: extends with sentence-level chunking. `build_chunk_embeddings(documents)` → splits each doc's description into overlapping sentence chunks, encodes all chunks, and saves to `cache/chunk_embeddings.npy` + `cache/chunk_metadata.json`; `load_or_create_chunk_embeddings(documents)` → loads both files if present, otherwise rebuilds; `_load_movies_idx_from_chunk_idx(flat_idx)` → maps flat chunk array index to its document's position in `self.documents` via `chunk_metadata[flat_idx]["doc_id"]`; `search_chunks(query, limit)` → scores every chunk, aggregates max score per document, and returns the top `limit` results with `id`, `title`, `document`, `score`, and `metadata` keys. `ChunkMetadata` TypedDict: `{doc_id, chunk_idx, total_chunks}`. Module-level helper: `embed_chunks()` loads or creates chunk embeddings and prints the count.
- `cli/core/hybrid_search.py` — Module-level helpers: `normalize_scores(scores)` applies min-max scaling to a list of floats, returning `[]` on empty input and `1.0` for all-equal values; `hybrid_score(bm25_score, semantic_score, alpha)` computes `alpha * bm25 + (1 - alpha) * semantic`; `rrf_score(rank, k)` computes `1 / (k + rank)` for Reciprocal Rank Fusion. `HybridSearch(Singleton)`: `__init__(documents)` stores the corpus, instantiates `ChunkedSemanticSearch` and loads chunk embeddings, instantiates `InvertedIndex` and builds+saves the index if no cache exists; `_bm25_search(query, limit)` loads the index and returns BM25 results; `_fetch_candidates(query, sample_limit)` calls both retrievers and returns `(raw_bm25, raw_semantic, doc_ids)` — the shared preamble for both search methods; `weighted_search(query, alpha, limit)` fetches candidates, min-max normalises each score list to [0, 1] across the full candidate set (documents absent from a retriever receive 0.0 before normalisation), computes `hybrid_score` on the normalised values, and returns the top `limit` results as `list[dict]` with `id`, `title`, `document` (full description), `bm25_score`, `semantic_score` (both normalised to [0, 1]), and `hybrid_score` keys; `rrf_search(query, k, limit)` fetches candidates, assigns each document a rank in each list, sums `rrf_score(rank, k)` contributions (documents absent from a retriever contribute 0.0), and returns the top `limit` results with `id`, `title`, `document`, `bm25_rank`, `semantic_rank` (both `None` if absent from that retriever), and `rrf_score` keys.
- `cli/core/keyword_search.py` — `InvertedIndex(Singleton)`: inherits singleton behaviour from `Singleton`; `__init__` is guarded by `_initialized` so it runs only on the first instantiation. Builds a token→doc-ID index using `ThreadPoolExecutor` (tokenization runs in parallel; a `threading.Lock` serializes index writes), tracks per-document term frequencies and lengths (`doc_lengths`, `avg_doc_length`), supports `get_documents(term)`, `search(query, limit)` (unranked token-overlap), `get_tf(doc_id, term)`, `get_idf(term)` (smoothed log IDF), `get_bm25_idf(term)` (Okapi BM25 IDF), `get_bm25_tf(doc_id, term, k1, b)` (length-normalized BM25 TF), `bm25(doc_id, term)` (full BM25 score = BM25 TF × BM25 IDF), `bm25_search(query, limit)` (ranked by cumulative BM25), and persists to/loads from `cache/` via pickle. `Document` is a `TypedDict` for movie records.
- `cli/utils.py` — Text processing and data loading helpers: `load_movies(data_path)` (loads the movies JSON and returns `list[Document]`); `remove_all_punctuations` (strips ASCII and common Unicode punctuation: curly quotes, en/em dashes), `tokenize_text`, `get_stemmed_tokens` (Porter stemmer via NLTK; filters stop words; returns ordered list with duplicates), `get_stop_words` (loads from `data/stopwords.txt`), `get_term_token` (validates and stems a single-word term), `cosine_similarity(vec1, vec2)` (returns cosine similarity in [-1, 1] between two `npt.NDArray[Any]` vectors; returns 0.0 if either norm is zero), `get_overlapping_chunks(text_parts, chunk_size, overlap)` (splits a list of text parts into overlapping fixed-size chunks; omits a trailing chunk whose elements are entirely covered by the previous chunk's overlap; raises `ValueError` if `overlap >= chunk_size`), `get_sentences(text)` (splits text at `!`, `.`, or `?` boundaries; strips and filters empty sentences; returns `[]` for blank input), `timer` (context manager that prints elapsed wall-clock time to stderr), the shared `STEMMER` instance, and `STOP_WORDS` (loaded once at import time).
- `cli/constants.py` — Project-wide constants: `BM25_K1` (default BM25 saturation parameter, `1.5`), `BM25_B` (default length normalization parameter, `0.75`), `CACHE_DIR` (cache directory name, `"cache"`), `CACHE_DIR_PATH` (`Path(CACHE_DIR)`), `SEARCH_LIMIT` (default result count, `5`), `CHUNKED_SEARCH_LIMIT` (result count for chunked searches, `10`), `DEFAULT_ALPHA` (default BM25 weight for hybrid scoring, `0.5`), `DEFAULT_K` (default RRF smoothing constant, `60`), `CHUNK_SIZE` (default words per chunk, `200`), `SEMANTIC_CHUNK_SIZE` (default sentences per semantic chunk, `4`), `SENTENCE_SPLIT_PATTERN` (regex for sentence boundary splitting, `r"(?<=[.!?])\s+"`), `DEFAULT_EMBEDDING_MODEL` (sentence-transformers model name, `"all-MiniLM-L6-v2"`), `SCORE_PRECISION` (decimal places for displayed scores, `4`).
- `cache/` — Pickle files (`index.pkl`, `docmap.pkl`, `term_frequencies.pkl`, `doc_lengths.pkl`) written by the `build` command. Excluded from git.
- `data/movies.json` — Movie dataset (~25 MB) with fields: `id`, `title`, `description`, and more. Used as the corpus for search.
- `data/stopwords.txt` — Plain-text list of stop words (one per line) excluded from query tokens.
- `examples/movies.json` — 20-movie sample of `data/movies.json` for quick testing without the full dataset.
- `examples/stopwords.txt` — Stop words file; copy to `data/stopwords.txt` before running the CLI.
- `tests/` — Pytest test suite mirroring the `cli/` package structure. 100% coverage is enforced by the pre-commit hook.
- `docs/` — Sphinx documentation. `conf.py` configures Furo theme, MyST-Parser (Markdown), autodoc, and typehints. `index.md` is the root; `usage.md` covers setup and CLI usage; `api/` contains per-module autodoc pages. Build output goes to `docs/_build/` (git-ignored).

The next phase is a full RAG pipeline: keyword retrieval (BM25) and/or chunked semantic retrieval as the first stage, followed by an LLM generating a grounded answer from the retrieved context.
