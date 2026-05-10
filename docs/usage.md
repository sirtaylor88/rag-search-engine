# Usage

## Setup

[Python 3.12.9](https://www.python.org/) and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run pre-commit install
```

## Dataset

Create a `data/` folder and copy the stop words file:

```bash
mkdir -p data
cp examples/stopwords.txt data/
```

Download the full movie dataset (~25 MB):

```bash
curl -o data/movies.json https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json
```

## CLI Commands

### Build the index

```bash
uv run python cli/keyword_search_cli.py build
uv run python cli/keyword_search_cli.py build --data-path examples/movies.json
```

### Keyword search

```bash
uv run python cli/keyword_search_cli.py search "<query>"
```

### BM25 search

```bash
uv run python cli/keyword_search_cli.py bm25search "<query>"
```

### Term frequency

```bash
uv run python cli/keyword_search_cli.py tf <doc_id> <term>
```

### Inverse document frequency

```bash
uv run python cli/keyword_search_cli.py idf <term>
```

### TF-IDF score

```bash
uv run python cli/keyword_search_cli.py tfidf <doc_id> <term>
```

### BM25 IDF score

```bash
uv run python cli/keyword_search_cli.py bm25idf <term>
```

### BM25 TF score

```bash
uv run python cli/keyword_search_cli.py bm25tf <doc_id> <term> [k1] [b]
```

## Semantic Search CLI

### Verify model

```bash
uv run python cli/semantic_search_cli.py verify
```

### Verify embeddings

```bash
uv run python cli/semantic_search_cli.py verify_embeddings
```

### Embed text

```bash
uv run python cli/semantic_search_cli.py embed_text "<text>"
```

### Embed query

```bash
uv run python cli/semantic_search_cli.py embed_query "<query>"
```

### Embed corpus chunks

```bash
uv run python cli/semantic_search_cli.py embed_chunks
```

### Semantic search

```bash
uv run python cli/semantic_search_cli.py search "<query>"
```

### Chunked semantic search

```bash
uv run python cli/semantic_search_cli.py search_chunked "<query>" [--limit N]
```

### Chunk text (word-based)

```bash
uv run python cli/semantic_search_cli.py chunk "<text>" [--chunk-size N] [--overlap N]
```

### Chunk text (sentence-based)

```bash
uv run python cli/semantic_search_cli.py semantic_chunk "<text>" [--max-chunk-size N] [--overlap N]
```

## Development

```bash
uv run pytest                                         # run tests
uv run pytest --cov=cli --cov-report=term-missing    # with coverage
uv run mypy .                                         # type check
uv run ruff check .                                   # lint
uv run ruff format .                                  # format
uv run pydocstyle cli/ --convention=google            # docstring style
uv run bandit -r cli/                                 # security scan
```
