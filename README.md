# RAG Search Engine

![Python](https://img.shields.io/badge/python-3.12.9-blue?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet?logo=astral)
![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Ruff](https://img.shields.io/badge/lint-ruff-orange?logo=ruff)

A search engine built with **Retrieval Augmented Generation (RAG)**. The current focus is keyword search over a movie dataset using an inverted index with [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) scoring. RAG-based semantic search is planned for the next phase.

---

## Table of Contents

- [Setup](#setup)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Build the inverted index](#build-the-inverted-index)
  - [Keyword search](#keyword-search)
  - [BM25 search](#bm25-search)
  - [Term frequency](#term-frequency)
  - [Inverse document frequency](#inverse-document-frequency)
  - [TF-IDF score](#tf-idf-score)
  - [BM25 IDF score](#bm25-idf-score)
  - [BM25 TF score](#bm25-tf-score)
  - [Verify semantic model](#verify-semantic-model)
  - [Verify embeddings](#verify-embeddings)
  - [Embed text](#embed-text)
  - [Embed query](#embed-query)
  - [Embed corpus chunks](#embed-corpus-chunks)
  - [Semantic search](#semantic-search)
  - [Chunked semantic search](#chunked-semantic-search)
  - [Chunk text](#chunk-text)
  - [Semantic chunk text](#semantic-chunk-text)
- [Documentation](#documentation)
- [Development](#development)

---

## Setup

[Python 3.12.9](https://www.python.org/) and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run pre-commit install
```

---

## Dataset

Create a `data/` folder and copy the stop words file from the `examples/` folder:

```bash
mkdir -p data
cp examples/stopwords.txt data/
```

Download the full movie dataset (~25 MB):

```bash
curl -o data/movies.json https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json
```

---

## Usage

### Build the inverted index

Build and cache the index from the movie dataset before searching. Tokenization runs in parallel using a thread pool:

```bash
uv run python cli/keyword_search_cli.py build
```

To use the sample dataset instead:

```bash
uv run python cli/keyword_search_cli.py build --data-path examples/movies.json
```

### Keyword search

Returns documents whose tokens overlap with the query (unranked):

```bash
uv run python cli/keyword_search_cli.py search "<query>"
```

```
$ uv run python cli/keyword_search_cli.py search "the dark knight"
Searching for: the dark knight
1. The Dark Knight
2. The Dark Knight Rises
```

### BM25 search

Returns documents ranked by cumulative [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) score:

```bash
uv run python cli/keyword_search_cli.py bm25search "<query>"
```

```
$ uv run python cli/keyword_search_cli.py bm25search "the dark knight"
Searching for: the dark knight
1. (1) The Dark Knight - Score: 8.50
2. (2) The Dark Knight Rises - Score: 7.20
```

### Term frequency

Look up how many times a term appears in a specific document:

```bash
uv run python cli/keyword_search_cli.py tf <doc_id> <term>
```

```
$ uv run python cli/keyword_search_cli.py tf 1 knight
The term frequency of ``knight`` in document 1 is 2
```

### Inverse document frequency

Compute the smoothed [inverse document frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Inverse_document_frequency) of a term across the full corpus:

```bash
uv run python cli/keyword_search_cli.py idf <term>
```

```
$ uv run python cli/keyword_search_cli.py idf knight
Inverse document frequency of 'knight': 3.45
```

### TF-IDF score

Compute the [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) score for a term in a specific document:

```bash
uv run python cli/keyword_search_cli.py tfidf <doc_id> <term>
```

```
$ uv run python cli/keyword_search_cli.py tfidf 1 knight
TF-IDF score of 'knight' in document '1': 6.91
```

### BM25 IDF score

Compute the [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) inverse document frequency for a term:

```bash
uv run python cli/keyword_search_cli.py bm25idf <term>
```

```
$ uv run python cli/keyword_search_cli.py bm25idf knight
BM25 IDF score of 'knight': 3.37
```

### BM25 TF score

Compute the [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) saturated term frequency for a term in a specific document:

```bash
uv run python cli/keyword_search_cli.py bm25tf <doc_id> <term> [k1] [b]
```

```
$ uv run python cli/keyword_search_cli.py bm25tf 1 knight
BM25 TF score of 'knight' in document '1': 1.60
```

The optional `k1` parameter (default: `1.5`) controls term frequency saturation. The optional `b` parameter (default: `0.75`) controls document length normalization.

### Verify semantic model

Load and verify the [sentence-transformers](https://www.sbert.net/) model (`all-MiniLM-L6-v2`). Downloads automatically on first use:

```bash
uv run python cli/semantic_search_cli.py verify
```

```
$ uv run python cli/semantic_search_cli.py verify
Model loaded: SentenceTransformer('all-MiniLM-L6-v2')
Max sequence length: 256
```

### Verify embeddings

Load or create dense embeddings for the full movie corpus and print their shape. Embeddings are cached to `cache/movie_embeddings.npy` on first run:

```bash
uv run python cli/semantic_search_cli.py verify_embeddings
```

```
$ uv run python cli/semantic_search_cli.py verify_embeddings
Number of docs:   9000
Embeddings shape: 9000 vectors in 384 dimensions
```

### Embed text

Encode a text string into a dense embedding vector using `all-MiniLM-L6-v2` and print the first 3 dimensions and total embedding size:

```bash
uv run python cli/semantic_search_cli.py embed_text "<text>"
```

```
$ uv run python cli/semantic_search_cli.py embed_text "the dark knight"
Text: the dark knight
First 3 dimensions: [-0.0123  0.0456 -0.0789]
Dimensions: 384
```

### Embed query

Encode a query string and print its full embedding shape (useful for inspecting query representations before semantic search):

```bash
uv run python cli/semantic_search_cli.py embed_query "<query>"
```

```
$ uv run python cli/semantic_search_cli.py embed_query "the dark knight"
Query: the dark knight
First 3 dimensions: [-0.0123  0.0456 -0.0789]
Shape: (384,)
```

### Embed corpus chunks

Split each movie description into overlapping sentence-level chunks and encode them. Chunk embeddings are cached to `cache/chunk_embeddings.npy` and metadata to `cache/chunk_metadata.json` on first run:

```bash
uv run python cli/semantic_search_cli.py embed_chunks
```

```
$ uv run python cli/semantic_search_cli.py embed_chunks
Generated 72909 chunked embeddings
```

### Semantic search

Search the movie corpus using dense embeddings ranked by [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity). The [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model encodes both the query and corpus. Embeddings are loaded from cache or built on first run:

```bash
uv run python cli/semantic_search_cli.py search "<query>"
```

```
$ uv run python cli/semantic_search_cli.py search "superhero battles villain"
Searching for: superhero battles villain
1. The Dark Knight (score: 0.7823)
  Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon...
2. Spider-Man (score: 0.7541)
  When bitten by a genetically altered spider, nerdy high school student Peter...
```

### Chunked semantic search

Search the movie corpus using sentence-level chunk embeddings. Each document is scored by the maximum [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) of its chunks to the query, so a highly relevant passage in a long description can surface the document even if the rest is off-topic. Chunk embeddings are loaded from cache or built on first run:

```bash
uv run python cli/semantic_search_cli.py search_chunked "<query>" [--limit N]
```

```
$ uv run python cli/semantic_search_cli.py search_chunked "superhero action movie" --limit 3
Searching for: superhero action movie

1. Kick-Ass (score: 0.6386)
   Dave Lizewski (Aaron Taylor-Johnson) opens the film with a narration about how superhe...

2. The Incredibles (score: 0.5386)
   The film opens with a series of short interviews between three famous superheroes incl...

3. Logan (score: 0.528)
   The film is preceded by a short film:On the streets of New York City, a mugging is ta...
```

### Chunk text

Split a text string into fixed-size word chunks with optional overlap. Useful for preparing long documents for embedding:

```bash
uv run python cli/semantic_search_cli.py chunk "<text>" [--chunk-size N] [--overlap N]
```

```
$ uv run python cli/semantic_search_cli.py chunk "The bear attack was very terrifying." --chunk-size 4 --overlap 1
Chunking 36 characters.
1. The bear attack was
2. was very terrifying.
```

The optional `--chunk-size` parameter (default: `200`) controls the number of words per chunk. The optional `--overlap` parameter (default: `0`) controls how many words are shared between consecutive chunks.

### Semantic chunk text

Split a text string into fixed-size sentence chunks using punctuation boundaries (`!`, `.`, `?`) with optional overlap. Useful for preparing long documents for sentence-level embedding:

```bash
uv run python cli/semantic_search_cli.py semantic_chunk "<text>" [--max-chunk-size N] [--overlap N]
```

```
$ uv run python cli/semantic_search_cli.py semantic_chunk "This is the first sentence. This is the second sentence. This is the third sentence." --max-chunk-size 2
Semantically chunking 85 characters.
1. This is the first sentence. This is the second sentence.
2. This is the third sentence.
```

The optional `--max-chunk-size` parameter (default: `4`) controls the maximum number of sentences per chunk. The optional `--overlap` parameter (default: `0`) controls how many sentences are shared between consecutive chunks.

---

## Documentation

Build the HTML docs locally with [Sphinx](https://www.sphinx-doc.org/):

```bash
uv run sphinx-build -b html docs docs/_build/html
```

Or serve with live reload at <http://127.0.0.1:8000>:

```bash
uv run sphinx-autobuild docs docs/_build/html
```

---

## Development

| Command | Description |
|---|---|
| `uv run pytest` | Run tests ([pytest](https://docs.pytest.org/)) |
| `uv run pytest --cov=cli --cov-report=term-missing` | Run tests with coverage report ([pytest-cov](https://pytest-cov.readthedocs.io/)) |
| `uv run mypy .` | Type check ([mypy](https://mypy-lang.org/)) |
| `uv run ruff check .` | Lint ([ruff](https://docs.astral.sh/ruff/)) |
| `uv run ruff format .` | Format ([ruff](https://docs.astral.sh/ruff/)) |
| `uv run pylint <file_or_dir>` | Lint with [pylint](https://pylint.readthedocs.io/) |
| `uv run bandit -r cli/` | Security scan ([bandit](https://bandit.readthedocs.io/)) |
| `uv run pydocstyle cli/ --convention=google` | Docstring style check ([pydocstyle](https://www.pydocstyle.org/)) |

[pre-commit](https://pre-commit.com/) hooks run `ruff check`, `ruff format`, `pylint`, `mypy`, `pydocstyle`, `bandit`, and `pytest` (with 100% coverage enforcement) automatically on each commit.
