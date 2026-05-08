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
  - [Embed text](#embed-text)
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
| `uv run pytest` | Run tests |
| `uv run pytest --cov=cli --cov-report=term-missing` | Run tests with coverage report |
| `uv run mypy .` | Type check |
| `uv run ruff check .` | Lint |
| `uv run ruff format .` | Format |
| `uv run pylint <file_or_dir>` | Lint with pylint |
| `uv run bandit -r cli/` | Security scan |

Pre-commit hooks run `ruff check`, `ruff format`, `pylint`, `mypy`, `bandit`, and `pytest` (with 100% coverage enforcement) automatically on each commit.
