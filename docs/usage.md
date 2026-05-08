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

### Search

```bash
uv run python cli/keyword_search_cli.py search "<query>"
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
uv run python cli/keyword_search_cli.py bm25tf <doc_id> <term> [k1]
```

## Development

```bash
uv run pytest                                         # run tests
uv run pytest --cov=cli --cov-report=term-missing    # with coverage
uv run mypy .                                         # type check
uv run ruff check .                                   # lint
uv run ruff format .                                  # format
uv run bandit -r cli/                                 # security scan
```
