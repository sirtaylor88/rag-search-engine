# Usage

## Setup

[Python 3.12.9](https://www.python.org/) and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run pre-commit install
```

## Dataset

Create a `data/` folder and copy the files from the `examples/` folder:

```bash
mkdir -p data
cp examples/* data/
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

## Hybrid Search CLI

### Normalize scores

```bash
uv run python cli/hybrid_search_cli.py normalize <score1> <score2> ...
```

### Weighted search

```bash
uv run python cli/hybrid_search_cli.py weighted-search "<query>" [--alpha A] [--limit N]
```

### RRF search

```bash
uv run python cli/hybrid_search_cli.py rrf-search "<query>" [--k K] [--limit N] [--enhance {spell,rewrite,expand}] [--rerank-method {individual,batch,cross_encoder}] [--evaluate] [-v]
```

Pass `--enhance` to send the query through the Gemini API before retrieval.
Three methods are available: `spell` (typo correction), `rewrite` (Google-style
query rewrite), and `expand` (synonym/related-term expansion). Requires
`GEMINI_API_KEY` set in `.env`.

Pass `--rerank-method` to re-rank the top `5 × limit` RRF candidates before
returning the top `limit`. Three methods are available:

- `individual` — scores each candidate separately on a 0–10 scale via Gemini
  (one API call per result, with a 3 s delay between calls) and re-sorts by
  that score. Requires `GEMINI_API_KEY`.
- `batch` — sends all candidates in a single Gemini API call and asks the
  model to return a JSON-ordered list of IDs; falls back to the original RRF
  order if the response is empty. Requires `GEMINI_API_KEY`.
- `cross_encoder` — scores all query–document pairs locally using a
  `CrossEncoder` model (`cross-encoder/ms-marco-TinyBERT-L2-v2`) via
  `sentence-transformers`; no API key required.

Pass `--evaluate` to score each retrieved result on a 0–3 relevance scale via
Gemini after printing. Requires `GEMINI_API_KEY`.

Pass `-v` / `--verbose` to enable `DEBUG`-level logging for each pipeline stage:
original query, enhanced query (when `--enhance` is set), RRF candidate list,
and final results after re-ranking.

See {doc}`/api/gemini_agent` for details on the Gemini-based prompts and
{doc}`/api/commands/search/hybrid_search_command` for implementation details.

## Evaluation

Measure Precision@k, Recall@k, and F1 of RRF search over a golden dataset.
Each test case specifies a query and a set of relevant document titles; the
evaluator computes all three metrics and prints results sorted by F1 score
(descending):

```bash
uv run python cli/evaluation_cli.py [--limit K]
```

The `--limit` parameter (default: `5`) controls how many results are retrieved
per query. The golden dataset is read from `data/golden_dataset.json` (already
in place after the Dataset setup above). F1 is `0.0` when both precision and
recall are zero (no relevant results retrieved).

```
k=5 (Top results)
- Query: cute british bear marmalade
  - Precision@5: 1.0000
  - Recall@5: 1.0000
  - F1 Score: 1.0000
  - Retrieved: Paddington, ...
  - Relevant: Paddington
```

See {doc}`/api/evaluation_cli` for implementation details.

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
