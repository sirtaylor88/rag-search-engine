# RAG Search Engine

A search engine built with **Retrieval Augmented Generation (RAG)** over a movie
dataset. Implements keyword search via an inverted index with
[Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) scoring, semantic search
via dense [sentence embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
and hybrid retrieval combining both signals through weighted scoring or
[Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).
Optional query enhancement via the Gemini API corrects spelling before retrieval.

## Features

- **Keyword search** — inverted index with unranked token overlap or Okapi BM25 ranking
- **Semantic search** — cosine similarity over corpus-level and sentence-chunk embeddings
- **Hybrid search** — weighted `alpha * BM25 + (1 − alpha) * semantic` or Reciprocal Rank Fusion
- **Query enhancement** — Gemini-powered spell correction via `--enhance spell`

```{toctree}
:maxdepth: 2
:caption: Contents

usage
api/index
```
