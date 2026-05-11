# Hybrid Search

Core retrieval logic that fuses BM25 keyword scores and chunked semantic scores
into a single ranked list. Implemented as a {class}`~cli.singleton.Singleton` so
the inverted index and chunk embeddings are loaded only once per process.

Two ranking strategies are supported:

- **Weighted search** — min-max normalises both score lists then combines them as
  `alpha × BM25 + (1 − alpha) × semantic`.
- **RRF search** — assigns each document `1 / (k + rank)` from each retriever and
  sums the contributions.

```{eval-rst}
.. automodule:: cli.core.hybrid_search
   :members:
   :undoc-members:
   :show-inheritance:

.. seealso::

   :doc:`/api/inverted_index` — BM25 keyword retrieval backend

   :doc:`/api/semantic_search` — chunked semantic retrieval backend

   :doc:`/api/hybrid_search_cli` — CLI entry point for weighted and RRF search
```
