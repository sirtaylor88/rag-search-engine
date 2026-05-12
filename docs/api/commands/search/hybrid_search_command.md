# Hybrid Search Commands

Command classes for the `weighted-search` and `rrf-search` subcommands. Both
inherit from `BaseHybridSearchCommand[P]`, which owns the shared
load → search → print loop via three hooks:

- `_get_query(payload)` — resolves the final query string (default: `payload.query`; overridden by `RRFSearchCommand` to call `enhance_query`).
- `_search(hs, payload, query)` — calls the appropriate {class}`~cli.core.hybrid_search.HybridSearch` method; `RRFSearchCommand` optionally re-ranks results via `rerank_query` when `--rerank-method individual` is set.
- `_format_scores(result)` — formats the per-result score line.

`RRFSearchCommand` accepts three optional flags beyond `--k` and `--limit`:

- `--enhance {spell,rewrite,expand}` — query enhancement via `enhance_query` before retrieval.
- `--rerank-method {individual}` — re-ranks the top `5 × limit` candidates by Gemini relevance score and returns the top `limit`.

```{eval-rst}
.. automodule:: cli.commands.search.hybrid_search_commands
   :members:
   :undoc-members:
   :show-inheritance:

.. seealso::

   :doc:`/api/hybrid_search` — ``HybridSearch`` singleton called by both commands

   :doc:`/api/gemini_agent` — ``enhance_query`` and ``rerank_query`` used by ``RRFSearchCommand``

   :doc:`/api/schemas` — ``WeightedSearchPayload`` and ``RRFSearchPayload``
```
