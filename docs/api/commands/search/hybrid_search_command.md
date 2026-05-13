# Hybrid Search Commands

Command classes for the `weighted-search` and `rrf-search` subcommands. Both
inherit from `BaseHybridSearchCommand[P]`, which owns the shared
load → search → print loop via three hooks:

- `_get_query(payload)` — resolves the final query string (default: `payload.query`; overridden by `RRFSearchCommand` to log the original query at `DEBUG` then call `enhance_query`, logging the enhanced result when `--enhance` is set).
- `_search(hs, payload, query)` — calls the appropriate {class}`~cli.core.hybrid_search.HybridSearch` method; `RRFSearchCommand` logs RRF results at `DEBUG` and optionally re-ranks results via `rerank_query` when `--rerank-method` is set, logging the final re-ranked results at `DEBUG`.
- `_format_result(idx, result, rerank_method)` — builds the multi-line display block for a single ranked result, including the optional re-rank score line for `individual`, `batch`, and `cross_encoder` methods.
- `_format_scores(result)` — formats the per-result score line.

`RRFSearchCommand` accepts five optional flags beyond `--k` and `--limit`:

- `--enhance {spell,rewrite,expand}` — query enhancement via `enhance_query` before retrieval.
- `--rerank-method {individual,batch,cross_encoder}` — re-ranks the top `5 × limit` candidates and returns the top `limit`:
  - `individual` — scores each document separately via {func}`~cli.api.gemini_agent.rerank_query` (one API call per result, with a 3 s sleep between calls) then sorts descending by score.
  - `batch` — sends all candidates in a single {func}`~cli.api.gemini_agent.rerank_query` call and asks Gemini to return a JSON-ordered list of IDs; falls back to original RRF order if the response is empty.
  - `cross_encoder` — scores all query–document pairs locally using a `CrossEncoder` model (`DEFAULT_CROSS_ENCODER_MODEL`) via `sentence-transformers`; no API key required.
- `--evaluate` — after printing results, calls {func}`~cli.api.gemini_agent.evaluate_query` to score each result 0–3 for relevance and prints `{title}: {score}/3` per result. Requires `GEMINI_API_KEY`.
- `-v` / `--verbose` — enables `DEBUG`-level logging for each pipeline stage: original query, enhanced query (when `--enhance` is set), RRF candidates, and final results after re-ranking.

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
