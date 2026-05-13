# Augmented Generation CLI

Entry point for `augmented_generation_cli.py`. Registers a single `rag`
subcommand via {class}`~cli.commands.search.augmented_generation_commands.RagCommand`
and dispatches to it with a `SearchRequest`.

The `rag` command accepts:

- **`query`** (positional) — the search query.
- **`--limit`** (optional, default `SEARCH_LIMIT`) — number of results to
  retrieve and pass as context to the model.

`RagCommand.run()` performs the full pipeline:

1. Load the movie corpus via `load_movies()`.
2. Retrieve the top `limit` results using
   {class}`~cli.core.hybrid_search.HybridSearch` RRF search (`k=DEFAULT_K`).
3. Print each retrieved title under a `"Search results:"` banner.
4. Pass the formatted results to
   {func}`~cli.api.gemini_agent.augment_query` and print the generated answer
   under a `"RAG Response:"` banner.

```{eval-rst}
.. automodule:: cli.augmented_generation_cli
   :members:
   :undoc-members:

.. seealso::

   :doc:`/api/commands/search/augmented_generation_commands` — ``RagCommand`` implementation

   :doc:`/api/gemini_agent` — ``augment_query`` used to generate the grounded answer

   :doc:`/api/hybrid_search` — ``HybridSearch.rrf_search`` used for retrieval
```
