# Augmented Generation CLI

Entry point for `augmented_generation_cli.py`. Registers two subcommands:

- **`rag`** via {class}`~cli.commands.search.augmented_generation_commands.RagCommand` — generates a grounded natural-language answer.
- **`summarize`** via {class}`~cli.commands.search.augmented_generation_commands.SummarizeCommand` — generates an information-dense multi-document summary.

Both subcommands accept:

- **`query`** (positional) — the search query.
- **`--limit`** (optional, default `SEARCH_LIMIT`) — number of results to
  retrieve and pass as context to the model.

Each command's `run()` performs the same pipeline (implemented in
{class}`~cli.commands.search.augmented_generation_commands.BaseAugmentedCommand`):

1. Load the movie corpus via `load_movies()`.
2. Retrieve the top `limit` results using
   {class}`~cli.core.hybrid_search.HybridSearch` RRF search (`k=DEFAULT_K`).
3. Print each retrieved title under a `"Search results:"` banner.
4. Pass the formatted results to
   {func}`~cli.api.gemini_agent.augment_result` and print the generated answer
   under the command's response banner (`"RAG Response:"` or `"LLM Summary:"`).

```{eval-rst}
.. automodule:: cli.augmented_generation_cli
   :members:
   :undoc-members:

.. seealso::

   :doc:`/api/commands/search/augmented_generation_commands` — ``BaseAugmentedCommand``, ``RagCommand``, and ``SummarizeCommand`` implementations

   :doc:`/api/gemini_agent` — ``augment_result`` used to generate the answer

   :doc:`/api/hybrid_search` — ``HybridSearch.rrf_search`` used for retrieval

   :doc:`/api/schemas` — ``SearchPayload`` and ``SearchRequest``
```
