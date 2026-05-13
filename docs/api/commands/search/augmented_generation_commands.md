# RAG Command

`RagCommand` implements the full Retrieval Augmented Generation pipeline in a single
`run()` call:

1. Load the movie corpus via `load_movies()`.
2. Retrieve the top `limit` results using {class}`~cli.core.hybrid_search.HybridSearch`
   with Reciprocal Rank Fusion (`rrf_search`, `k=DEFAULT_K`).
3. Print each retrieved title under a `"Search results:"` banner.
4. Format each result as `"- {title} - {document}"` and pass the list to
   {func}`~cli.api.gemini_agent.augment_query` as context.
5. Print the generated answer under a `"RAG Response:"` banner (empty string when the
   model returns nothing).

`RagCommand` extends `BaseSearchCommand` and inherits the `query` positional argument
and `--limit` optional argument with no additional flags.

```{eval-rst}
.. automodule:: cli.commands.search.augmented_generation_commands
   :members:
   :undoc-members:
   :show-inheritance:

.. seealso::

   :doc:`/api/hybrid_search` — ``HybridSearch`` singleton used for RRF retrieval

   :doc:`/api/gemini_agent` — ``augment_query`` used to generate the grounded answer

   :doc:`/api/schemas` — ``SearchPayload`` and ``SearchRequest``
```
