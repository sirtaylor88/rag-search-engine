# Augmented Generation Commands

`BaseAugmentedCommand` owns the full pipeline shared by all augmented-generation
subcommands:

1. Load the movie corpus via `load_movies()`.
2. Retrieve the top `limit` results using {class}`~cli.core.hybrid_search.HybridSearch`
   with Reciprocal Rank Fusion (`rrf_search`, `k=DEFAULT_K`).
3. Print each retrieved title under a `"Search results:"` banner.
4. Format each result as `"- {title} - {document}"` and pass the list to
   {func}`~cli.api.gemini_agent.augment_result` as context.
5. Print the generated answer under the command's `_label` banner (empty string when
   the model returns nothing).

Subclasses configure the pipeline via two class attributes:

- **`_method`** — key into `AugmentedGenerationPromptPattern` (selects the Gemini prompt).
- **`_label`** — response banner printed before the generated answer.

Both concrete subclasses extend `BaseSearchCommand` and inherit the `query` positional
argument and `--limit` optional argument with no additional flags.

| Command | `_method` | `_label` |
|---|---|---|
| `RagCommand` | `"rag"` | `"RAG Response:"` |
| `SummarizeCommand` | `"summarize"` | `"LLM Summary:"` |
| `CitationsCommand` | `"citations"` | `"LLM Answer:"` |

```{eval-rst}
.. automodule:: cli.commands.search.augmented_generation_commands
   :members:
   :undoc-members:
   :show-inheritance:

.. seealso::

   :doc:`/api/augmented_generation_cli` — CLI entry point that dispatches to these commands

   :doc:`/api/hybrid_search` — ``HybridSearch`` singleton used for RRF retrieval

   :doc:`/api/gemini_agent` — ``augment_result`` used to generate the grounded answer

   :doc:`/api/commands/base` — ``BaseSearchCommand`` that ``BaseAugmentedCommand`` extends

   :doc:`/api/schemas` — ``SearchPayload`` and ``SearchRequest``
```
