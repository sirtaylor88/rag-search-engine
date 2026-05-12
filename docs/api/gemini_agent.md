# Gemini Agent

Thin wrapper around the [Google Gemini API](https://ai.google.dev/) for
query enhancement and result re-ranking.

## Query enhancement

`enhance_query` selects a prompt from `EnhancePromptPattern` based on the
`method` argument and sends the query to Gemini. Three methods are available:

- **`spell`** — corrects only high-confidence typos, leaving the rest
  of the query unchanged.
- **`rewrite`** — expands a vague query into a concise, Google-style search
  phrase using common movie knowledge and genre conventions.
- **`expand`** — appends synonyms and related concepts to improve recall.

Passing `method=None` (the default) returns the original query immediately
without any API call.

## Result re-ranking

`rerank_query` selects a prompt from `ReRankPromptPattern` and asks Gemini
to score a single document's relevance to the query on a 0–10 scale.
Currently one method is available:

- **`individual`** — rates each document independently against the query.

Passing `method=None` returns `0.0` immediately without any API call.

```{eval-rst}
.. note::

   ``GEMINI_API_KEY`` must be set in the environment (or in a ``.env`` file) before
   calling any function in this module.

.. automodule:: cli.api.gemini_agent
   :members:
   :undoc-members:
   :show-inheritance:

.. seealso::

   :doc:`/api/commands/search/hybrid_search_command` — ``RRFSearchCommand`` that calls ``enhance_query`` and ``rerank_query``

   :doc:`/api/constants` — ``GEMINI_MODEL``
```
