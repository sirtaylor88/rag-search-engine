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
to score or rank documents against the query. Two methods are available:

- **`individual`** — rates each document independently on a 0–10 scale and
  returns the raw score string.
- **`batch`** — sends all candidate documents in one request and asks Gemini
  to return a JSON-ordered list of document IDs by relevance.

The return type is `Optional[str]` (the raw model response text) in both
cases. Callers are responsible for parsing the value — `float()` for
`individual`, `json.loads()` for `batch`.

Passing `method=None` returns `None` immediately without any API call.

## Result evaluation

`evaluate_query` uses `ReRankPromptPattern.EVALUATE` to ask Gemini to rate
each retrieved result on a 0–3 relevance scale. It joins the formatted result
strings with newlines, sends them alongside the query, and returns a JSON list
of integer scores — one per result, in the same order — or `None` if the model
returns nothing. Callers parse the response with `json.loads()`.

The three scale points:

- **3** — Highly relevant
- **2** — Relevant
- **1** — Marginally relevant
- **0** — Not relevant

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
