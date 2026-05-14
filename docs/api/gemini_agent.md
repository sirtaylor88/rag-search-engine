# Gemini Agent

Thin wrapper around the [Google Gemini API](https://ai.google.dev/) for
query enhancement, result re-ranking, and augmented generation.

## Image-enhanced query rewriting

`describe_image` accepts raw image bytes, a MIME type, and a text query. It
builds a three-part multimodal prompt from ``DESCRIBE_IMAGE_PATTERN``,
the image, and the query, sends them to Gemini, and returns the raw
``types.GenerateContentResponse`` so the caller can access ``.text`` and
``.usage_metadata``.

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

`rerank_query` selects a prompt from `RankingPromptPattern` and asks Gemini
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

`evaluate_result` uses `RankingPromptPattern.EVALUATE` to ask Gemini to rate
each retrieved result on a 0–3 relevance scale. It joins the formatted result
strings with newlines, sends them alongside the query, and returns a JSON list
of integer scores — one per result, in the same order — or `None` if the model
returns nothing. Callers parse the response with `json.loads()`.

The three scale points:

- **3** — Highly relevant
- **2** — Relevant
- **1** — Marginally relevant
- **0** — Not relevant

## Augmented generation

`augment_result` selects a prompt from `AugmentedGenerationPromptPattern` based
on the `method` argument, joins the formatted result strings with newlines as
the `doc_input` context, sends them alongside the query, and returns the model's
answer text — or `None` if the model returns nothing. Four methods are available:

- **`rag`** — grounded natural-language answer that directly addresses the
  query using only the retrieved documents.
- **`summarize`** — information-dense multi-document synthesis covering genre,
  plot, and other key details to help users choose between options.
- **`citations`** — cited answer using `[1]`, `[2]` markers; acknowledges gaps
  when the retrieved documents don't contain enough information.
- **`question`** — casual, conversational Q&A answer in a direct chat-style
  tone; instructs the model to avoid hype and talk like a normal person.

All four public functions (`enhance_query`, `rerank_query`, `evaluate_result`,
`augment_result`) delegate token-count logging to the private
`_display_token_usage` helper, which logs prompt and response token counts at
`INFO` level via `logging.getLogger(__name__)`.

```{eval-rst}
.. note::

   ``GEMINI_API_KEY`` must be set in the environment (or in a ``.env`` file) before
   calling any function in this module.

.. automodule:: cli.api.gemini_agent
   :members:
   :undoc-members:
   :show-inheritance:

.. seealso::

   :doc:`/api/augmented_generation_cli` — CLI that dispatches to ``augment_result``

   :doc:`/api/commands/search/augmented_generation_commands` — ``BaseAugmentedCommand`` that calls ``augment_result``

   :doc:`/api/commands/search/hybrid_search_command` — ``RRFSearchCommand`` that calls ``enhance_query`` and ``rerank_query``

   :doc:`/api/constants` — ``GEMINI_MODEL``
```
