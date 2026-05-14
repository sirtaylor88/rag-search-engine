# Multimodal Search Commands

`BaseMultimodalCommand` is the abstract base that registers the `image_path`
positional argument (stored as `term`) for both multimodal subcommands.

- **`VerifyImageEmbeddingCommand`** — calls `verify_image_embedding(image_path)`,
  which loads the corpus, embeds the image via the CLIP model, and prints
  `"Embedding shape: {N} dimensions"`.
- **`ImageSearchCommand`** — loads the corpus, instantiates
  {class}`~cli.core.multimodal_search.MultimodalSearch`, calls
  `search_with_image(image_path)`, and prints ranked results as
  `"{idx}. {title} (similarity: {score:.3f})\n   {excerpt}..."`.

Both are registered in {doc}`/api/multimodal_search_cli` via
`multimodal_search_cli.py`.

```{eval-rst}
.. automodule:: cli.commands.search.multimodal_search_commands
   :members:
   :undoc-members:
   :show-inheritance:

.. seealso::

   :doc:`/api/multimodal_search_cli` — CLI entry point that registers these commands

   :doc:`/api/multimodal_search` — ``MultimodalSearch`` and ``search_with_image``

   :doc:`/api/commands/verify` — ``VerifyCommand`` and ``VerifyEmbeddingsCommand``
```
