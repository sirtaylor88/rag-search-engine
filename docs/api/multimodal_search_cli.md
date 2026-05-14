# Multimodal Search CLI

Entry point for `multimodal_search_cli.py`. Registers two subcommands:

- **`verify_image_embedding`** via
  {class}`~cli.commands.search.multimodal_search_commands.VerifyImageEmbeddingCommand` —
  embeds an image using the CLIP model and prints the embedding size.
- **`image_search`** via
  {class}`~cli.commands.search.multimodal_search_commands.ImageSearchCommand` —
  ranks movies by cosine similarity between the image embedding and each document's
  text embedding.

Both subcommands accept one positional argument:

- **`image_path`** — path to the image file.

```{eval-rst}
.. automodule:: cli.multimodal_search_cli
   :members:
   :undoc-members:

.. seealso::

   :doc:`/api/multimodal_search` — ``MultimodalSearch``, ``search_with_image``, and ``verify_image_embedding``

   :doc:`/api/commands/search/multimodal_search_commands` — ``VerifyImageEmbeddingCommand`` and ``ImageSearchCommand``
```
