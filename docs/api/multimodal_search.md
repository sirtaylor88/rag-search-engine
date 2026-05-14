# Multimodal Search

`MultimodalSearch` is a singleton wrapper around a CLIP-based
[sentence-transformers](https://www.sbert.net/) model. It encodes images into
dense embedding vectors that can be compared against text query embeddings for
cross-modal retrieval.

## Image embedding

`embed_image(file_path)` opens the image at `file_path` via
`PIL.Image.open`, encodes it with the CLIP model using
`SentenceTransformer.encode`, and returns the result as a 1-D
`npt.NDArray[Any]` via `np.asarray`. The default model is
`clip-ViT-B-32` (512-dimensional embeddings).

## Verification helper

`verify_image_embedding(file_path)` is a module-level helper that
instantiates `MultimodalSearch`, embeds the given image, and prints
`"Embedding shape: {N} dimensions"`. It is exposed as the
`verify_image_embedding` subcommand of
{doc}`/api/multimodal_search_cli`.

```{eval-rst}
.. automodule:: cli.core.multimodal_search
   :members:
   :undoc-members:
   :show-inheritance:

.. seealso::

   :doc:`/api/multimodal_search_cli` — CLI that calls ``verify_image_embedding``

   :doc:`/api/commands/verify` — ``VerifyImageEmbeddingCommand`` that dispatches to ``verify_image_embedding``

   :doc:`/api/semantic_search` — ``SemanticSearch`` — the text-only counterpart
```
