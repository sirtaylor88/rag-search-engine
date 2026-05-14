# Multimodal Search

`MultimodalSearch` is a singleton wrapper around a CLIP-based
[sentence-transformers](https://www.sbert.net/) model. On first instantiation it
encodes all document titles and descriptions into `text_embeddings` so that an
image query can be matched against the corpus in a single pass.

## Initialisation

`__init__(documents, model_name)` loads the CLIP model and encodes every
document's `"{title}: {description}"` text into `self.text_embeddings`.
The default model is `clip-ViT-B-32` (512-dimensional embeddings).

## Image embedding

`embed_image(image_path)` opens the image via `PIL.Image.open`, encodes it
with the CLIP model, and returns a 1-D `npt.NDArray[Any]` via `np.asarray`.

## Image search

`search_with_image(image_path, limit)` embeds the image, computes cosine
similarity against each pre-built text embedding, and returns the top `limit`
results sorted by descending score. Each result dict contains `id`, `score`,
`title`, and `document` (full description).

## Verification helper

`verify_image_embedding(image_path)` is a module-level helper that calls
`load_movies()`, instantiates `MultimodalSearch(documents)`, embeds the image,
and prints `"Embedding shape: {N} dimensions"`. It is exposed as the
`verify_image_embedding` subcommand of {doc}`/api/multimodal_search_cli`.

```{eval-rst}
.. automodule:: cli.core.multimodal_search
   :members:
   :undoc-members:
   :show-inheritance:

.. seealso::

   :doc:`/api/multimodal_search_cli` — CLI that registers ``verify_image_embedding`` and ``image_search``

   :doc:`/api/commands/search/multimodal_search_commands` — ``VerifyImageEmbeddingCommand`` and ``ImageSearchCommand``

   :doc:`/api/semantic_search` — ``SemanticSearch`` — the text-only counterpart
```
