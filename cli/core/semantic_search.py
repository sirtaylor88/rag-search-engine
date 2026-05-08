"""Semantic search: loads a SentenceTransformer model for text encoding."""

from sentence_transformers import SentenceTransformer


def verify_model() -> None:
    """Instantiate SemanticSearch and print model info to verify it loaded correctly."""
    sem_search = SemanticSearch()
    print(f"Model loaded: {sem_search.model}")
    print(f"Max sequence length: {sem_search.model.max_seq_length}")


class SemanticSearch:
    """Wraps SentenceTransformer to encode text into dense embedding vectors."""

    def __init__(self) -> None:
        """Load the all-MiniLM-L6-v2 model (downloads automatically on first use)."""
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
