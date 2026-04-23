"""Embedding adapters."""

from __future__ import annotations

from collections.abc import Sequence


class SentenceTransformerEmbedder:
    """SentenceTransformers adapter that satisfies the Embedder protocol."""

    def __init__(self, model_name: str, *, normalize_embeddings: bool = True) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            list(texts),
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=len(texts) >= 128,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

