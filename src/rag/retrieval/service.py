"""Retriever service."""

from __future__ import annotations

from rag.domain.models import RetrievedChunk
from rag.ports import Embedder, VectorStore


def retrieve_top_k(query: str, embedder: Embedder, vector_store: VectorStore, *, top_k: int) -> list[RetrievedChunk]:
    """Embed one query and return the top-k retrieved chunks."""
    embedding = embedder.embed_query(query)
    return vector_store.search(embedding, top_k)


class CodeRetriever:
    """Embeds a query and retrieves nearest code chunks."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore, *, top_k: int) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        return retrieve_top_k(query, self.embedder, self.vector_store, top_k=top_k or self.top_k)
