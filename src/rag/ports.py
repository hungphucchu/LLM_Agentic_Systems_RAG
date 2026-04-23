"""Small protocols that keep core logic independent from vendor SDKs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

from rag.domain.models import CodeDocument, RetrievedChunk


class Embedder(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple texts into vectors."""

    def embed_query(self, query: str) -> list[float]:
        """Embed one query into a vector."""


class VectorStore(Protocol):
    def upsert(self, documents: Sequence[CodeDocument], embeddings: Sequence[Sequence[float]]) -> None:
        """Insert or update documents and vectors."""

    def search(self, query_embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        """Return nearest chunks for the query embedding."""

    def count(self) -> int:
        """Return the number of indexed documents."""


class DocumentSource(Protocol):
    def iter_documents(self, limit: int | None = None) -> Iterable[CodeDocument]:
        """Yield source documents for ingestion."""


class Generator(Protocol):
    def generate(self, query: str, chunks: Sequence[RetrievedChunk]) -> str:
        """Generate a grounded answer from retrieved chunks."""
