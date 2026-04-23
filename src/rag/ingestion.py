"""Batch document ingestion into a vector store."""

from __future__ import annotations

from collections.abc import Iterable

from rag.domain.models import CodeDocument
from rag.ports import Embedder, VectorStore


class DocumentIngestor:
    """Coordinates embedding and vector-store insertion."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore, *, batch_size: int = 64) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.batch_size = batch_size

    def ingest(self, documents: Iterable[CodeDocument], *, limit: int | None = None) -> int:
        batch: list[CodeDocument] = []
        total = 0
        for document in documents:
            batch.append(document)
            if len(batch) >= self.batch_size:
                total += self._flush(batch)
                batch = []
            if limit is not None and total + len(batch) >= limit:
                break
        if batch:
            total += self._flush(batch)
        return total

    def _flush(self, batch: list[CodeDocument]) -> int:
        embeddings = self.embedder.embed_texts([document.text for document in batch])
        self.vector_store.upsert(batch, embeddings)
        return len(batch)
