"""Vector-store adapters for tests and production."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag.domain.models import CodeDocument, RetrievedChunk


class InMemoryVectorStore:
    """A deterministic vector store used by unit tests and local experiments."""

    def __init__(self) -> None:
        self._records: dict[str, tuple[CodeDocument, list[float]]] = {}

    def upsert(self, documents: Sequence[CodeDocument], embeddings: Sequence[Sequence[float]]) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length")
        for document, embedding in zip(documents, embeddings):
            self._records[document.id] = (document, list(embedding))

    def search(self, query_embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        scored: list[tuple[float, CodeDocument]] = []
        for document, embedding in self._records.values():
            scored.append((_cosine_similarity(query_embedding, embedding), document))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(document=document, score=score, rank=rank)
            for rank, (score, document) in enumerate(scored[:top_k], start=1)
        ]

    def count(self) -> int:
        return len(self._records)


class ChromaCodeVectorStore:
    """Chroma persistent client adapter."""

    def __init__(self, path: Path, collection_name: str, *, distance_metric: str = "cosine") -> None:
        import chromadb

        self.path = path
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric},
        )

    def upsert(self, documents: Sequence[CodeDocument], embeddings: Sequence[Sequence[float]]) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length")
        if not documents:
            return
        self._collection.upsert(
            ids=[document.id for document in documents],
            embeddings=[list(embedding) for embedding in embeddings],
            documents=[document.text for document in documents],
            metadatas=[_clean_metadata(document.metadata) for document in documents],
        )

    def search(self, query_embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        raw = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = raw.get("ids", [[]])[0]
        texts = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        chunks: list[RetrievedChunk] = []
        for rank, doc_id in enumerate(ids, start=1):
            distance = float(distances[rank - 1]) if rank - 1 < len(distances) else 0.0
            score = _distance_to_score(distance, self.distance_metric)
            document = CodeDocument(
                id=str(doc_id),
                text=str(texts[rank - 1]) if rank - 1 < len(texts) else "",
                metadata=metadatas[rank - 1] if rank - 1 < len(metadatas) and metadatas[rank - 1] else {},
            )
            chunks.append(RetrievedChunk(document=document, score=score, rank=rank))
        return chunks

    def count(self) -> int:
        return int(self._collection.count())


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _distance_to_score(distance: float, metric: str) -> float:
    if metric == "cosine":
        return 1.0 - distance
    return -distance


def _clean_metadata(metadata: dict[str, Any] | Any) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in dict(metadata).items():
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif value is None:
            clean[key] = ""
        else:
            clean[key] = str(value)
    return clean
