"""Factory helpers for script entrypoints."""

from __future__ import annotations

from rag.config import AppConfig
from rag.embeddings import SentenceTransformerEmbedder
from rag.generation.factory import build_generator
from rag.retrieval.service import CodeRetriever
from rag.vector_store import ChromaCodeVectorStore


def build_embedder(config: AppConfig) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        config.embedding.model_name,
        normalize_embeddings=config.embedding.normalize_embeddings,
    )


def build_vector_store(config: AppConfig) -> ChromaCodeVectorStore:
    return ChromaCodeVectorStore(
        path=config.vector_db.path,
        collection_name=config.vector_db.collection_name,
        distance_metric=config.vector_db.distance_metric,
    )


def build_retriever(config: AppConfig, embedder: SentenceTransformerEmbedder, vector_store: ChromaCodeVectorStore) -> CodeRetriever:
    return CodeRetriever(embedder=embedder, vector_store=vector_store, top_k=config.retrieval.top_k)


def build_pipeline_generator(config: AppConfig):
    return build_generator(config.generator)
