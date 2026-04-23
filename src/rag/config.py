"""Typed configuration for the Code RAG pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rag.io import load_yaml
from rag.project_paths import resolve_repo_path


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    config: str
    split: str
    starter_size: int


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    batch_size: int
    normalize_embeddings: bool


@dataclass(frozen=True)
class VectorDbConfig:
    path: Path
    collection_name: str
    distance_metric: str


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int


@dataclass(frozen=True)
class GeneratorConfig:
    model_name: str
    base_url: str
    temperature: float
    max_tokens: int
    timeout_seconds: float


@dataclass(frozen=True)
class PathConfig:
    custom_functions_dir: Path
    part1_queries: Path
    part2_targeted_queries: Path
    part2_cross_queries: Path
    results_dir: Path


@dataclass(frozen=True)
class AppConfig:
    dataset: DatasetConfig
    embedding: EmbeddingConfig
    vector_db: VectorDbConfig
    retrieval: RetrievalConfig
    generator: GeneratorConfig
    paths: PathConfig

    @classmethod
    def from_yaml(cls, path: str | Path = "config/config.yaml") -> "AppConfig":
        load_dotenv(resolve_repo_path(".env"))
        data = load_yaml(resolve_repo_path(path))
        return cls(
            dataset=_dataset_config(data.get("dataset", {})),
            embedding=_embedding_config(data.get("embedding", {})),
            vector_db=_vector_db_config(data.get("vector_db", {})),
            retrieval=_retrieval_config(data.get("retrieval", {})),
            generator=_generator_config(data.get("generator", {})),
            paths=_path_config(data.get("paths", {})),
        )


def _dataset_config(data: dict[str, Any]) -> DatasetConfig:
    return DatasetConfig(
        name=str(data.get("name", "code_search_net")),
        config=str(data.get("config", "python")),
        split=str(data.get("split", "train")),
        starter_size=int(os.getenv("STARTER_SIZE", data.get("starter_size", 1000))),
    )


def _embedding_config(data: dict[str, Any]) -> EmbeddingConfig:
    return EmbeddingConfig(
        model_name=os.getenv("EMBEDDING_MODEL", str(data.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))),
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", data.get("batch_size", 64))),
        normalize_embeddings=bool(data.get("normalize_embeddings", True)),
    )


def _vector_db_config(data: dict[str, Any]) -> VectorDbConfig:
    return VectorDbConfig(
        path=resolve_repo_path(os.getenv("CHROMA_PATH", str(data.get("path", "artifacts/chroma_code")))),
        collection_name=os.getenv("CHROMA_COLLECTION", str(data.get("collection_name", "csn_python"))),
        distance_metric=str(data.get("distance_metric", "cosine")),
    )


def _retrieval_config(data: dict[str, Any]) -> RetrievalConfig:
    return RetrievalConfig(top_k=int(os.getenv("TOP_K", data.get("top_k", 4))))


def _generator_config(data: dict[str, Any]) -> GeneratorConfig:
    return GeneratorConfig(
        model_name=os.getenv("GENERATOR_MODEL") or os.getenv("UTSA_MODEL", str(data.get("model_name", "Qwen/Qwen3-8B"))),
        base_url=os.getenv("BASE_URL") or os.getenv("UTSA_BASE_URL", str(data.get("base_url", "http://149.165.171.140:8888/v1"))),
        temperature=float(os.getenv("GENERATOR_TEMPERATURE", data.get("temperature", 0.0))),
        max_tokens=int(os.getenv("GENERATOR_MAX_TOKENS", data.get("max_tokens", 512))),
        timeout_seconds=float(os.getenv("GENERATOR_TIMEOUT_SECONDS", data.get("timeout_seconds", 30))),
    )


def _path_config(data: dict[str, Any]) -> PathConfig:
    return PathConfig(
        custom_functions_dir=resolve_repo_path(data.get("custom_functions_dir", "data/custom_functions")),
        part1_queries=resolve_repo_path(data.get("part1_queries", "data/queries/part1_queries.json")),
        part2_targeted_queries=resolve_repo_path(data.get("part2_targeted_queries", "data/queries/part2_targeted_queries.json")),
        part2_cross_queries=resolve_repo_path(data.get("part2_cross_queries", "data/queries/part2_cross_corpus_queries.json")),
        results_dir=resolve_repo_path(data.get("results_dir", "artifacts/results")),
    )
