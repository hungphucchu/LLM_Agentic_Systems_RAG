"""Domain models for Code RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class CodeDocument:
    """A source-code chunk that can be indexed and cited."""

    id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def func_name(self) -> str:
        return str(self.metadata.get("func_name") or "unknown_func")

    @property
    def source_type(self) -> str:
        return str(self.metadata.get("source") or "starter")

    def citation(self) -> str:
        repo = str(self.metadata.get("repo") or "unknown_repo").strip("/")
        path = str(self.metadata.get("path") or "unknown_path").strip("/")
        return f"{repo}/{path}::{self.func_name}"


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved document plus search metadata."""

    document: CodeDocument
    score: float
    rank: int

    def citation(self) -> str:
        return self.document.citation()


@dataclass(frozen=True)
class Query:
    """A natural-language question evaluated against the RAG pipeline."""

    id: str
    text: str


@dataclass(frozen=True)
class RAGResult:
    """One end-to-end retrieval and generation result."""

    query: Query
    retrieved: tuple[RetrievedChunk, ...]
    answer: str
    grounded: bool

    def source_mix(self) -> str:
        kinds = sorted({chunk.document.source_type for chunk in self.retrieved})
        return "+".join(kinds) if kinds else "none"
