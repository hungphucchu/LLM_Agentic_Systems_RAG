"""Query loading and result-table formatting helpers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag.domain.models import Query, RAGResult
from rag.io import load_json, write_csv, write_jsonl


def load_queries(path: Path, *, limit: int | None = None) -> list[Query]:
    """Load evaluation queries from JSON."""
    raw = load_json(path)
    queries = [Query(id=str(item["id"]), text=str(item["text"])) for item in raw]
    return queries[:limit] if limit is not None else queries


def write_result_tables(results: Sequence[RAGResult], *, jsonl_path: Path, csv_path: Path) -> None:
    """Write both JSONL and CSV versions of the result table."""
    rows = [result_to_record(result) for result in results]
    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)


def result_to_record(result: RAGResult) -> dict[str, Any]:
    """Convert one result to a flat record for assignment tables."""
    return {
        "query_id": result.query.id,
        "query_text": result.query.text,
        "top_k_sources": " | ".join(chunk.citation() for chunk in result.retrieved),
        "similarity_scores": " | ".join(f"{chunk.score:.4f}" for chunk in result.retrieved),
        "source_mix": result.source_mix(),
        "answer_first_two_sentences": first_two_sentences(result.answer),
        "grounded": "yes" if result.grounded else "no",
        "answer": result.answer,
    }


def first_two_sentences(text: str) -> str:
    """Return a short preview suitable for report tables."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    return " ".join(sentences[:2])
