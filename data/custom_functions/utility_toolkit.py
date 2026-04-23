from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from statistics import mean
from typing import Any


def normalize_snake_case_label(label: str) -> str:
    """Convert a messy human label into a stable snake case identifier."""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", label.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unnamed"


def rolling_window_average(values: Sequence[float], window_size: int) -> list[float]:
    """Compute rolling arithmetic averages for a numeric sequence."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if window_size > len(values):
        return []
    return [mean(values[i : i + window_size]) for i in range(len(values) - window_size + 1)]


def detect_json_schema_missing_keys(record: Mapping[str, Any], required_keys: Iterable[str]) -> list[str]:
    """Return required JSON-style keys that are missing from a mapping."""
    return [key for key in required_keys if key not in record]


def compact_chat_transcript(turns: Sequence[Mapping[str, str]], max_turns: int = 6) -> list[dict[str, str]]:
    """Keep the most recent chat turns while preserving role and content fields."""
    if max_turns <= 0:
        return []
    compacted: list[dict[str, str]] = []
    for turn in turns[-max_turns:]:
        compacted.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
    return compacted


def rank_retrieval_hits_by_confidence(hits: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rerank retrieval hits by similarity score and whether a citation is present."""

    def confidence(hit: Mapping[str, Any]) -> tuple[float, int]:
        score = float(hit.get("score", 0.0))
        has_citation = 1 if hit.get("citation") else 0
        return score, has_citation

    return [dict(hit) for hit in sorted(hits, key=confidence, reverse=True)]

