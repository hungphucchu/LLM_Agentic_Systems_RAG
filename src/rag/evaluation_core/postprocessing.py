"""Answer post-processing helpers for evaluation output."""

from __future__ import annotations

from collections.abc import Sequence

from rag.domain.models import RetrievedChunk


def ensure_exact_citations(answer: str, chunks: Sequence[RetrievedChunk], *, max_sources: int = 2) -> str:
    """Append exact citations when the answer is otherwise supported but underspecified."""
    if not chunks or "do not know based on the retrieved code" in answer.lower():
        return answer
    exact_citations = [chunk.citation() for chunk in chunks]
    if any(citation in answer for citation in exact_citations):
        return answer

    selected = _select_relevant_citations(answer, chunks, max_sources=max_sources)
    if not selected:
        selected = exact_citations[:max_sources]
    return f"{answer.rstrip()} Sources: {'; '.join(selected)}."


def normalize_abstention(answer: str) -> str:
    """Collapse noisy abstentions into the assignment-required response."""
    if "do not know based on the retrieved code" in answer.lower():
        return "I do not know based on the retrieved code."
    return answer


def _select_relevant_citations(answer: str, chunks: Sequence[RetrievedChunk], *, max_sources: int) -> list[str]:
    lowered = answer.lower()
    selected: list[str] = []
    for chunk in chunks:
        func_name = chunk.document.func_name.lower()
        short_func = func_name.split(".")[-1]
        path_name = str(chunk.document.metadata.get("path", "")).lower()
        if func_name in lowered or short_func in lowered or path_name in lowered:
            selected.append(chunk.citation())
        if len(selected) >= max_sources:
            break
    return selected
