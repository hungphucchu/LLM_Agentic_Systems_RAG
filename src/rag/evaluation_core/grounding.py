"""Transparent grounding heuristics for result tables."""

from __future__ import annotations

from collections.abc import Sequence

from rag.domain.models import RetrievedChunk


class GroundingHeuristic:
    """Apply a transparent binary grounding check for result tables."""

    def is_grounded(self, answer: str, chunks: Sequence[RetrievedChunk]) -> bool:
        normalized_answer = answer.lower()
        if "do not know based on the retrieved code" in normalized_answer:
            return True
        if ("do not know" in normalized_answer or "don't know" in normalized_answer) and not _looks_like_external_advice(answer):
            return True
        if not chunks:
            return False
        citations = [chunk.citation() for chunk in chunks]
        if any(citation in answer for citation in citations):
            return True
        func_names = [chunk.document.func_name for chunk in chunks]
        return any(func_name in answer for func_name in func_names) and "do not know" not in normalized_answer


def _looks_like_external_advice(answer: str) -> bool:
    lower = answer.lower()
    red_flags = [
        "for example:",
        "typically",
        "you would",
        "use python's",
        "use built-in",
        "libraries like",
        "this is unrelated",
        "```",
    ]
    return any(flag in lower for flag in red_flags)
