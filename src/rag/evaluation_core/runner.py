"""End-to-end evaluation orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag.domain.models import Query, RAGResult
from rag.evaluation_core.grounding import GroundingHeuristic
from rag.evaluation_core.postprocessing import ensure_exact_citations, normalize_abstention
from rag.ports import Generator
from rag.retrieval.service import CodeRetriever


@dataclass
class EvaluationRunner:
    """Run retrieval, generation, and grounding checks for a set of queries."""

    retriever: CodeRetriever
    generator: Generator
    grounding: GroundingHeuristic

    def run(self, queries: Sequence[Query]) -> list[RAGResult]:
        results: list[RAGResult] = []
        for query in queries:
            chunks = tuple(self.retriever.retrieve(query.text))
            answer = self.generator.generate(query.text, chunks)
            answer = normalize_abstention(answer)
            answer = ensure_exact_citations(answer, chunks)
            results.append(
                RAGResult(
                    query=query,
                    retrieved=chunks,
                    answer=answer,
                    grounded=self.grounding.is_grounded(answer, chunks),
                )
            )
        return results
