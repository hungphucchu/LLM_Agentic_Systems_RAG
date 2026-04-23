from collections.abc import Sequence

from rag.domain.models import CodeDocument, Query, RetrievedChunk
from rag.evaluation_core.grounding import GroundingHeuristic
from rag.evaluation_core.postprocessing import ensure_exact_citations, normalize_abstention
from rag.evaluation_core.result_tables import first_two_sentences, result_to_record
from rag.evaluation_core.runner import EvaluationRunner


class StubRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return self.chunks


class StubGenerator:
    def generate(self, query: str, chunks: Sequence[RetrievedChunk]) -> str:
        return f"Use this code. Cite: {chunks[0].citation()}."


def test_first_two_sentences_limits_answer_preview() -> None:
    assert first_two_sentences("One. Two. Three.") == "One. Two."


def test_evaluation_runner_writes_assignment_fields() -> None:
    doc = CodeDocument(
        id="1",
        text="Documentation:\nValidate fields",
        metadata={"repo": "repo", "path": "schema.py", "func_name": "validate_fields", "source": "custom"},
    )
    chunk = RetrievedChunk(document=doc, score=0.7, rank=1)
    runner = EvaluationRunner(
        retriever=StubRetriever([chunk]),  # type: ignore[arg-type]
        generator=StubGenerator(),
        grounding=GroundingHeuristic(),
    )

    result = runner.run([Query(id="q1", text="How do I validate fields?")])[0]
    record = result_to_record(result)

    assert record["query_id"] == "q1"
    assert record["top_k_sources"] == "repo/schema.py::validate_fields"
    assert record["similarity_scores"] == "0.7000"
    assert record["grounded"] == "yes"
    assert record["source_mix"] == "custom"


def test_grounding_heuristic_accepts_honest_do_not_know() -> None:
    heuristic = GroundingHeuristic()

    assert heuristic.is_grounded("I do not know based on the retrieved code.", []) is True


def test_grounding_heuristic_rejects_do_not_know_with_external_advice() -> None:
    heuristic = GroundingHeuristic()

    answer = "I do not know. For example: use Python's statistics module."

    assert heuristic.is_grounded(answer, []) is False


def test_ensure_exact_citations_appends_required_source_format() -> None:
    doc = CodeDocument(
        id="1",
        text="Documentation:\nValidate fields",
        metadata={"repo": "repo", "path": "schema.py", "func_name": "validate_fields", "source": "custom"},
    )
    chunk = RetrievedChunk(document=doc, score=0.7, rank=1)

    answer = ensure_exact_citations("Use validate_fields to check input.", [chunk])

    assert answer.endswith("Sources: repo/schema.py::validate_fields.")


def test_normalize_abstention_removes_extra_generated_text() -> None:
    answer = normalize_abstention("I do not know based on the retrieved code. Sources: irrelevant/source.py::func")

    assert answer == "I do not know based on the retrieved code."
