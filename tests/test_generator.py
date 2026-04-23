from rag.domain.models import CodeDocument, RetrievedChunk
from rag.generation.prompt_builder import GroundedPromptBuilder
from rag.generation.response_cleaner import strip_reasoning_tags


def test_grounded_prompt_uses_assignment_citation_format() -> None:
    doc = CodeDocument(
        id="1",
        text="Documentation:\nDoes work\n\nCode:\ndef work(): pass",
        metadata={"repo": "owner/repo", "path": "pkg/mod.py", "func_name": "work"},
    )
    chunk = RetrievedChunk(document=doc, score=0.9, rank=1)

    messages = GroundedPromptBuilder().build_messages("How does this work?", [chunk])

    assert messages[0]["role"] == "system"
    assert "Cite sources as repo/path::func_name" in messages[0]["content"]
    assert "owner/repo/pkg/mod.py::work" in messages[1]["content"]


def test_strip_reasoning_tags_removes_qwen_thinking_block() -> None:
    answer = strip_reasoning_tags("<think>private reasoning</think>\nFinal answer with citation.")

    assert answer == "Final answer with citation."


def test_strip_reasoning_tags_handles_unclosed_qwen_thinking_block() -> None:
    answer = strip_reasoning_tags("<think>private reasoning that never closes")

    assert answer == ""
