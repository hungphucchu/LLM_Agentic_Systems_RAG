"""Prompt construction for grounded code answers."""

from __future__ import annotations

from collections.abc import Sequence

from rag.domain.models import RetrievedChunk


class GroundedPromptBuilder:
    """Build the assignment-required grounded prompt."""

    system_message = (
        "Answer using only the provided code. "
        "Do not use outside Python knowledge, even if you know the general answer. "
        "If the provided code does not contain the answer, reply exactly: "
        "\"I do not know based on the retrieved code.\" "
        "Write 2 to 4 concise prose sentences. Do not use bullet lists, numbered lists, or code blocks. "
        "Cite sources as repo/path::func_name. End every supported answer with a final sentence "
        "starting with \"Sources:\" followed by exact citations copied from the context. "
        "Return only the final answer; do not include chain-of-thought, hidden reasoning, or <think> tags. "
        "/no_think"
    )

    def build_messages(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[dict[str, str]]:
        context = self.build_context(chunks)
        user_message = f"Context:\n{context}\n\nUser query: {query}"
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]

    def build_context(self, chunks: Sequence[RetrievedChunk]) -> str:
        if not chunks:
            return "[no retrieved code]"
        blocks: list[str] = []
        for chunk in chunks:
            blocks.append(
                "\n".join(
                    [
                        f"[{chunk.rank}] repo/path::func: {chunk.citation()}",
                        f"score: {chunk.score:.4f}",
                        chunk.document.text,
                    ]
                )
            )
        return "\n\n".join(blocks)
