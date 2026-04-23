"""Streaming starter-corpus source from CodeSearchNet."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rag.domain.models import CodeDocument
from datasets import load_dataset


class CodeSearchNetSource:
    """Streaming Hugging Face CodeSearchNet source for Python functions."""

    def __init__(self, dataset_name: str, config_name: str, split: str) -> None:
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split = split

    def iter_documents(self, limit: int | None = None) -> Iterable[CodeDocument]:

        dataset = load_dataset(
            self.dataset_name,
            self.config_name,
            split=self.split,
            streaming=True,
        )

        emitted = 0
        for row in dataset:
            document = _code_search_net_row_to_document(row, emitted)
            if document is None:
                continue
            yield document
            emitted += 1
            if limit is not None and emitted >= limit:
                break


def _code_search_net_row_to_document(row: dict[str, Any], index: int) -> CodeDocument | None:
    code = str(row.get("func_code_string") or "").strip()
    if not code:
        return None
    docstring = str(row.get("func_documentation_string") or "").strip()
    text = f"Documentation:\n{docstring}\n\nCode:\n{code}"
    func_name = str(row.get("func_name") or f"function_{index}")
    repo = str(row.get("repository_name") or "unknown_repo")
    path = str(row.get("func_path_in_repository") or row.get("path") or "unknown_path")
    return CodeDocument(
        id=f"csn_python_{index:06d}",
        text=text,
        metadata={
            "repo": repo,
            "path": path,
            "func_name": func_name,
            "source": "starter",
        },
    )
