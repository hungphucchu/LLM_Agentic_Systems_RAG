"""Local custom-function source for Part 2."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from rag.document_sources.python_function_parser import PythonFunctionParser
from rag.domain.models import CodeDocument


class LocalPythonFunctionSource:
    """Yield citable documents from local custom Python functions."""

    def __init__(self, directory: Path, *, parser: PythonFunctionParser | None = None) -> None:
        self.directory = directory
        self.parser = parser or PythonFunctionParser()

    def iter_documents(self, limit: int | None = None) -> Iterable[CodeDocument]:
        emitted = 0
        for path in sorted(self.directory.glob("*.py")):
            for parsed in self.parser.parse_file(path, root=self.directory.parent):
                yield parsed.to_document()
                emitted += 1
                if limit is not None and emitted >= limit:
                    return
