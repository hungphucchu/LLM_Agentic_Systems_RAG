"""AST parsing for local custom Python functions."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

from rag.domain.models import CodeDocument


@dataclass(frozen=True)
class ParsedFunction:
    """A parsed top-level function ready to be turned into a document."""

    name: str
    code: str
    docstring: str
    path: Path
    relative_path: str

    def to_document(self) -> CodeDocument:
        digest = hashlib.sha1(f"{self.relative_path}:{self.name}".encode("utf-8")).hexdigest()[:12]
        text = f"Documentation:\n{self.docstring}\n\nCode:\n{self.code}"
        return CodeDocument(
            id=f"custom_{digest}",
            text=text,
            metadata={
                "repo": "local-custom",
                "path": self.relative_path,
                "func_name": self.name,
                "source": "custom",
            },
        )


class PythonFunctionParser:
    """Parse top-level functions from Python files."""

    def parse_file(self, path: Path, *, root: Path | None = None) -> list[ParsedFunction]:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        relative_path = _relative_path(path, root)
        functions: list[ParsedFunction] = []
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            code = ast.get_source_segment(source, node) or ""
            functions.append(
                ParsedFunction(
                    name=node.name,
                    code=code,
                    docstring=ast.get_docstring(node) or "",
                    path=path,
                    relative_path=relative_path,
                )
            )
        return functions


def _relative_path(path: Path, root: Path | None) -> str:
    if root is None:
        return path.name
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name
