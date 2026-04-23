from pathlib import Path

from rag.document_sources.local_python import LocalPythonFunctionSource
from rag.document_sources.python_function_parser import PythonFunctionParser


def test_python_function_parser_extracts_top_level_functions(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.py"
    file_path.write_text(
        '''
def alpha(value: str) -> str:
    """Uppercase a string."""
    return value.upper()


class Hidden:
    def method(self) -> None:
        pass
''',
        encoding="utf-8",
    )

    parsed = PythonFunctionParser().parse_file(file_path, root=tmp_path)

    assert [func.name for func in parsed] == ["alpha"]
    assert parsed[0].docstring == "Uppercase a string."
    assert "return value.upper()" in parsed[0].code


def test_local_python_function_source_builds_citable_documents(tmp_path: Path) -> None:
    directory = tmp_path / "custom_functions"
    directory.mkdir()
    file_path = directory / "tools.py"
    file_path.write_text(
        '''
def normalize_name(name: str) -> str:
    """Normalize a display name."""
    return name.strip().lower()
''',
        encoding="utf-8",
    )

    docs = list(LocalPythonFunctionSource(directory).iter_documents())

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "custom"
    assert docs[0].metadata["func_name"] == "normalize_name"
    assert docs[0].citation() == "local-custom/custom_functions/tools.py::normalize_name"
