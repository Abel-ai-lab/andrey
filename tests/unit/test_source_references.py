"""Keep package provenance mentions inside docstring References sections."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_PACKAGE_NAME = re.compile(rb"causal[-_]?learn", re.IGNORECASE)
_HEADING = re.compile(rb"([ \t]*)(\S[^\r\n]*?)[ \t]*\r?\n")


def _reference_spans(source: bytes) -> list[tuple[int, int]]:
    """Locate References sections within module, class, and function docstrings."""
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    spans = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not ast.get_docstring(node, clean=False):
            continue
        doc = node.body[0].value
        start = offsets[doc.lineno - 1] + doc.col_offset
        end = offsets[doc.end_lineno - 1] + doc.end_col_offset
        doc_lines = source[start:end].splitlines(keepends=True)
        cursor = start
        reference_start = None
        reference_indent = None
        for index, line in enumerate(doc_lines[:-1]):
            heading = _HEADING.fullmatch(line)
            if heading and doc_lines[index + 1].strip() == b"-" * len(heading[2]):
                indent, title = heading.groups()
                if reference_start is not None and len(indent) <= reference_indent:
                    spans.append((reference_start, cursor))
                    reference_start = None
                if title == b"References":
                    reference_start = cursor + len(line) + len(doc_lines[index + 1])
                    reference_indent = len(indent)
            cursor += len(line)
        if reference_start is not None:
            spans.append((reference_start, end))
    return spans


def _violations(source: bytes, *, python: bool = True) -> list[int]:
    spans = _reference_spans(source) if python else []
    return [
        source.count(b"\n", 0, match.start()) + 1
        for match in _PACKAGE_NAME.finditer(source)
        if not any(start <= match.start() and match.end() <= end for start, end in spans)
    ]


def test_package_mentions_only_in_references():
    root = Path(__file__).resolve().parents[2]
    violations = []
    for path in sorted((root / "src").rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        source = path.read_bytes()
        if not _PACKAGE_NAME.search(source):
            continue
        violations.extend(
            f"{path.relative_to(root)}:{line}"
            for line in _violations(source, python=path.suffix in {".py", ".pyi"})
        )
    assert not violations, "Package provenance outside References:\n" + "\n".join(violations)


@pytest.mark.parametrize("name", ["causal-learn", "causallearn", "causal_learn", "Causal-Learn"])
@pytest.mark.parametrize("template", ['"""Uses {name}."""', "# Uses {name}.", 'name = "{name}"'])
def test_guard_rejects_mentions_outside_references(name, template):
    assert _violations(template.format(name=name).encode()) == [1]


@pytest.mark.parametrize("prefix", ["", "class Model:\n    ", "def fit():\n    "])
def test_guard_allows_docstring_references(prefix):
    indent = "    " if prefix else ""
    doc = "\n".join(
        [
            '"""Method.',
            "",
            "References",
            "----------",
            "causal-learn; causallearn; causal_learn.",
            '"""',
        ]
    )
    source = prefix + doc.replace("\n", "\n" + indent)
    assert _violations(source.encode()) == []


@pytest.mark.parametrize("section", ["Notes", "Examples", "Returns"])
def test_guard_stops_at_next_section(section):
    source = (
        '"""Method.\n\nReferences\n----------\ncausal-learn.\n\n'
        f'{section}\n{"-" * len(section)}\ncausallearn.\n"""'
    )
    assert _violations(source.encode()) == [9]


@pytest.mark.parametrize(
    "suffix",
    [" # causallearn", '\nname = "causallearn"', '\ndef fit():\n    """causallearn."""'],
)
def test_guard_stops_at_docstring_end(suffix):
    source = '"""Method.\n\nReferences\n----------\ncausal-learn.\n"""' + suffix
    assert len(_violations(source.encode())) == 1


@pytest.mark.parametrize("heading", ["References", "References\n---", "Examples\n--------"])
def test_guard_requires_references_heading(heading):
    source = f'"""Method.\n\n{heading}\ncausal-learn.\n"""'
    assert len(_violations(source.encode())) == 1


def test_guard_does_not_exempt_other_strings_or_files():
    source = b'notes = """Method.\n\nReferences\n----------\ncausal-learn.\n"""'
    assert _violations(source) == [5]
    assert _violations(source, python=False) == [5]
