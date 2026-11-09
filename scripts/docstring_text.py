"""Write the shared sentences of the method docstrings from one table, or check them.

Some sentences of the method docstrings in ``src/andrey/api`` state one rule for many methods: the
openers of the Raises entries, the seed, knowledge, and env entries, the closing sentences of
**Missing values.** and **Limitations.**, the ``tie_break`` record, and the Warns entries of the
shared data checks. ``SHARED`` holds each once, with its place and the methods that hold it. A
place is the first or last sentence of a docstring entry or of a labeled Notes paragraph, or a
whole entry. Writing replaces each copy with the table's text and rewraps its paragraph at 100
columns. A copy is found by its place and its first three words, so a method that should hold a
sentence and has no copy is named, for the first copy to be written in by hand.

    uv run python scripts/docstring_text.py
    uv run python scripts/docstring_text.py --check

``tests/unit/test_facade_docstrings.py`` fails while a copy differs from the table.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from andrey.api._adapt import MAX_CATEGORIES
from andrey.api._boundary import CAP_STOPS, measured
from andrey.spec import REGISTRY
from andrey.spec.concepts import CONCEPTS

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "src" / "andrey" / "api"
WIDTH = 100
#: The end of a sentence: a period, then the capital, literal, or role that opens the next one.
SENTENCE_END = re.compile(r"\.\s+(?=[A-Z`:])")
#: The arguments a score search records in ``metadata`` as the penalties its score used.
PENALTIES = ("penalty_discount", "ebic_gamma", "equivalent_sample_size")
#: The score searches followed by FCI's tests and rules, which refuse missing values.
FCI_SEARCHES = ("gfci", "boss_fci", "grasp_fci")


class Shared(NamedTuple):
    """One shared sentence or entry.

    ``section`` and ``anchor`` locate the copy: an entry's name (``""`` for the Returns entry) or
    a Notes paragraph's bold label. ``place`` is ``"first"`` or ``"last"``, the first or last
    sentence of that entry's description or paragraph, or ``"entry"``, the whole description.
    ``holds`` tells the methods whose docstrings hold the copy.
    """

    section: str
    anchor: str
    place: str
    text: str
    holds: Callable[[str], bool]


def takes(method: str, name: str) -> bool:
    """Whether ``method`` has the parameter ``name``."""
    return any(p.name == name for p in REGISTRY[method].params)


def can_pass_boundary(method: str) -> bool:
    """Whether a fit of ``method`` can warn past its measured boundary: a ladder ended on a cap."""
    return any(ladder.stop in CAP_STOPS for ladder in measured().get(method, ()))


def records_penalties(method: str) -> bool:
    """Whether a fit of ``method`` records the penalties its score used."""
    return any(key in REGISTRY[method].output.metadata for key in PENALTIES)


def refuses_missing(method: str, *, series: bool) -> bool:
    """Whether ``method`` refuses missing values and reads time series (``series``) or rows."""
    spec = REGISTRY[method]
    return not spec.data.missing and (spec.family == "temporal") == series


SHARED = {
    "argument_type": Shared(
        "Raises",
        "TypeError",
        "first",
        "If an argument has the wrong type, such as a string for a number; the message names the "
        "method, the argument, what it accepts, and the value.",
        lambda m: True,
    ),
    "argument_value": Shared(
        "Raises",
        "ValueError",
        "first",
        "If an argument lies outside its range or is not one of its choices; the message names the "
        "method, the argument, what it accepts, and the value.",
        lambda m: True,
    ),
    "seed": Shared(
        "Parameters",
        "seed",
        "first",
        "``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset, "
        "not a fresh random one, so a fit repeats.",
        lambda m: REGISTRY[m].seeded,
    ),
    # The time-series methods read knowledge over lags, and gnies honors it in each of its
    # searches over graphs.
    "knowledge": Shared(
        "Parameters",
        "knowledge",
        "first",
        "Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the "
        "pairs allowed to be adjacent, and temporal tiers, all honored.",
        lambda m: takes(m, "knowledge") and REGISTRY[m].family != "temporal" and m != "gnies",
    ),
    # cdnod and fci_jci read env as a variable, coerced to numbers.
    "env": Shared(
        "Parameters",
        "env",
        "first",
        "The environment of each row: one label per row, strings or finite numbers, not both, or "
        "the name of a column of the DataFrame ``data`` that holds them.",
        lambda m: (
            any(i.name == "env" for i in REGISTRY[m].data.inputs) and m not in ("cdnod", "fci_jci")
        ),
    ),
    # A method whose missing takes more than the template's values words its own entry.
    "missing": Shared(
        "Parameters",
        "missing",
        "entry",
        'How missing values (``NaN``) are handled. ``"testwise"`` deletes rows per test: each test '
        "reads only the rows where all of its variables are observed. ``None`` picks the method's "
        'default, ``"testwise"``. On data with missing values, ``metadata["missing"]`` records the '
        "handling used. See **Missing values.** in Notes.",
        lambda m: m in CONCEPTS["missing"].conforming,
    ),
    # icp, igsp, and ut_igsp end on "Drop the incomplete rows first."
    "drop_rows": Shared(
        "Notes",
        "**Missing values.**",
        "last",
        "Drop the incomplete rows first, which suits values missing completely at random, or use "
        ":func:`~andrey.pc` or :func:`~andrey.fci`, whose tests delete rows per test.",
        lambda m: (
            refuses_missing(m, series=False) and m not in (*FCI_SEARCHES, "icp", "igsp", "ut_igsp")
        ),
    ),
    "drop_rows_fci": Shared(
        "Notes",
        "**Missing values.**",
        "last",
        "Drop the incomplete rows first, which suits values missing completely at random, or use "
        ":func:`~andrey.fci`, whose tests delete rows per test.",
        lambda m: m in FCI_SEARCHES,
    ),
    # longitudinal_lingam reads units at each time point, not one series.
    "fill_steps": Shared(
        "Notes",
        "**Missing values.**",
        "last",
        "Fill in the missing steps first, as dropping one would join the steps around it, or use "
        ":func:`~andrey.pcmci`, :func:`~andrey.pcmciplus`, or :func:`~andrey.lpcmci`, whose tests "
        "delete rows per test.",
        lambda m: refuses_missing(m, series=True) and m != "longitudinal_lingam",
    ),
    "limits": Shared(
        "Notes",
        "**Limitations.**",
        "last",
        "The numbers are a reference, not a hard limit: they change with the data generator, the "
        "density, the method's settings, and the machine.",
        lambda m: True,
    ),
    "tie_break": Shared(
        "Returns",
        "",
        "last",
        '``metadata["tie_break"]`` is the tie-break rule the fit ran under, '
        "``andrey.config.tie_break``.",
        lambda m: REGISTRY[m].tie_break and not records_penalties(m),
    ),
    "tie_break_penalties": Shared(
        "Returns",
        "",
        "last",
        '``metadata["tie_break"]`` is the tie-break rule the fit ran under, '
        "``andrey.config.tie_break``, and ``metadata`` also records the penalties its score used, "
        "under their parameter names.",
        lambda m: REGISTRY[m].tie_break and records_penalties(m),
    ),
    "boundary": Shared(
        "Warns",
        "BoundaryWarning",
        "entry",
        "When the data are larger than the largest size whose fits finished within the time and "
        "memory caps of the published benchmark, at the measured setting nearest the data's rows, "
        "and fits of the next size ran past a cap. It warns once per process and names that size "
        "and the release card; the size is a reference, not a hard limit.",
        can_pass_boundary,
    ),
    # A method that takes ``categorical`` reads declared columns as categories and words its own
    # entry.
    "category_codes": Shared(
        "Warns",
        "DataTypeWarning",
        "entry",
        f"When a column holds at most {MAX_CATEGORIES} distinct whole numbers and so may hold "
        "category codes, which the fit reads as numbers. Each message warns once per process.",
        lambda m: not takes(m, "categorical"),
    ),
}


def fill(text: str, indent: str) -> list[str]:
    """Wrap ``text`` at ``WIDTH`` columns after ``indent``.

    A backtick span breaks only when it is longer than a line.
    """
    width = WIDTH - len(indent)

    def keep(match: re.Match) -> str:
        span = match.group()
        return span.replace(" ", "\0") if len(span) <= width else span

    joined = re.sub(r"``.+?``|`[^`]+`", keep, text)
    lines = textwrap.wrap(joined, width, break_long_words=False, break_on_hyphens=False)
    return [indent + line.replace("\0", " ") for line in lines]


def sections(lines: list[str]) -> dict[str, tuple[int, int]]:
    """Return the body of each numpydoc section of ``lines`` as a ``(start, stop)`` range."""
    heads = [
        i
        for i in range(len(lines) - 1)
        if lines[i].strip() and re.fullmatch(r"-+", lines[i + 1].strip())
    ]
    stops = [*heads[1:], len(lines) - 1]
    return {lines[i].strip(): (i + 2, stop) for i, stop in zip(heads, stops, strict=True)}


def unit(lines: list[str], shared: Shared) -> tuple[int, int] | None:
    """Return the line range of the paragraph that holds ``shared``'s copy, or ``None``."""
    found = sections(lines).get(shared.section)
    if found is None:
        return None
    start, stop = found
    if shared.section == "Notes":
        first = next(
            (i for i in range(start, stop) if lines[i].strip().startswith(shared.anchor)), None
        )
        if first is None:
            return None
        last = next((i for i in range(first, stop) if not lines[i].strip()), stop)
        return first, last
    heads = [i for i in range(start, stop) if re.match(r" {4}\S", lines[i])]
    for k, head in enumerate(heads):
        if shared.anchor and lines[head].strip().split(" :")[0] != shared.anchor:
            continue
        end = heads[k + 1] if k + 1 < len(heads) else stop
        blocks, block = [], []
        for i in range(head + 1, end):
            if lines[i].strip():
                block.append(i)
            elif block:
                blocks.append(block)
                block = []
        if block:
            blocks.append(block)
        if not blocks:
            return None
        chosen = blocks[-1] if shared.place == "last" else blocks[0]
        return chosen[0], chosen[-1] + 1
    return None


def rewrite(text: str, shared: Shared) -> str | None:
    """Return ``text`` with its copy of ``shared`` replaced, or ``None`` when it holds none."""
    if shared.place == "entry":
        return shared.text
    if shared.place == "first":
        end = SENTENCE_END.search(text)
        old = text[: end.start() + 1] if end else text
        head, tail = "", text[len(old) :]
    else:
        ends = list(SENTENCE_END.finditer(text))
        cut = ends[-1].end() if ends else 0
        head, old, tail = text[:cut], text[cut:], ""
    if old.split()[:3] != shared.text.split()[:3]:
        return None
    return head + shared.text + tail


def write_docstring(lines: list[str], method: str) -> list[str]:
    """Replace each shared copy ``method``'s docstring ``lines`` hold.

    Returns the keys of the copies it should hold and does not.
    """
    missing = []
    for key, shared in SHARED.items():
        if not shared.holds(method):
            continue
        found = unit(lines, shared)
        if found is None:
            missing.append(key)
            continue
        start, stop = found
        text = " ".join(line.strip() for line in lines[start:stop])
        new = rewrite(text, shared)
        if new is None:
            missing.append(key)
        elif new != text:
            indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
            lines[start:stop] = fill(new, indent)
    return missing


def facades(source: str) -> list[tuple[str, int, int]]:
    """Return each ``@method_spec`` facade of ``source``: its method and docstring lines."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef) or not ast.get_docstring(node):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and getattr(decorator.func, "id", None) == "method_spec"
                and isinstance(decorator.args[0], ast.Constant)
            ):
                doc = node.body[0]
                found.append((str(decorator.args[0].value), doc.lineno - 1, doc.end_lineno or 0))
    return found


def write_source(source: str) -> tuple[str, dict[str, list[str]]]:
    """Return ``source`` with its facades' shared copies written.

    Also returns, per facade, the keys of the copies it should hold and does not.
    """
    lines = source.split("\n")
    missing = {}
    for method, start, stop in sorted(facades(source), key=lambda f: -f[1]):
        doc = lines[start:stop]
        absent = write_docstring(doc, method)
        lines[start:stop] = doc
        if absent:
            missing[method] = absent
    return "\n".join(lines), missing


def run(*, write: bool) -> list[str]:
    """Write the shared copies, or with ``write=False`` only compare them.

    Returns the problems: a file with a copy that differs from the table, or a missing copy.
    """
    problems, seen = [], set()
    for path in sorted(API.glob("*.py")):
        source = path.read_text()
        updated, missing = write_source(source)
        seen.update(method for method, _, _ in facades(source))
        problems += [f"{method}: no copy of {', '.join(keys)}" for method, keys in missing.items()]
        if updated != source:
            if write:
                path.write_text(updated)
            else:
                problems.append(f"{path.relative_to(ROOT)}: a copy differs from the table")
    problems += [f"{method}: no facade" for method in sorted(set(REGISTRY) - seen)]
    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="change nothing; fail on a drift")
    found = run(write=not parser.parse_args().check)
    if found:
        sys.exit("\n".join(found))
