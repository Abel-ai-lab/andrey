"""Check that no published benchmark row has Andrey making more errors than its reference package.

Reads only the published summaries (``benchmarks/published/*/summary.json``), so it runs without
the reference packages. A row pairs Andrey with one reference package on one dataset size. It fails
when the reference has fewer errors on more series than Andrey; where the row records only its
central errors (the mean or median over the series), it fails when Andrey's are higher. Any
margin fails: a consistent shortfall, however small, is a shortfall.

A reference whose result depends on the column order (a row's ``reference_order_dependent``) is
compared with Andrey in its column-order mode, run on the same series: the row's
``andrey_column_shd_by_seed`` and ``comparator_shd_by_seed`` hold one error count per series. The
row is judged by those series, and Andrey's order-free default is printed beside them, unjudged. A
row of an order-dependent reference without that arm is judged by the default.

``parity-allowlist.json``, next to the summaries, exempts a failing row only when it names the
cause (``CAUSES``), gives the reason and a citation, and pins the row's two error values, so a
re-run that changes the row voids the entry:

- ``correct-by-paper``: Andrey follows the method's paper and the reference does not.
- ``reference-defect``: the reference has a defect.
- ``tie-break``: the two packages end at different local optima after an exact score tie that
  Andrey breaks by column order.
- ``order-dependent-reference``: a choice the reference makes in column order, such as which of two
  conflicting colliders it keeps or the order of its possible-d-separation step, decides the row.
- ``random-draw``: the random draws of the CI test decide the row.
- ``experimental-test``: the row measures an experimental CI test, published as measured.

A ``tie-break`` or ``order-dependent-reference`` entry gives, as its evidence, the row's two error
values with the columns in reversed order, where the row must pass; a ``random-draw`` entry gives
the row's values under at least two other seeds of the draws, where the row must pass on each. An
entry that matches no failing row also fails. A released summary never changes, so a method
re-measured against the same reference in a later summary is judged by the later rows only. Where a
summary has two entries of one method against one reference, one per regime, a row's key also names
its ``regime``.

    uv run python -m qa.reference_parity              # exit 1 if any row fails
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PUBLISHED = Path(__file__).resolve().parents[1] / "benchmarks" / "published"
ALLOWLIST = PUBLISHED / "parity-allowlist.json"

#: Per-series counts as ``(series where Andrey has fewer errors, series where the reference has
#: fewer)``. When a row has one of these pairs, it decides the row alone.
SERIES_COUNTS: tuple[tuple[str, str], ...] = (
    ("andrey_fewer_errors", "comparator_fewer_errors"),
    ("andrey_right_comparator_wrong", "comparator_right_andrey_wrong"),
)
#: Per-series errors of Andrey's column-order mode and of the reference, on the same series.
COLUMN_SERIES: tuple[str, str] = ("andrey_column_shd_by_seed", "comparator_shd_by_seed")
#: Error fields recorded as ``andrey_<field>`` and ``comparator_<field>``: the row's mean or median.
ERRORS: tuple[str, ...] = ("shd", "directed_errors")
#: Row fields that name a row within its method's table.
ROW_KEYS: tuple[str, ...] = ("mean_degree", "d", "observed", "n")
#: The causes an allowlist entry may name.
CAUSES: frozenset[str] = frozenset(
    {
        "correct-by-paper",
        "reference-defect",
        "tie-break",
        "order-dependent-reference",
        "random-draw",
        "experimental-test",
    }
)
#: Causes whose evidence is the row with the columns in reversed order.
REVERSED_ORDER_CAUSES: frozenset[str] = frozenset({"tie-break", "order-dependent-reference"})


@dataclass(frozen=True)
class Row:
    """One published row that pairs Andrey with a reference package.

    ``andrey`` and ``reference`` are each side's errors: the row's mean or median for an error
    field, or, under a series rule, the number of series where that side has more errors. Under the
    ``column-order series`` rule they compare Andrey's column-order mode with the reference, and
    ``default`` holds the order-free default's two counts, which are printed but not judged.
    """

    summary: str
    method: str
    comparator: str
    key: dict[str, Any]
    rule: str
    andrey: float
    reference: float
    default: tuple[float, float] | None = None

    @property
    def failed(self) -> bool:
        return self.andrey > self.reference

    def label(self) -> str:
        where = " ".join(f"{k}={v}" for k, v in self.key.items())
        return f"{self.method} vs {self.comparator}, {where} ({self.summary})"

    def numbers(self) -> str:
        if self.rule in ("series", "column-order series"):
            text = (
                f"{self.comparator} has fewer errors on {self.andrey:g} series, "
                f"Andrey on {self.reference:g}"
            )
            if self.default is None:
                return text
            return (
                f"against Andrey's column-order mode, {text}; order-free default (not judged): "
                f"{self.comparator} on {self.default[0]:g}, Andrey on {self.default[1]:g}"
            )
        return f"{self.rule} Andrey {self.andrey:g}, {self.comparator} {self.reference:g}"


def paired_rows(summary: str, data: dict) -> list[Row]:
    """Return the rows of one summary that pair Andrey with a reference package.

    A row where either side has no result (a fit over the time cap) is skipped, and so is a series
    of the column-order comparison. A paired row with no known error field, a column-order arm on a
    reference not marked order-dependent, and an arm with one of its two series missing or of
    unequal lengths raise, so a new summary format cannot pass unchecked.
    """
    entries = Counter((m["method"], m.get("comparator")) for m in data["methods"])
    rows = []
    for method in data["methods"]:
        comparator = method.get("comparator")
        if not comparator:
            continue
        by_regime = entries[(method["method"], comparator)] > 1
        if by_regime and not method.get("regime"):
            raise ValueError(
                f"{summary}: {method['method']} vs {comparator} has two entries and no regime"
            )
        for row in method.get("rows", []):
            key = {"regime": method["regime"]} if by_regime else {}
            key.update({k: row[k] for k in ROW_KEYS if k in row})
            counts = next(((a, c) for a, c in SERIES_COUNTS if a in row and c in row), None)
            if counts:
                rule, andrey, reference = "series", row[counts[1]], row[counts[0]]
            else:
                field = next((f for f in ERRORS if f"andrey_{f}" in row), None)
                if field is None or f"comparator_{field}" not in row:
                    raise ValueError(
                        f"{summary}: {method['method']} vs {comparator} {key} has no error field "
                        f"from {ERRORS} or {SERIES_COUNTS}"
                    )
                rule, andrey, reference = field, row[f"andrey_{field}"], row[f"comparator_{field}"]
            if andrey is None or reference is None:
                continue
            default = None
            column = _column_series(row, f"{summary}: {method['method']} vs {comparator} {key}")
            if column is not None:
                default = (andrey, reference)
                rule, andrey, reference = "column-order series", *column
            rows.append(
                Row(summary, method["method"], comparator, key, rule, andrey, reference, default)
            )
    return rows


def _column_series(row: dict, where: str) -> tuple[int, int] | None:
    """Return the column-order comparison's ``(reference fewer, Andrey fewer)`` series counts.

    ``None`` when the row has no column-order arm. Only a row whose reference depends on the
    column order may carry one, and the arm needs both series, of equal length.
    """
    present = [name for name in COLUMN_SERIES if name in row]
    if not present:
        return None
    if row.get("reference_order_dependent") is not True:
        raise ValueError(f"{where} has a column-order arm, but its reference is order-free")
    if len(present) != len(COLUMN_SERIES):
        raise ValueError(f"{where} has {present[0]} without the other of {COLUMN_SERIES}")
    mine, theirs = (row[name] for name in COLUMN_SERIES)
    if len(mine) != len(theirs):
        raise ValueError(f"{where}: {COLUMN_SERIES} differ in length")
    pairs = [(a, c) for a, c in zip(mine, theirs, strict=True) if a is not None and c is not None]
    return sum(c < a for a, c in pairs), sum(a < c for a, c in pairs)


def published_rows(root: Path = PUBLISHED) -> list[Row]:
    """Return each method's rows against each reference from the newest summary that has them."""
    newest: dict[tuple[str, str], tuple[str, list[Row]]] = {}
    for path in root.glob("*/summary.json"):
        data = json.loads(path.read_text())
        pairs: dict[tuple[str, str], list[Row]] = {}
        for row in paired_rows(path.parent.name, data):
            pairs.setdefault((row.method, row.comparator), []).append(row)
        for pair, rows in pairs.items():
            seen = newest.get(pair)
            if seen and seen[0] == data["generated"]:
                raise ValueError(f"{pair} is in two summaries generated on {seen[0]}")
            if not seen or seen[0] < data["generated"]:
                newest[pair] = (data["generated"], rows)
    return [row for _, rows in sorted(newest.values(), key=lambda v: v[0]) for row in rows]


def _matches(entry: dict, row: Row) -> bool:
    return (
        entry.get("summary") == row.summary
        and entry.get("method") == row.method
        and entry.get("comparator") == row.comparator
        and entry.get("row") == row.key
    )


def check(rows: list[Row], allowlist: list[dict]) -> tuple[list[str], list[str]]:
    """Return ``(failures, exemptions)``, one line each, for ``rows`` under ``allowlist``."""
    failed = [r for r in rows if r.failed]
    failures, exemptions = [], []
    for row in failed:
        entries = [e for e in allowlist if _matches(e, row)]
        entry = entries[0] if len(entries) == 1 else None
        problem = _entry_problem(entry, row) if entry else None
        if entry and not problem:
            exemptions.append(f"EXEMPT {row.label()}: {row.numbers()}; {entry['cause']}")
            continue
        detail = f"; allowlist entry rejected: {problem}" if problem else ""
        if len(entries) > 1:
            detail = f"; {len(entries)} allowlist entries match"
        failures.append(f"FAIL {row.label()}: {row.numbers()}{detail}")
    for entry in allowlist:
        if not any(_matches(entry, r) for r in failed):
            where = {k: entry.get(k) for k in ("summary", "method", "comparator", "row")}
            failures.append(f"FAIL allowlist entry matches no failing row: {where}")
    return failures, exemptions


def _entry_problem(entry: dict, row: Row) -> str | None:
    if entry.get("cause") not in CAUSES:
        return f"cause must be one of {sorted(CAUSES)}"
    if not str(entry.get("reason", "")).strip() or not str(entry.get("citation", "")).strip():
        return "it needs a reason and a citation"
    pinned = (entry.get("andrey_errors"), entry.get("comparator_errors"))
    if pinned != (row.andrey, row.reference):
        return f"it pins {pinned}, the row has ({row.andrey:g}, {row.reference:g})"
    if entry["cause"] in REVERSED_ORDER_CAUSES:
        return _reversed_order_problem(entry["cause"], entry.get("evidence"))
    if entry["cause"] == "random-draw":
        return _random_draw_problem(entry.get("evidence"))
    return None


def _passes(values: object) -> str | None:
    """Check one result's ``(andrey_errors, comparator_errors)``: numbers, where the row passes."""
    if not isinstance(values, dict):
        return "it needs andrey_errors and comparator_errors"
    pair = (values.get("andrey_errors"), values.get("comparator_errors"))
    if not all(isinstance(v, (int, float)) for v in pair):
        return "it needs andrey_errors and comparator_errors"
    if pair[0] > pair[1]:
        return f"the row still fails: {pair}"
    return None


def _reversed_order_problem(cause: str, evidence: object) -> str | None:
    """Check the row's error values with the columns reversed: present, and passing."""
    if not isinstance(evidence, dict) or evidence.get("column_order") != "reversed":
        return f"a {cause} entry needs evidence with the result on the reversed column order"
    problem = _passes(evidence)
    return f"its evidence: {problem}" if problem else None


def _random_draw_problem(evidence: object) -> str | None:
    """Check the row's error values under other seeds of the draws: at least two, each passing."""
    draws = evidence.get("draws") if isinstance(evidence, dict) else None
    if not isinstance(draws, list) or len(draws) < 2:
        return "a random-draw entry needs evidence with the row under at least two other draws"
    for draw in draws:
        problem = _passes(draw)
        if problem:
            return f"its evidence, draw {draw}: {problem}"
    return None


def main() -> int:
    rows = published_rows()
    allowlist = json.loads(ALLOWLIST.read_text())["entries"]
    failures, exemptions = check(rows, allowlist)
    print("Reference parity: Andrey's errors against the reference's, per published row.\n")
    for line in exemptions + failures:
        print(line)
    print(f"\n{len(rows)} rows, {len(exemptions)} exempt, {len(failures)} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
