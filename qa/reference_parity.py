"""Check that no published benchmark row has Andrey making more errors than its reference package.

Reads only the published summaries (``benchmarks/published/*/summary.json``), so it runs without
the reference packages. A row pairs Andrey with one reference package on one dataset size. It fails
when the reference has fewer errors on more series than Andrey; where the row records only its
central errors (the mean or median over the series), it fails when Andrey's are higher. Any
margin fails: a consistent shortfall, however small, is a shortfall.

``parity-allowlist.json``, next to the summaries, exempts a failing row only when it names the
cause: Andrey is correct by the paper the method comes from, the reference has a defect, or the two
packages end at different local optima after an exact score tie that Andrey breaks by column order
(a tie-break). Each entry gives the reason and a citation, and pins the row's two error values, so
a re-run that changes the row voids the entry. A tie-break entry also gives, as its evidence, the
row's two error values with the columns in reversed order, where the row must pass. An entry that
matches no failing row also fails. A released
summary never changes, so a method re-measured against the same reference in a later summary is
judged by the later rows only.

    uv run python -m qa.reference_parity              # exit 1 if any row fails
"""

from __future__ import annotations

import json
import sys
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
#: Error fields recorded as ``andrey_<field>`` and ``comparator_<field>``: the row's mean or median.
ERRORS: tuple[str, ...] = ("shd", "directed_errors")
#: Row fields that name a row within its method's table.
ROW_KEYS: tuple[str, ...] = ("mean_degree", "d", "observed", "n")
#: The causes an allowlist entry may name.
CAUSES: frozenset[str] = frozenset({"correct-by-paper", "reference-defect", "tie-break"})


@dataclass(frozen=True)
class Row:
    """One published row that pairs Andrey with a reference package.

    ``andrey`` and ``reference`` are each side's errors: the row's mean or median for an error
    field, or, under the ``series`` rule, the number of series where that side has more errors.
    """

    summary: str
    method: str
    comparator: str
    key: dict[str, Any]
    rule: str
    andrey: float
    reference: float

    @property
    def failed(self) -> bool:
        return self.andrey > self.reference

    def label(self) -> str:
        where = " ".join(f"{k}={v}" for k, v in self.key.items())
        return f"{self.method} vs {self.comparator}, {where} ({self.summary})"

    def numbers(self) -> str:
        if self.rule == "series":
            return (
                f"{self.comparator} has fewer errors on {self.andrey:g} series, "
                f"Andrey on {self.reference:g}"
            )
        return f"{self.rule} Andrey {self.andrey:g}, {self.comparator} {self.reference:g}"


def paired_rows(summary: str, data: dict) -> list[Row]:
    """Return the rows of one summary that pair Andrey with a reference package.

    A row where either side has no result (a fit over the time cap) is skipped. A paired row with
    no known error field raises, so a new summary format cannot pass unchecked.
    """
    rows = []
    for method in data["methods"]:
        comparator = method.get("comparator")
        if not comparator:
            continue
        for row in method.get("rows", []):
            key = {k: row[k] for k in ROW_KEYS if k in row}
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
            rows.append(Row(summary, method["method"], comparator, key, rule, andrey, reference))
    return rows


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
    if entry["cause"] == "tie-break":
        return _tie_break_problem(entry.get("evidence"))
    return None


def _tie_break_problem(evidence: object) -> str | None:
    """Check a tie-break's evidence: the row's error values with the columns reversed, passing."""
    if not isinstance(evidence, dict) or evidence.get("column_order") != "reversed":
        return "a tie-break needs evidence with the result on the reversed column order"
    values = (evidence.get("andrey_errors"), evidence.get("comparator_errors"))
    if not all(isinstance(v, (int, float)) for v in values):
        return "its evidence needs andrey_errors and comparator_errors"
    if values[0] > values[1]:
        return f"with the columns reversed the row still fails: {values}"
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
