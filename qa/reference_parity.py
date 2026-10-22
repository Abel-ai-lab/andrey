"""Check that no published benchmark row has Andrey making more errors than its reference package.

Reads only the published summaries (``benchmarks/published/*/summary.json``), so it runs without
the reference packages. A row pairs Andrey with one reference package on one dataset size. The
summary's ``parity_rule`` sets how its rows are judged:

- ``"paired bootstrap"``: the row's ``bootstrap_95`` is the percentile bootstrap 95% interval of the
  mean of D = SHD(Andrey) - SHD(reference) over its paired datasets (``paired``, the datasets where
  both finished). A row with at least ``MIN_PAIRS`` paired datasets fails only when that interval
  lies entirely above 0, and a row with fewer is reported, not gated. A row of a randomized test
  with several draws per dataset (``multi_draw``) is judged by those draws, with D per dataset the
  mean over the draws.
- no ``parity_rule`` (the summaries released before 0.8.0): a row fails when the reference has
  fewer errors on more series than Andrey; where the row records only its central errors (the mean
  or median over the series), it fails when Andrey's are higher, at any margin.

A row of a reference whose result depends on the column order may also hold Andrey's column-order
mode on the same datasets (``andrey_column_shd_by_seed``, ``comparator_shd_by_seed``). That arm is
context: its series counts are printed beside the row and never judged.

``parity-allowlist.json``, next to the summaries, exempts a failing row only when it names the
cause: Andrey is correct by the paper the method comes from, the reference has a defect, or the two
packages end at different local optima after an exact score tie that Andrey breaks by column order
(a tie-break). Each entry gives the reason and a citation, and pins the row's two error values, so
a re-run that changes the row voids the entry. A tie-break entry also gives, as its evidence, the
row's two error values with the columns in reversed order, where the row must pass. An entry that
matches no failing row also fails. A released summary never changes, so a method re-measured
against the same reference in a later summary is judged by the later rows only. Where a summary has
two entries of one method against one reference, one per regime, a row's key also names its
``regime``.

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

#: The ``parity_rule`` of a summary whose rows carry the paired bootstrap interval.
PAIRED_BOOTSTRAP = "paired bootstrap"
#: Paired datasets a row needs for its interval to gate it; a row with fewer is reported.
MIN_PAIRS = 10
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
CAUSES: frozenset[str] = frozenset({"correct-by-paper", "reference-defect", "tie-break"})


@dataclass(frozen=True)
class Row:
    """One published row that pairs Andrey with a reference package.

    ``andrey`` and ``reference`` are each side's errors: the row's mean or median for an error
    field, or, under the ``series`` rule, the number of series where the other side has fewer
    errors. Under the ``paired bootstrap`` rule they are each side's mean SHD over the ``pairs``
    paired datasets, ``mean_d`` is the mean of their difference, and ``interval`` its bootstrap 95%
    interval, over ``draws`` draws per dataset. A row of a ``paired bootstrap`` summary with fewer
    than ``MIN_PAIRS`` paired datasets keeps its counts or errors and its ``pairs``, and is
    reported. ``column`` holds the column-order arm's ``(reference fewer, Andrey fewer)`` series
    counts, which are printed but not judged.
    """

    summary: str
    method: str
    comparator: str
    key: dict[str, Any]
    rule: str
    andrey: float
    reference: float
    pairs: int | None = None
    mean_d: float | None = None
    interval: tuple[float, float] | None = None
    draws: int = 1
    column: tuple[int, int] | None = None

    @property
    def reported(self) -> bool:
        """Whether the row is too small to gate: under ``MIN_PAIRS`` paired datasets."""
        return self.pairs is not None and self.pairs < MIN_PAIRS

    @property
    def failed(self) -> bool:
        if self.interval is not None:
            return self.interval[0] > 0
        return not self.reported and self.andrey > self.reference

    def label(self) -> str:
        where = " ".join(f"{k}={v}" for k, v in self.key.items())
        return f"{self.method} vs {self.comparator}, {where} ({self.summary})"

    def numbers(self) -> str:
        if self.interval is not None:
            lo, hi = self.interval
            each = f", the mean of {self.draws} draws each" if self.draws > 1 else ""
            text = (
                f"mean D {self.mean_d:+.2f}, bootstrap 95% [{lo:+.2f}, {hi:+.2f}], over "
                f"{self.pairs} paired datasets{each} (mean SHD Andrey {self.andrey:.2f}, "
                f"{self.comparator} {self.reference:.2f})"
            )
        elif self.rule == "series":
            text = (
                f"{self.comparator} has fewer errors on {self.andrey:g} series, "
                f"Andrey on {self.reference:g}"
            )
        else:
            text = f"{self.rule} Andrey {self.andrey:g}, {self.comparator} {self.reference:g}"
        if self.reported:
            text += f"; {self.pairs} paired datasets, under {MIN_PAIRS}: reported, not gated"
        if self.column is not None:
            text += (
                f"; Andrey's column-order mode (context, not judged): {self.comparator} fewer "
                f"errors on {self.column[0]} series, Andrey on {self.column[1]}"
            )
        return text


def paired_rows(summary: str, data: dict) -> list[Row]:
    """Return the rows of one summary that pair Andrey with a reference package.

    A row where either side has no result (a fit over the time cap) is skipped, and so is a series
    of the column-order arm. A paired row with no known error field, a ``paired bootstrap`` row
    with ``MIN_PAIRS`` paired datasets and no consistent interval, an unknown ``parity_rule``, and
    a column-order arm on a reference not marked order-dependent or with one of its two series
    missing or of unequal lengths raise, so a new summary format cannot pass unchecked.
    """
    rule = data.get("parity_rule")
    if rule not in (None, PAIRED_BOOTSTRAP):
        raise ValueError(f"{summary}: unknown parity_rule {rule!r}")
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
            where = f"{summary}: {method['method']} vs {comparator} {key}"
            name = (summary, method["method"], comparator, key)
            column = _column_series(row, where)
            judged = row.get("multi_draw") or row
            pairs = judged.get("paired") if rule == PAIRED_BOOTSTRAP else None
            if rule == PAIRED_BOOTSTRAP and not isinstance(pairs, int):
                raise ValueError(f"{where} has no paired count")
            if pairs == 0:
                continue
            if pairs is not None and pairs >= MIN_PAIRS:
                rows.append(_judged(name, judged, column, where))
                continue
            counted = _errors(row, where)
            if counted is not None:
                rows.append(Row(*name, *counted, pairs=pairs, column=column))
    return rows


def _errors(row: dict, where: str) -> tuple[str, float, float] | None:
    """Return the row's ``(rule, andrey, reference)`` by its counts or errors; ``None`` without
    a result on one side."""
    counts = next(((a, c) for a, c in SERIES_COUNTS if a in row and c in row), None)
    if counts:
        rule, andrey, reference = "series", row[counts[1]], row[counts[0]]
    else:
        field = next((f for f in ERRORS if f"andrey_{f}" in row), None)
        if field is None or f"comparator_{field}" not in row:
            raise ValueError(f"{where} has no error field from {ERRORS} or {SERIES_COUNTS}")
        rule, andrey, reference = field, row[f"andrey_{field}"], row[f"comparator_{field}"]
    if andrey is None or reference is None:
        return None
    return rule, andrey, reference


def _judged(
    name: tuple[str, str, str, dict[str, Any]],
    fields: dict,
    column: tuple[int, int] | None,
    where: str,
) -> Row:
    """Return a row judged by its paired interval; ``fields`` is the row or its ``multi_draw``.

    The fields must all be present, and ``above_zero`` must say what the interval says.
    """
    names = ("andrey_shd_mean", "comparator_shd_mean", "mean_D", "bootstrap_95", "above_zero")
    if any(fields.get(field) is None for field in names):
        raise ValueError(f"{where} needs {names} for the paired bootstrap rule")
    lo, hi = fields["bootstrap_95"]
    if fields["above_zero"] != (lo > 0):
        raise ValueError(f"{where}: above_zero disagrees with bootstrap_95 {[lo, hi]}")
    return Row(
        *name,
        PAIRED_BOOTSTRAP,
        fields["andrey_shd_mean"],
        fields["comparator_shd_mean"],
        pairs=fields["paired"],
        mean_d=fields["mean_D"],
        interval=(lo, hi),
        draws=fields.get("draws", 1),
        column=column,
    )


def _column_series(row: dict, where: str) -> tuple[int, int] | None:
    """Return the column-order arm's ``(reference fewer, Andrey fewer)`` series counts.

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
    reported = [f"REPORTED {r.label()}: {r.numbers()}" for r in rows if r.reported]
    print("Reference parity: Andrey's errors against the reference's, per published row.\n")
    for line in exemptions + reported + failures:
        print(line)
    print(
        f"\n{len(rows)} rows, {len(exemptions)} exempt, {len(reported)} reported (under "
        f"{MIN_PAIRS} paired datasets, not gated), {len(failures)} failed."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
