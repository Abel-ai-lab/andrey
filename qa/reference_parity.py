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
- ``replication``: a row whose setting came out worse may hold its rerun on fresh seeds, with the
  same fields (``paired``, ``andrey_shd_mean``, ``comparator_shd_mean``, ``mean_D``,
  ``bootstrap_95``, ``above_zero``) and the ``seeds`` it ran. A worse setting counts only if it
  replicates, so a row with a replication is judged by the replication's own interval, and a
  replication with fewer than ``MIN_PAIRS`` paired datasets is reported, not gated.
- ``claim``: ``"out"``, on a row or on the entry that holds it, marks a row measured outside the
  setting its method claims (the model, data regime, and scale its paper states). Such a row is
  measured for the method's Limitations, so it is reported with its numbers and never gated. A row
  with ``"claim": "in"``, or with no ``claim``, is in claim and is judged as above; the row's own
  field overrides its entry's.
- ``metric``, on an entry: the error its rows pair, ``"shd"`` (the default when absent),
  ``"target_error"`` (the size of the symmetric difference of the learned and true intervention
  targets), or ``"direct_errors"`` (the directed edges against the edges between observed
  variables that no unobserved causal or backdoor path joins). A row's means are
  ``andrey_<metric>_mean`` and ``comparator_<metric>_mean``, or the ``*_shd_mean`` fields, which
  then hold that error's means. A method with a gated metric beside SHD has one entry per metric,
  and a row's key names the ``metric``.
- ``role``: ``"context"`` on an entry marks a comparator the method is reported against and never
  judged by, such as another method on the same data. Its rows are reported with their numbers.
- ``coverage``: a row's ``andrey_short`` counts the datasets the reference finished and Andrey did
  not, under the time cap. An in-claim row with ``andrey_short`` above 0 fails, whatever its number
  of paired datasets: a timeout never becomes a smaller mean. So does its replication's.
- ``floor_95`` or ``target_floor_95``, or a ``floor`` record with the interval in
  ``andrey_minus_empty.bootstrap_95``: the bootstrap 95% interval of the mean of Andrey's error
  minus a floor's, the empty graph's error, or for ``target_error`` the error of the targets the
  method was given or of no targets. An in-claim row fails when the interval's upper end is not
  below 0. A ``floor`` record's ``applies`` holds its lane's rule for the row, such as applying the
  floor only where the reference's own interval against the empty graph lies wholly below 0. A
  floor with ``applies`` false is printed with the row's numbers and not judged; one with
  ``applies`` true or absent is judged, and a ``pass`` beside it must agree with the interval. A
  row with a ``replication`` fails a floor that applies only when the replication's floor, where it
  records one, fails too.
- no ``parity_rule`` (the summaries released before 0.8.0): a row fails when the reference has
  fewer errors on more series than Andrey; where the row records only its central errors (the mean
  or median over the series), it fails when Andrey's are higher, at any margin.

A row of a reference whose result depends on the column order may also hold Andrey's column-order
mode on the same datasets (``andrey_column_shd_by_seed``, ``comparator_shd_by_seed``). That arm is
context: its series counts are printed beside the row and never judged.

``parity-allowlist.json``, next to the summaries, exempts a failing row only when it names the
cause: Andrey is correct by the paper the method comes from, the reference has a defect, the two
packages end at different local optima after an exact score tie that Andrey breaks by column order
(a tie-break), or the row is the cost of a default chosen for what it gains on other rows (a
trade-off). Each entry gives the reason and a citation, and pins the row's two error values, so
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
from dataclasses import dataclass, replace
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
#: The ``claim`` values: a row in the setting its method claims, and one outside it.
IN_CLAIM, OUT_OF_CLAIM = "in", "out"
#: The ``role`` of an entry whose comparator is reported and never judged.
CONTEXT = "context"
#: A row's floor fields: Andrey's error minus the empty graph's, minus the given targets', or a
#: ``floor`` record that holds the interval and whether its lane's rule applies it to the row.
#: ``floor_95`` and ``target_floor_95`` always apply.
FLOORS: tuple[str, ...] = ("floor_95", "target_floor_95", "floor")
#: The ``metric`` values an entry may name, with the text a row prints for each.
METRICS: dict[str, str] = {
    "shd": "SHD",
    "target_error": "target error",
    "direct_errors": "direct-edge errors",
}
#: The causes an allowlist entry may name.
CAUSES: frozenset[str] = frozenset(
    {"correct-by-paper", "reference-defect", "tie-break", "trade-off"}
)


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
    counts, which are printed but not judged. ``replicated`` marks a row judged by its
    ``replication`` on fresh seeds: its numbers are the rerun's. ``out_of_claim`` marks a row
    measured outside its method's claim and ``context`` a row against a comparator the method is
    not judged by: both are reported with their numbers and never gated. ``metric`` names the
    error the row compares, ``andrey_short`` the datasets the reference finished and Andrey did
    not, ``floor`` the interval of Andrey's error minus its floor's, and ``floor_applies`` whether
    the lane's rule applies that floor to the row; a floor that does not apply is printed, never
    judged.
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
    replicated: bool = False
    out_of_claim: bool = False
    context: bool = False
    metric: str = "shd"
    andrey_short: int = 0
    floor: tuple[float, float] | None = None
    floor_applies: bool = True

    @property
    def reported(self) -> bool:
        """Whether the row is not gated: out of claim, context, or under ``MIN_PAIRS`` paired
        datasets."""
        return (
            self.out_of_claim or self.context or (self.pairs is not None and self.pairs < MIN_PAIRS)
        )

    @property
    def failed(self) -> bool:
        if self.out_of_claim or self.context:
            return False
        if self.andrey_short > 0 or (self.floor_applies and self.floor and self.floor[1] >= 0):
            return True
        if self.interval is not None:
            return self.interval[0] > 0
        return not self.reported and self.andrey > self.reference

    def label(self) -> str:
        where = " ".join(f"{k}={v}" for k, v in self.key.items())
        return f"{self.method} vs {self.comparator}, {where} ({self.summary})"

    def numbers(self) -> str:
        text = self._numbers()
        return f"replication on fresh seeds: {text}" if self.replicated else text

    def _numbers(self) -> str:
        if self.interval is not None:
            lo, hi = self.interval
            each = f", the mean of {self.draws} draws each" if self.draws > 1 else ""
            text = (
                f"mean D {self.mean_d:+.2f}, bootstrap 95% [{lo:+.2f}, {hi:+.2f}], over "
                f"{self.pairs} paired datasets{each} (mean {METRICS[self.metric]} Andrey "
                f"{self.andrey:.2f}, {self.comparator} {self.reference:.2f})"
            )
        elif self.pairs == 0:
            text = "no paired dataset"
        elif self.rule == "series":
            text = (
                f"{self.comparator} has fewer errors on {self.andrey:g} series, "
                f"Andrey on {self.reference:g}"
            )
        else:
            text = f"{self.rule} Andrey {self.andrey:g}, {self.comparator} {self.reference:g}"
        if self.andrey_short:
            text += (
                f"; {self.andrey_short} datasets finished by {self.comparator} and not by Andrey"
            )
        if self.floor is not None:
            lo, hi = self.floor
            below = "below 0" if hi < 0 else "not below 0"
            text += f"; Andrey minus its floor: bootstrap 95% [{lo:+.2f}, {hi:+.2f}], {below}"
            if not self.floor_applies:
                text += ", a floor its lane's rule does not apply to this row: not judged"
        if self.out_of_claim:
            text += "; out of the method's claim: reported, not gated"
        elif self.context:
            text += "; a context comparator: reported, not gated"
        elif self.reported:
            text += f"; {self.pairs} paired datasets, under {MIN_PAIRS}: reported, not gated"
        if self.column is not None:
            text += (
                f"; Andrey's column-order mode (context, not judged): {self.comparator} fewer "
                f"errors on {self.column[0]} series, Andrey on {self.column[1]}"
            )
        return text


def paired_rows(summary: str, data: dict) -> list[Row]:
    """Return the rows of one summary that pair Andrey with a reference package.

    An entry's ``metric`` joins the key of each of its rows, and two entries of one method against
    one reference need a ``regime`` only when they also share their metric. A row where either side
    has no result (a fit over the time cap) is skipped, and so is a series of the column-order arm.
    A paired row with no known error field, a ``paired bootstrap`` row with ``MIN_PAIRS`` paired
    datasets and no consistent interval, an unknown ``parity_rule``, an unknown ``claim``,
    ``metric`` or ``role``, a ``coverage`` without ``andrey_short``, a floor that is not an
    interval, and a column-order arm on a reference not marked order-dependent or with one of its
    two series missing or of unequal lengths raise, so a new summary format cannot pass unchecked.
    """
    rule = data.get("parity_rule")
    if rule not in (None, PAIRED_BOOTSTRAP):
        raise ValueError(f"{summary}: unknown parity_rule {rule!r}")
    entries = Counter(
        (m["method"], m.get("comparator"), m.get("metric", "shd")) for m in data["methods"]
    )
    rows: list[Row] = []
    for method in data["methods"]:
        comparator = method.get("comparator")
        if not comparator:
            continue
        metric = method.get("metric", "shd")
        if metric not in METRICS:
            raise ValueError(f"{summary}: {method['method']} has an unknown metric {metric!r}")
        by_regime = entries[(method["method"], comparator, metric)] > 1
        if by_regime and not method.get("regime"):
            raise ValueError(
                f"{summary}: {method['method']} vs {comparator} has two entries and no regime"
            )
        role = method.get("role")
        if role not in (None, CONTEXT):
            raise ValueError(
                f"{summary}: {method['method']} vs {comparator}: unknown role {role!r}"
            )
        for row in method.get("rows", []):
            key = {"regime": method["regime"]} if by_regime else {}
            if "metric" in method:
                key["metric"] = metric
            key.update({k: row[k] for k in ROW_KEYS if k in row})
            where = f"{summary}: {method['method']} vs {comparator} {key}"
            name = (summary, method["method"], comparator, key)
            column = _column_series(row, where)
            marks = {
                "out_of_claim": _out_of_claim(row, method, where),
                "context": role == CONTEXT,
                "metric": metric,
                "andrey_short": _andrey_short(row, where),
            }
            marks["floor"], marks["floor_applies"] = _floor(row, where)
            judged = row.get("multi_draw") or row
            pairs = judged.get("paired") if rule == PAIRED_BOOTSTRAP else None
            if rule == PAIRED_BOOTSTRAP and not isinstance(pairs, int):
                raise ValueError(f"{where} has no paired count")
            if "replication" in row:
                if rule != PAIRED_BOOTSTRAP:
                    raise ValueError(f"{where} has a replication, but no paired bootstrap rule")
                replication = row["replication"]
                short = marks["andrey_short"] + _andrey_short(replication, f"{where}, replication")
                floor = marks["floor"]
                fails = marks["floor_applies"] and floor is not None and floor[1] >= 0
                if fails and any(k in replication for k in FLOORS):
                    marks["floor"], marks["floor_applies"] = _floor(
                        replication, f"{where}, replication"
                    )
                rows.append(
                    replace(
                        _replicated(name, replication, column, where, metric),
                        **{**marks, "andrey_short": short},
                    )
                )
                continue
            if pairs == 0:
                if marks["andrey_short"]:
                    nan = float("nan")
                    rows.append(Row(*name, PAIRED_BOOTSTRAP, nan, nan, pairs=0, **marks))
                continue
            if pairs is not None and pairs >= MIN_PAIRS:
                rows.append(replace(_judged(name, judged, column, where, metric), **marks))
                continue
            counted = _errors(row, where)
            if counted is not None:
                rows.append(Row(*name, *counted, pairs=pairs, column=column, **marks))
    return rows


def _andrey_short(row: dict, where: str) -> int:
    """Return the datasets the reference finished and Andrey did not, from ``coverage``."""
    if "coverage" not in row:
        return 0
    short = row["coverage"].get("andrey_short")
    if not isinstance(short, int):
        raise ValueError(f"{where}: coverage needs andrey_short")
    return short


def _floor(row: dict, where: str) -> tuple[tuple[float, float] | None, bool]:
    """Return the row's floor interval, ``None`` without one, and whether its lane applies it.

    A ``floor`` record's ``applies`` must be absent (applies) or a bool. Its ``pass`` must be empty
    where the floor does not apply and otherwise agree with the interval, so a lane's verdict and
    this check cannot disagree unseen.
    """
    present = [name for name in FLOORS if name in row]
    if not present:
        return None, True
    if len(present) > 1:
        raise ValueError(f"{where}: one floor interval of two numbers, from {FLOORS}")
    value = row[present[0]]
    applies = None
    if present[0] == "floor":
        if not isinstance(value, dict):
            raise ValueError(f"{where}: floor must be a record with andrey_minus_empty")
        applies = value.get("applies")
        if applies not in (None, True, False):
            raise ValueError(f"{where}: floor applies must be true or false, not {applies!r}")
        value = value.get("andrey_minus_empty", {}).get("bootstrap_95")
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{where}: one floor interval of two numbers, from {FLOORS}")
    lo, hi = float(value[0]), float(value[1])
    verdict = row["floor"].get("pass") if present[0] == "floor" else None
    if applies is False and verdict is not None:
        raise ValueError(f"{where}: floor pass {verdict!r} on a floor that does not apply")
    if verdict not in (None, hi < 0):
        raise ValueError(f"{where}: floor pass disagrees with its interval [{lo}, {hi}]")
    return (lo, hi), applies is not False


def _out_of_claim(row: dict, method: dict, where: str) -> bool:
    """Return whether the row is marked ``claim: "out"``, on itself or on its entry.

    The row's own ``claim`` overrides its entry's. A ``claim`` other than ``"in"`` and ``"out"``
    raises.
    """
    claim = row.get("claim", method.get("claim", IN_CLAIM))
    if claim not in (IN_CLAIM, OUT_OF_CLAIM):
        raise ValueError(f"{where}: claim must be {IN_CLAIM!r} or {OUT_OF_CLAIM!r}, not {claim!r}")
    return claim == OUT_OF_CLAIM


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


def _means(fields: dict, metric: str) -> tuple[str, str]:
    """Return the names of the row's mean fields for ``metric``: ``andrey_<metric>_mean`` and
    ``comparator_<metric>_mean``, or, when neither is there, the ``*_shd_mean`` fields that hold
    the metric's means."""
    names = (f"andrey_{metric}_mean", f"comparator_{metric}_mean")
    if all(fields.get(name) is None for name in names):
        return ("andrey_shd_mean", "comparator_shd_mean")
    return names


def _judged(
    name: tuple[str, str, str, dict[str, Any]],
    fields: dict,
    column: tuple[int, int] | None,
    where: str,
    metric: str = "shd",
) -> Row:
    """Return a row judged by its paired interval; ``fields`` is the row or its ``multi_draw``.

    The fields must all be present, with the means of ``metric``, and ``above_zero`` must say what
    the interval says.
    """
    means = _means(fields, metric)
    names = (*means, "mean_D", "bootstrap_95", "above_zero")
    if any(fields.get(field) is None for field in names):
        raise ValueError(f"{where} needs {names} for the paired bootstrap rule")
    lo, hi = fields["bootstrap_95"]
    if fields["above_zero"] != (lo > 0):
        raise ValueError(f"{where}: above_zero disagrees with bootstrap_95 {[lo, hi]}")
    return Row(
        *name,
        PAIRED_BOOTSTRAP,
        fields[means[0]],
        fields[means[1]],
        pairs=fields["paired"],
        mean_d=fields["mean_D"],
        interval=(lo, hi),
        draws=fields.get("draws", 1),
        column=column,
    )


def _replicated(
    name: tuple[str, str, str, dict[str, Any]],
    replication: dict,
    column: tuple[int, int] | None,
    where: str,
    metric: str = "shd",
) -> Row:
    """Return the row as its replication judges it, the rerun on fresh seeds.

    The replication names the ``seeds`` it ran. With ``MIN_PAIRS`` paired datasets it is judged by
    its own interval, like any row; with fewer, it keeps its means and is reported, not gated.
    """
    where = f"{where}, replication"
    if not str(replication.get("seeds", "")).strip():
        raise ValueError(f"{where} needs the seeds it ran")
    pairs = replication.get("paired")
    if not isinstance(pairs, int):
        raise ValueError(f"{where} has no paired count")
    if pairs >= MIN_PAIRS:
        return replace(_judged(name, replication, column, where, metric), replicated=True)
    fields = _means(replication, metric)
    means = (replication.get(fields[0]), replication.get(fields[1]))
    if None in means:
        raise ValueError(f"{where} needs {fields[0]} and {fields[1]}")
    return Row(*name, PAIRED_BOOTSTRAP, *means, pairs=pairs, column=column, replicated=True)


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
    unfloored = [
        f"FLOOR NOT APPLIED {r.label()}: {r.numbers()}"
        for r in rows
        if r.floor is not None and not r.floor_applies and not r.reported
    ]
    print("Reference parity: Andrey's errors against the reference's, per published row.\n")
    for line in exemptions + reported + unfloored + failures:
        print(line)
    print(
        f"\n{len(rows)} rows, {len(exemptions)} exempt, {len(reported)} reported (out of "
        f"claim, context, or under {MIN_PAIRS} paired datasets; not gated), {len(unfloored)} "
        f"judged without a floor their lane's rule does not apply, {len(failures)} failed."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
