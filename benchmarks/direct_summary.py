#!/usr/bin/env python
"""Judge a target-centered method of ``direct_cpu.py`` per cell: against global PC read at the
target, against the empty answer, and beside its paper's published counts.

The paired quantity is a dataset's mean over its targets of the errors at the target against the
truth's CPDAG (:func:`~andrey_bench.direct_lane.cpdag_counts`): the extra, missing, and reversed
edges, whose sum is ``total``; ``lenient`` leaves out the ``under`` reversals, edges the CPDAG
directs that the answer leaves undirected. For each comparison, :func:`~andrey_bench.pairing.compare`
gives the mean over paired datasets of the method's ``total`` minus the other's, with its
percentile-bootstrap 95% interval; the verdict is ``better`` or ``worse`` when the interval lies
wholly on that side of 0, over at least :data:`~andrey_bench.pairing.MIN_PAIRS` datasets. A cell
is judged on the ``--first`` seeds; a cell worse there is judged again on the ``--fresh`` seeds,
and counts as worse only when it is worse again. The floor is the method's ``total`` minus the
empty answer's, which passes when its interval lies wholly below 0.

    $ANDREY_BENCH_PYTHON benchmarks/direct_summary.py RUN [RUN ...] --method cmb --json rows.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from andrey_bench.pairing import MIN_PAIRS, compare

#: Each method's paper: its cells (network, rows, level) and its published counts per target, the
#: means over its datasets of the extra, missing, and reversed edges and their sum.
PAPERS = {
    "cmb": {
        "solution": "andrey.cmb",
        "source": "Gao and Ji (2015), Table 1",
        "cells": {
            ("alarm", 1000, 0.02): {"extra": 0.69, "missing": 0.61, "reversed": 0.51, "total": 1.81}
        },
    },
}

#: The solutions each method is judged against, and the empty answer of the floor.
AGAINST = ("andrey.pc.gsq", "bnlearn.pc-stable.graph")
EMPTY = "baseline.empty"
MEASURES = ("extra", "missing", "reversed", "under", "total", "lenient")


def load(runs: list[str]) -> list[dict]:
    """Every record of the run folders, each ``records.jsonl`` of ``direct_cpu.py``."""
    return [
        json.loads(line)
        for run in runs
        for line in (Path(run) / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]


def dataset_means(records: list[dict]) -> dict[tuple[str, str, float, int], dict]:
    """Per ``(cell, solution, alpha, seed)``: the mean of each measure over the targets, the
    number of targets, and whether every target was answered (no timeout or failure)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["cell"], r["solution"], r["params"].get("alpha"), r["seed"])].append(r)
    out = {}
    for key, rows in groups.items():
        scored = [r["scores"]["cpdag"] for r in rows if r["status"] == "ok" and r["scores"]]
        complete = len(scored) == len(rows)
        means = {}
        if complete:
            for m in MEASURES:
                values = [s["total"] - s["under"] if m == "lenient" else s[m] for s in scored]
                means[m] = float(np.mean(values))
        walls = {r["wall_s"] for r in rows if r["wall_s"] is not None}
        tests = [r["n_tests"] for r in rows if r["n_tests"] is not None]
        scope = rows[0]["scope"]
        out[key] = {
            "complete": complete,
            "targets": len(rows),
            "means": means,
            "wall_s": sum(walls) if scope == "graph" else sum(r["wall_s"] or 0 for r in rows),
            "tests": (tests[0] if scope == "graph" else sum(tests)) if tests else None,
        }
    return out


def _side(per: dict, seeds: list[int]) -> dict:
    fits = [per[s] for s in seeds if s in per and per[s]["complete"]]
    side = {"datasets": len(fits)}
    for m in MEASURES:
        side[m] = round(float(np.mean([f["means"][m] for f in fits])), 4) if fits else None
    side["wall_s_median"] = (
        round(float(np.median([f["wall_s"] for f in fits])), 3) if fits else None
    )
    return side


def _paired(a: dict, b: dict, seeds: list[int], measure: str = "total") -> dict:
    both = [s for s in seeds if a.get(s, {}).get("complete") and b.get(s, {}).get("complete")]
    if not both:
        return {"paired": 0, "mean_D": None, "bootstrap_95": None, "verdict": "too few pairs"}
    result = compare([a[s]["means"][measure] for s in both], [b[s]["means"][measure] for s in both])
    if result["paired"] < MIN_PAIRS:
        result["verdict"] = "too few pairs"
    return result


def replicated(first: dict, fresh: dict | None) -> str:
    """The cell's verdict: a cell worse on the first seeds is ``worse`` only when it is worse again
    on the fresh ones, ``tie`` when it is not, and ``worse, not replicated`` when the fresh seeds
    give too few pairs to judge."""
    if first["verdict"] != "worse":
        return first["verdict"]
    if fresh is None or fresh["verdict"] == "too few pairs":
        return "worse, not replicated"
    return "worse" if fresh["verdict"] == "worse" else "tie"


def cell_row(per: dict, method: str, cell: tuple, first: list[int], fresh: list[int]) -> dict:
    """The row of one cell: each solution's means, the comparisons, the floor, the paper."""
    network, n, alpha = cell
    label = f"net:{network}:{n}"
    solution = PAPERS[method]["solution"]

    def of(name: str) -> dict:
        # The empty answer takes no level.
        level = None if name == EMPTY else alpha
        return {k[3]: v for k, v in per.items() if k[:3] == (label, name, level)}

    mine, empty = of(solution), of(EMPTY)
    row = {
        "cell": label,
        "network": network,
        "n": n,
        "alpha": alpha,
        "solutions": {name: _side(of(name), first) for name in (solution, *AGAINST, EMPTY)},
        "paper": PAPERS[method]["cells"][cell],
        "against": {},
        "floor": _paired(mine, empty, first),
    }
    for name in AGAINST:
        first_d = _paired(mine, of(name), first)
        entry = {"first": first_d, "lenient": _paired(mine, of(name), first, "lenient")}
        if fresh:
            entry["fresh"] = _paired(mine, of(name), fresh)
        entry["verdict"] = replicated(first_d, entry.get("fresh"))
        row["against"][name] = entry
    if fresh:
        row["fresh_solutions"] = {
            name: _side(of(name), fresh) for name in (solution, *AGAINST, EMPTY)
        }
    interval = row["floor"]["bootstrap_95"]
    row["floor"]["pass"] = row["floor"]["paired"] >= MIN_PAIRS and interval[1] < 0
    return row


def _seeds(text: str) -> list[int]:
    low, high = (int(x) for x in text.split("-"))
    return list(range(low, high + 1))


def _said(result: dict, verdict: str) -> str:
    mean = result["mean_D"]
    return verdict if mean is None else f"{verdict} ({mean:+.3f})"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+", help="run folders of direct_cpu.py holding records.jsonl")
    p.add_argument("--method", choices=sorted(PAPERS), required=True)
    p.add_argument("--first", default="0-19", help="seeds judged first (default: 0-19)")
    p.add_argument("--fresh", default="20-39", help="seeds of the reruns (default: 20-39)")
    p.add_argument("--json", type=Path, default=None, help="write the rows here")
    args = p.parse_args(argv)
    per = dataset_means(load(args.runs))
    rows = [
        cell_row(per, args.method, cell, _seeds(args.first), _seeds(args.fresh))
        for cell in PAPERS[args.method]["cells"]
    ]
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")
    for row in rows:
        mine = row["solutions"][PAPERS[args.method]["solution"]]
        verdicts = ", ".join(
            f"{k}: {_said(v['first'], v['verdict'])}" for k, v in row["against"].items()
        )
        print(
            f"{row['cell']} alpha {row['alpha']}: total {mine['total']} (paper "
            f"{row['paper']['total']}); {verdicts}; floor {_said(row['floor'], 'pass' if row['floor']['pass'] else 'FAIL')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
