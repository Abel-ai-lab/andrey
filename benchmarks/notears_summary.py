#!/usr/bin/env python
"""Summarize the NOTEARS and DAGMA lane of ``notears_cpu.py`` into paired rows.

One row per method (``notears``, ``dagma``) and cell. A cell is a graph, a size, a noise, and a
scale; its datasets differ by seed. Per row: every arm's mean SHD on the datasets it finished, its
statuses and median fit time, and the paired comparisons of Andrey's default fit with its reference
(the authors' code), with the empty graph (the floor), and with the two sortability controls. The
reference's own comparison with the floor stands beside Andrey's.

``shd`` is the distance to the true DAG, the papers' measure, and decides; ``mec_shd``, the
distance between the CPDAGs of the estimate and the truth, is reported with it (the ``_mec``
entries): with Gaussian noise on standardized columns the DAG is identified only up to its class.

A comparison is the per-dataset difference ``D = SHD(a) - SHD(b)`` on the datasets both arms
finished, with the 95% percentile-bootstrap interval of its mean over 10,000 resamples
(``default_rng(0)``): worse when the interval lies wholly above 0, better when wholly below, a tie
otherwise; fewer than 10 paired datasets are reported, not judged. A dataset the other arm finished
and the first did not is counted in ``failures``: an error or a timeout is never scored as a graph.

The decision cells are the standardized ones. ``decision`` gives, per method, the two bars:

- ``reference_bar``: no decision cell is worse than the reference. A worse cell is rerun on fresh
  seeds (``--replications``) and counts only when the rerun is worse too; until then the bar is
  ``None`` and the cell is listed in ``rerun``. A decision cell that cannot be judged fails it.
- ``floor_bar``: the ``shd`` interval against the empty graph lies below 0 in more than half of the
  decision cells.

``outcome`` is ``supported`` when both hold, ``not supported: floor`` when the floor bar fails,
``not promoted: reference`` when only the reference bar fails, and ``rerun`` while a worse cell
waits for its rerun. On the raw scale, ``same_as_standardized`` is the share of datasets on which
``standardize=True`` returns the graph the default fit returns on the same seed's standardized
dataset.

    $ANDREY_BENCH_PYTHON benchmarks/notears_summary.py --runs outputs/part-*/ --out notears.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from notears_cpu import DECISION_SCALE, GRAPHS, REFERENCES, SCALES

BOOTSTRAP = 10_000
MIN_PAIRS = 10
EMPTY = "baseline.empty"
CONTROLS = {
    "r2_control": "causaldisco.r2_sort_regress",
    "var_control": "causaldisco.var_sort_regress",
}
MEASURES = ("shd", "mec_shd", "skeleton_f1")
CELL = ["graph", "d", "noise", "scale"]


def bootstrap(values: np.ndarray) -> list[float] | None:
    """Percentile bootstrap 95% interval of the mean, ``default_rng(0)``."""
    if len(values) == 0:
        return None
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(0)
    means = v[rng.integers(0, len(v), size=(BOOTSTRAP, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def verdict(interval: list[float] | None, paired: int) -> str | None:
    """``worse``, ``better``, or ``tie`` of an interval of ``D``; ``None`` below the pair floor."""
    if interval is None or paired < MIN_PAIRS:
        return None
    return "worse" if interval[0] > 0 else "better" if interval[1] < 0 else "tie"


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def load(dirs: list[str]) -> pd.DataFrame:
    """The records of the part folders `dirs`, one per dataset and arm, with the cell's names."""
    frames = [pd.read_parquet(p) for d in dirs if (p := Path(d) / "runs.parquet").exists()]
    if not frames:
        raise SystemExit("no runs.parquet under --runs")
    table = pd.concat(frames, ignore_index=True)
    table = table.drop_duplicates(["dataset_id", "name"], keep="last").reset_index(drop=True)
    graph_of = {value: name for name, value in GRAPHS.items()}
    scale_of = {value: name for name, value in SCALES.items()}
    table["graph"] = [graph_of[(t, float(g))] for t, g in zip(table["topology"], table["density"])]
    table["scale"] = table["standardize"].map(scale_of)
    # A CPDAG arm, the empty graph, has its class distance in ``shd``.
    table["mec_shd"] = table["mec_shd"].where(table["output_type"] == "dag", table["shd"])
    return table


def compare(mine: pd.Series, other: pd.Series) -> dict[str, Any]:
    """The paired comparison of two arms' values by seed."""
    both = mine.index.intersection(other.index)
    d = (mine[both] - other[both]).to_numpy()
    interval = bootstrap(d)
    return {
        "paired": len(both),
        "mean_D": _round(d.mean()) if len(both) else None,
        "bootstrap_95": interval,
        "verdict": verdict(interval, len(both)),
        "failures": int(len(other.index.difference(mine.index))),
    }


def _arm(frame: pd.DataFrame) -> dict[str, Any]:
    """One arm's statuses, its means over the datasets it finished, and its median fit time."""
    ok = frame[frame["status"] == "ok"]
    return {
        "status": {str(k): int(v) for k, v in frame["status"].value_counts().items()},
        **{m: _round(ok[m].mean()) if len(ok) else None for m in MEASURES},
        "median_s": _round(ok["wall_s"].median()) if len(ok) else None,
    }


def _pairs(by_arm: dict[str, pd.DataFrame], a: str, b: str) -> dict[str, Any] | None:
    """`a` against `b` on ``shd``, with the ``mec_shd`` comparison under ``mec``."""
    if a not in by_arm or b not in by_arm:
        return None
    out = compare(by_arm[a]["shd"], by_arm[b]["shd"])
    out["mec"] = compare(by_arm[a]["mec_shd"], by_arm[b]["mec_shd"])
    return out


def cell_rows(cell: dict[str, Any], frame: pd.DataFrame, table: pd.DataFrame) -> list[dict]:
    """The rows of one cell, one per method with records in it."""
    finished = {
        str(name): group[group["status"] == "ok"].set_index("data_seed")
        for name, group in frame.groupby("name")
    }
    rows = []
    for method, reference in REFERENCES.items():
        andrey = f"andrey.{method}"
        if andrey not in finished:
            continue
        names = [andrey, reference, f"{andrey}.standardize", f"{andrey}.robust", *CONTROLS.values()]
        row: dict[str, Any] = {
            "method": method,
            **cell,
            "decision": cell["scale"] == DECISION_SCALE,
            "datasets": int(frame["dataset_id"].nunique()),
            "varsortability": _round(frame["varsortability"].mean()),
            "r2_sortability": _round(frame["r2_sortability"].mean()),
            "arms": {
                name: _arm(frame[frame["name"] == name])
                for name in [*names, EMPTY]
                if (frame["name"] == name).any()
            },
            "reference": _pairs(finished, andrey, reference),
            "floor": _pairs(finished, andrey, EMPTY),
            "reference_floor": _pairs(finished, reference, EMPTY),
        }
        if row["reference"] is not None:
            both = finished[andrey].index.intersection(finished[reference].index)
            same = finished[andrey].loc[both, "structure_hash"].to_numpy()
            same = same == finished[reference].loc[both, "structure_hash"].to_numpy()
            row["reference"]["same_graph"] = _round(same.mean()) if len(both) else None
        for role, control in CONTROLS.items():
            row[role] = _pairs(finished, andrey, control)
        for option in ("standardize", "robust"):
            arm = f"{andrey}.{option}"
            if arm not in finished:
                continue
            row[option] = {
                "floor": _pairs(finished, arm, EMPTY),
                "default": _pairs(finished, arm, andrey),
            }
        if "standardize" in row and cell["scale"] == "raw":
            # The default fit on the same seeds' standardized datasets.
            twin = table[
                (table["name"] == andrey)
                & (table["status"] == "ok")
                & (table["scale"] == DECISION_SCALE)
                & (table["graph"] == cell["graph"])
                & (table["d"] == cell["d"])
                & (table["noise"] == cell["noise"])
            ].set_index("data_seed")["structure_hash"]
            mine = finished[f"{andrey}.standardize"]["structure_hash"]
            both = mine.index.intersection(twin.index)
            row["standardize"]["same_as_standardized"] = (
                _round((mine[both].to_numpy() == twin[both].to_numpy()).mean())
                if len(both)
                else None
            )
        row["result"] = (row["reference"] or {}).get("verdict") if row["decision"] else None
        row["rerun"] = row["result"] == "worse"
        rows.append(row)
    return rows


def all_rows(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Every row of `table`, cells in order."""
    rows: list[dict[str, Any]] = []
    for values, frame in table.groupby(CELL, sort=True):
        cell = dict(zip(CELL, values))
        cell["d"] = int(cell["d"])
        rows += cell_rows(cell, frame, table)
    return rows


def _key(row: dict[str, Any]) -> tuple:
    return (row["method"], *(row[c] for c in CELL))


def decide(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    """The two bars of `method` over its decision rows, and the outcome they give."""
    cells = [r for r in rows if r["method"] == method and r["decision"]]
    label = [" ".join(str(r[c]) for c in CELL) for r in cells]
    unjudged = [n for n, r in zip(label, cells) if r["result"] is None]
    worse = [n for n, r in zip(label, cells) if r["result"] == "worse" and not r["rerun"]]
    rerun = [n for n, r in zip(label, cells) if r["rerun"]]

    def below(role: str, measure: str | None = None) -> int:
        hits = 0
        for r in cells:
            pair = r.get(role) or {}
            hits += (pair.get(measure, {}) if measure else pair).get("verdict") == "better"
        return hits

    reference_bar = False if worse or unjudged else None if rerun else True
    floor_bar = below("floor") > len(cells) / 2
    if not cells:
        outcome = "no decision cells"
    elif not floor_bar:
        outcome = "not supported: floor"
    elif reference_bar is None:
        outcome = "rerun"
    else:
        outcome = "supported" if reference_bar else "not promoted: reference"
    return {
        "cells": len(cells),
        "reference": {
            v: sum(r["result"] is not None and r["result"].startswith(v) for r in cells)
            for v in ("better", "tie", "worse")
        },
        "worse": worse,
        "rerun": rerun,
        "unjudged": unjudged,
        "andrey_failures": sum((r["reference"] or {}).get("failures", 0) for r in cells),
        "below_empty": below("floor"),
        "below_empty_mec": below("floor", "mec"),
        "reference_below_empty": below("reference_floor"),
        "reference_below_empty_mec": below("reference_floor", "mec"),
        "reference_bar": reference_bar,
        "floor_bar": floor_bar,
        "outcome": outcome,
    }


def summarize(args) -> int:
    """Pair the records of ``--runs`` into rows; ``--replications`` settles the worse cells."""
    table = load(args.runs)
    rows = all_rows(table)
    if args.replications:
        again = {_key(r): r for r in all_rows(load(args.replications))}
        for row in rows:
            if (rep := again.get(_key(row))) is None:
                continue
            row["replication"] = {k: rep[k] for k in ("datasets", "arms", "reference", "floor")}
            if row["rerun"]:
                agrees = (rep["reference"] or {}).get("verdict") == "worse"
                row["result"] = "worse" if agrees else "tie (did not replicate)"
                row["rerun"] = False
    ok = table[table["status"] == "ok"]
    summary = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lane": "notears-dagma",
        "parity_rule": "paired bootstrap",
        "bootstrap": {
            "resamples": BOOTSTRAP,
            "interval": "percentile 95%",
            "rng": "default_rng(0)",
        },
        "min_pairs": MIN_PAIRS,
        "decision_scale": DECISION_SCALE,
        "records": len(table),
        "fit_hours": round(float(ok["wall_s"].sum()) / 3600, 1),
        "status": {str(k): int(v) for k, v in table["status"].value_counts().items()},
        "decision": {method: decide(rows, method) for method in REFERENCES},
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=1) + "\n")
    print(f"{len(rows)} rows; status {summary['status']}")
    for method, result in summary["decision"].items():
        print(f"{method}: {json.dumps({k: v for k, v in result.items() if k != 'unjudged'})}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True, help="part folders of notears_cpu.py")
    parser.add_argument("--replications", nargs="+", help="part folders of the reruns")
    parser.add_argument("--out", required=True)
    return summarize(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
