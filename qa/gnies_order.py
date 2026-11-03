"""Count how often reordering the columns or shuffling the rows changes GnIES's result.

Run ``uv run python -m qa.gnies_order -o order.json``; ``--reference`` also fits gnies 0.3.3 (the
interpreter ``ANDREY_BENCH_GNIES_PYTHON`` names, built by ``benchmarks/start_gnies.sh``), and
``--summarize`` prints the counts of one or more saved runs. The grid is the lane's A design
(``benchmarks/gnies_cpu.py``: an observational environment and three single-target noise
interventions) at 6, 8, 10, and 12 variables, 100 and 1000 rows per environment, seeds 0-9; at 10
variables the data are the lane's A100 and A1000 draws. Each model is fitted in its original order,
on three seeded column orders, and on three seeded row shuffles that carry each row's environment
with it. Each result is mapped back to the original columns, and a refit counts as changed when
an endpoint mark or a learned target differs. Andrey's fits run at the defaults through the lane's
adapter, which counts the DEBUG records of a step the tie ranks decide (``andrey.search.gies``) and
of an outer step between two candidates with one rounded total (``andrey.search.gnies``). Every
change is expected to come with such a record in one of its two fits; the summary counts the
changes with and without one. The reference is refitted on the same three column orders.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

import andrey

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import gnies_cpu  # noqa: E402

from andrey_bench.adapters import andrey_gnies, gnies_gnies  # noqa: E402

ORDERS = 3
SIZES = (6, 8, 10, 12)
ROWS = (100, 1000)
SEEDS = range(10)
#: The reference's cap per fit, seconds, as in the lane.
CAP_S = 1800.0


def model(d: int, n: int, seed: int) -> gnies_cpu.Draw:
    """The A design at ``d`` variables and ``n`` rows per environment, keyed as cell ``A<n>``."""
    cell = gnies_cpu.BY_NAME[f"A{n}"]
    return gnies_cpu.draw(dataclasses.replace(cell, d=d), seed, gnies_cpu.CELLS.index(cell))


def column_orders(d: int, seed: int) -> list[np.ndarray]:
    return [np.random.default_rng(seed * 31 + k + 991).permutation(d) for k in range(ORDERS)]


def row_orders(rows: int, seed: int) -> list[np.ndarray]:
    return [np.random.default_rng(seed * 31 + k + 1991).permutation(rows) for k in range(ORDERS)]


def andrey_fit(X: np.ndarray, env: np.ndarray) -> dict[str, Any]:
    out = andrey_gnies.fit(X, env, "andrey.gnies")
    counts = out["counts"]
    return {
        "marks": out["marks"],
        "targets": out["targets"],
        "score": out["deviance"],
        "tie": bool(counts.get("inner_ties", 0) or counts.get("outer_ties", 0)),
    }


def reference_fit(X: np.ndarray, env: np.ndarray) -> dict[str, Any]:
    answer = gnies_gnies.request(X, env, "fit", cap_s=CAP_S)
    if answer["status"] != "ok":
        return {"status": answer["status"], "error_text": answer.get("error_text")}
    marks = gnies_gnies.marks(answer["icpdag"])
    return {"status": "ok", "marks": marks, "targets": answer["targets"]}


def mapped_back(fit: dict[str, Any], order: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """A fit on ``X[:, order]`` in the original column positions."""
    back = np.argsort(order)
    return fit["marks"][np.ix_(back, back)], sorted(int(order[j]) for j in fit["targets"])


def same(fit: dict[str, Any], marks: np.ndarray, targets: list[int]) -> bool:
    return bool(np.array_equal(fit["marks"], marks)) and list(fit["targets"]) == targets


def check(d: int, n: int, seed: int, reference: bool) -> dict[str, Any]:
    """One model's refits, each as whether it changed and whether either fit logged a tie."""
    data = model(d, n, seed)
    X, env = data.X, data.env
    first = andrey_fit(X, env)
    row: dict[str, Any] = {"d": d, "n": n, "seed": seed, "data_sha256": data.sha256}
    columns = []
    for order in column_orders(d, seed):
        fit = andrey_fit(X[:, order], env)
        changed = not same(first, *mapped_back(fit, order))
        columns.append({"changed": changed, "tie": first["tie"] or fit["tie"]})
    shuffles = []
    for rows in row_orders(len(env), seed):
        fit = andrey_fit(X[rows], env[rows])
        changed = not same(first, fit["marks"], list(fit["targets"]))
        shuffles.append(
            {
                "changed": changed,
                "tie": first["tie"] or fit["tie"],
                "score_equal": fit["score"] == first["score"],
            }
        )
    row["andrey"] = {"tie": first["tie"], "columns": columns, "rows": shuffles}
    if reference:
        base = reference_fit(X, env)
        refits = []
        for order in column_orders(d, seed):
            fit = reference_fit(X[:, order], env)
            if base["status"] != "ok" or fit["status"] != "ok":
                refits.append({"changed": None, "status": fit["status"]})
                continue
            refits.append({"changed": not same(base, *mapped_back(fit, order)), "status": "ok"})
        row["reference"] = {"status": base["status"], "columns": refits}
    return row


def summary(rows: list[dict[str, Any]]) -> list[str]:
    """The published counts: changes, changes with a logged tie, changes without one."""
    lines = [f"{len(rows)} models"]
    for kind in ("columns", "rows"):
        refits = [r for row in rows for r in row["andrey"][kind]]
        changed = [r for r in refits if r["changed"]]
        tied = sum(r["tie"] for r in changed)
        what = "column orders" if kind == "columns" else "row shuffles"
        lines.append(
            f"andrey {what}: {len(changed)} of {len(refits)} change the result "
            f"({tied} with a logged tie, {len(changed) - tied} without)"
        )
    ties = sum(row["andrey"]["tie"] for row in rows)
    lines.append(f"andrey fits in the original order with a logged tie: {ties} of {len(rows)}")
    if any("reference" in row for row in rows):
        refits = [r for row in rows if "reference" in row for r in row["reference"]["columns"]]
        done = [r for r in refits if r["changed"] is not None]
        changed = sum(r["changed"] for r in done)
        models = sum(any(r["changed"] for r in row["reference"]["columns"]) for row in rows)
        lines.append(
            f"gnies 0.3.3 column orders: {changed} of {len(done)} change the result "
            f"({models} of {len(rows)} models; {len(refits) - len(done)} refits not finished)"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-o", "--out", help="write every model's refits to this JSON file")
    p.add_argument("--sizes", type=int, nargs="+", default=list(SIZES), help="variable counts")
    p.add_argument("--reference", action="store_true", help="also refit gnies 0.3.3")
    p.add_argument("--summarize", nargs="+", metavar="JSON", help="print the counts of saved runs")
    args = p.parse_args(argv)
    if args.summarize:
        rows = [row for path in args.summarize for row in json.loads(Path(path).read_text())]
        print("\n".join(summary(rows)))
        return 0
    if args.reference:
        gnies_gnies.interpreter()  # fail before any fit when the reference is not set up
    rows = []
    with (
        andrey.config(backend="numpy", n_jobs=1),
        threadpool_limits(limits=1),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        for d in args.sizes:
            for n in ROWS:
                for seed in SEEDS:
                    rows.append(check(d, n, seed, args.reference))
                    if args.out:
                        Path(args.out).write_text(json.dumps(rows))
    print("\n".join(summary(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
