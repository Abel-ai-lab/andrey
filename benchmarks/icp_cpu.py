#!/usr/bin/env python
"""Benchmark ICP on CPU: Andrey against R's InvariantCausalPrediction, on data from environments.

Two parts, each with the settings of ``comparisons/icp.md``:

- ``--fixed``: the fixed cases F1-F9 (:data:`FIXED`), small seeded datasets that reach each branch:
  two and more environments, both tests, few rows, a target without parents, copies of a parent,
  an intervention on the target, a cap on the set size, and another level. R reports each tested
  set's p-value from its own ``getpval``, and the comparison checks every output of Andrey's
  ``icp`` against the outputs those p-values imply under Andrey's rules, then against R's
  ``ICP()`` itself (:func:`compare_fixed`).
- the campaign: the cells of :data:`CELLS`, linear Gaussian SEMs drawn as in Appendix G of Peters,
  Buhlmann and Meinshausen (2016) (:func:`scenario`, :func:`sample`), scored against the target's
  true parents: false selection (an estimate that is not a subset of them, whose rate is the
  FWER), success (the estimate equals them), the share of them found, the model-rejected share,
  and the fit time. Cells B and E are out of the method's claim. R fits the first scenarios and
  datasets of a cell, on the same data as Andrey.

Andrey fits run in this interpreter, one thread each; R fits run in ``Rscript`` with
``andrey_bench/icp_worker.R``, one call per batch of datasets, and the clock covers each ``ICP()``
call alone. One worker process runs each scenario and writes ``tasks/<cell>/<scenario>.json``, so
``--resume`` skips the finished scenarios. ``--summary`` writes ``summary.json`` from the task files.

    $ANDREY_BENCH_PYTHON benchmarks/icp_cpu.py --fixed --out outputs/icp-fixed
    $ANDREY_BENCH_PYTHON benchmarks/icp_cpu.py --out outputs/icp-cpu --workers 4
    $ANDREY_BENCH_PYTHON benchmarks/icp_cpu.py --out outputs/icp-cpu --summary
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402 - BLAS threads are set before NumPy loads
import csv  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
WORKER = HERE / "andrey_bench" / "icp_worker.R"
ALPHA = 0.05
TESTS = {"approximate": "normal", "exact": "exact"}  # Andrey's name -> R's

# --- data -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SEM:
    """A linear Gaussian SEM and its environments.

    ``B[j, i]`` is the weight of ``i -> j`` and ``order`` a causal order; ``variances`` are the
    noise variances. Each environment ``k`` has ``rows[k]`` rows and changes the variables in
    ``changes[k]``: ``("scale", low, high)`` multiplies the noise by a draw from ``U(low, high)``
    per row, with the weights ``weights[k]`` into the variable, and ``("do", mean, sd)`` sets it to
    ``N(mean, sd^2)``. ``hidden`` is a variable left out of the data, ``nonlinear`` a parent whose
    effect on the target is ``tanh``, and ``target`` the response.
    """

    B: np.ndarray
    order: tuple[int, ...]
    variances: np.ndarray
    rows: tuple[int, ...]
    changes: tuple[dict, ...]
    weights: tuple[np.ndarray, ...]
    target: int
    hidden: int | None = None
    nonlinear: int | None = None


def _grid(rng: np.random.Generator, low: float, high: float) -> float:
    """A uniform draw from the grid ``low, low + 0.1, ..., high``."""
    return round(0.1 * rng.integers(round(10 * low), round(10 * high) + 1), 1)


def _weights(rng: np.random.Generator, size: int, low: float, high: float) -> np.ndarray:
    return rng.uniform(low, high, size) * rng.choice((-1.0, 1.0), size)


def scenario(cell: str, index: int, spec: Cell | None = None) -> SEM:
    """The SEM of one scenario: Appendix G's parameters, then the DAG, weights, target, and the
    intervened variables, in that order, from the scenario's seed.

    Cell B draws cell A's scenario and lets the intervened set hold the target, as Appendix G does.
    ``spec`` replaces the cell's spec, for a smaller run.
    """
    spec = CELLS[cell] if spec is None else spec
    rng = np.random.default_rng([658, CELL_IDS["A" if cell == "B" else cell], index])
    low, high = spec.nodes
    p = int(rng.integers(low, high + 1)) if low < high else low
    if spec.kind == "do":
        p = (6, 10)[index % 2]
    if spec.kind == "capped":
        p = spec.nodes[index * len(spec.nodes) // spec.scenarios] if low < high else low
    n_obs, n_int = (int(100 * rng.integers(1, 6)) for _ in range(2))
    k = int(rng.integers(1, 5))
    lb, delta_b = _grid(rng, 0.1, 2.0), _grid(rng, 0.1, 1.0)
    s2_min = _grid(rng, 0.1, 2.0)
    s2_max = _grid(rng, s2_min, 2.0)
    a_min = _grid(rng, 0.1, 4.0)
    delta_a = 0.0 if rng.random() < 1 / 3 else _grid(rng, 0.1, 2.0)
    change = rng.random() < 1 / 3
    lb2, ub2 = sorted(_grid(rng, 0.1, 2.0) for _ in range(2))
    single = rng.random() < 1 / 6
    inverse_theta = _grid(rng, 1.1, 3.0)
    if spec.kind == "speed":
        p, (n_obs, n_int) = spec.nodes[0], spec.rows

    order = tuple(int(j) for j in rng.permutation(p))
    B = np.zeros((p, p))
    for a in range(p):
        for b in range(a + 1, p):
            if rng.random() < min(1.0, k / max(p - 1, 1)):
                B[order[b], order[a]] = _weights(rng, 1, lb, lb + delta_b)[0]
    variances = rng.uniform(s2_min, s2_max, p)
    parents = (B != 0).sum(axis=1)
    pool = np.arange(p)
    if spec.kind == "capped":
        pool = np.flatnonzero(parents <= spec.max_parents)
    if spec.kind == "nonlinear":
        pool = np.flatnonzero(parents >= 1) if (parents >= 1).any() else pool
    target = int(rng.choice(pool))

    hidden = nonlinear = None
    if spec.kind == "hidden":  # a hidden parent of the target and of one other variable
        hidden = p
        B = np.pad(B, ((0, 1), (0, 1)))
        other = int(rng.choice([j for j in range(p) if j != target]))
        B[target, hidden], B[other, hidden] = _weights(rng, 2, 0.5, 1.0)
        variances = np.append(variances, 1.0)
        order = (hidden, *order)
    if spec.kind == "nonlinear":
        nonlinear = int(rng.choice(np.flatnonzero(B[target] != 0)))

    if spec.kind == "do":
        n_targets = (2, 3)[index // 2 % 2]
        others = [j for j in range(p) if j != target]
        hit = rng.choice(others, n_targets, replace=False)
        changes = ({}, *({int(j): ("do", 2.0, 0.2)} for j in hit))
        rows = (200,) * (n_targets + 1)
        return SEM(B, order, variances, rows, changes, (B,) * len(rows), target)

    candidates = [j for j in range(p) if spec.kind == "target" or j != target]
    size = 1 if single else max(1, round(len(candidates) / inverse_theta))
    A = [int(j) for j in rng.choice(candidates, size, replace=False)]
    B2 = B.copy()
    if change:
        for j in A:
            into = np.flatnonzero(B[j, :p])
            B2[j, into] = _weights(rng, into.size, lb2, ub2)
    intervened = {j: ("scale", a_min, a_min + delta_a) for j in A}
    return SEM(
        B, order, variances, (n_obs, n_int), ({}, intervened), (B, B2), target, hidden, nonlinear
    )


def sample(sem: SEM, seed: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """One dataset of ``sem``: the observed columns and each row's environment."""
    rng = np.random.default_rng(seed)
    blocks = []
    for rows, changes, B in zip(sem.rows, sem.changes, sem.weights, strict=True):
        X = np.zeros((rows, B.shape[0]))
        for j in sem.order:
            noise = rng.normal(0.0, np.sqrt(sem.variances[j]), rows)
            change = changes.get(j)
            if change is not None and change[0] == "scale":
                noise *= rng.uniform(change[1], change[2], rows)
            inputs = X.copy()
            if j == sem.target and sem.nonlinear is not None:
                inputs[:, sem.nonlinear] = np.tanh(2.0 * inputs[:, sem.nonlinear])
            X[:, j] = inputs @ B[j] + noise
            if change is not None and change[0] == "do":
                X[:, j] = rng.normal(change[1], change[2], rows)
        blocks.append(X)
    X = np.vstack(blocks)
    env = np.repeat(np.arange(len(sem.rows)), sem.rows)
    if sem.hidden is not None:
        X = np.delete(X, sem.hidden, axis=1)
    return X, env


def true_parents(sem: SEM) -> list[int]:
    """The target's observed parents."""
    p = len(sem.variances) - (sem.hidden is not None)
    return [int(j) for j in np.flatnonzero(sem.B[sem.target, :p])]


# --- the cells -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One cell of the campaign: ``scenarios`` SEMs, ``datasets`` Andrey fits of each per test,
    and R fits of the first ``r_datasets`` datasets of the first ``r_scenarios`` scenarios."""

    claim: str
    kind: str
    nodes: tuple[int, ...]
    scenarios: int
    datasets: int
    r_scenarios: int = 0
    r_datasets: int = 0
    max_parents: int | None = None
    rows: tuple[int, int] = (500, 500)
    r_tests: tuple[str, ...] = ("approximate", "exact")


CELLS = {
    "A": Cell("in", "appendix", (5, 12), 100, 1000, 100, 50),
    "B": Cell("out", "target", (5, 12), 100, 200),
    "C": Cell("in", "do", (6, 10), 20, 200, 20, 50),
    "D": Cell("in", "capped", (16, 21), 20, 200, 10, 20, max_parents=3),
    "D40": Cell("in", "capped", (40, 40), 20, 50, max_parents=3),
    "E-hidden": Cell("out", "hidden", (5, 12), 20, 200),
    "E-nonlinear": Cell("out", "nonlinear", (5, 12), 20, 200),
    "speed-9": Cell("in", "speed", (9, 9), 5, 1, 5, 1),
    "speed-12": Cell("in", "speed", (12, 12), 5, 1, 5, 1),
    "speed-15": Cell("in", "speed", (15, 15), 5, 1, 5, 1),
    "speed-12-large": Cell(
        "in", "speed", (12, 12), 5, 1, 5, 1, rows=(5000, 5000), r_tests=("approximate",)
    ),
}
CELL_IDS = {name: k for k, name in enumerate(CELLS)}


def r_scenario(cell: Cell, index: int) -> bool:
    """Whether R fits scenario ``index``: the first of each node count in a capped cell."""
    if cell.kind == "capped" and cell.r_datasets:
        per = max(1, cell.scenarios // len(cell.nodes))
        return index % per < max(1, cell.r_scenarios // len(cell.nodes))
    return index < cell.r_scenarios


# --- fits -----------------------------------------------------------------------------------------


def andrey_fit(X, env, target, test, max_parents, alpha: float = ALPHA) -> dict:
    """One Andrey fit, timed, by 0-based column position."""
    import andrey

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        out = andrey.icp(
            X, env, target=target, alpha=alpha, invariance_test=test, max_parents=max_parents
        )
        seconds = time.perf_counter() - start
    meta = out.metadata
    return {
        "parents": meta["parents"],
        "accepted": meta["accepted_sets"],
        "p_values": meta["p_values"],
        "intervals": meta["confidence_intervals"],
        "best": meta["best_p_value"],
        "model_rejected": meta["model_rejected"],
        "seconds": seconds,
    }


def _floats(text: str) -> list[float]:
    """R's values joined by ``;``, ``NA`` as ``NaN``."""
    return [float("nan") if v == "NA" else float(v) for v in text.split(";")] if text else []


def _sets(text: str, others: list[int]) -> list[list[int]]:
    """R's sets, 1-based predictor positions, as sorted column positions."""
    if not text:
        return []
    return [
        [] if s == "{}" else sorted(others[int(k) - 1] for k in s.split(","))
        for s in text.split("|")
    ]


def r_fits(
    datasets: list[tuple[np.ndarray, np.ndarray]],
    target: int,
    tests: tuple[str, ...],
    max_parents: int | None,
    *,
    alpha: float = ALPHA,
    fixed: bool = False,
    keep: Path | None = None,
) -> list[dict]:
    """R's fits of ``datasets``, one per dataset and test, by 0-based column position.

    The empty set is accepted when its own p-value exceeds the level, which ``ICP()`` leaves out
    of its accepted sets under a size cap. With ``fixed``, each fit also holds the tested sets'
    p-values and each accepted set's coefficients and standard errors; ``keep`` keeps the draws
    and R's files there.
    """
    work = Path(tempfile.mkdtemp(prefix="icp-r-")) if keep is None else keep
    work.mkdir(parents=True, exist_ok=True)
    others = [j for j in range(datasets[0][0].shape[1]) if j != target]
    for i, (X, env) in enumerate(datasets):
        header = [f"x{j}" for j in others] + ["y", "env"]
        table = np.column_stack([X[:, others], X[:, target], env])
        np.savetxt(work / f"d{i}.csv", table, delimiter=",", header=",".join(header), comments="")
    largest = "all" if max_parents is None else str(max_parents)
    names = ",".join(TESTS[t] for t in tests)
    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    command = ["Rscript", "--vanilla", str(WORKER), str(work), str(alpha), largest, names]
    subprocess.run(
        [*command, "1" if fixed else "0"], check=True, capture_output=True, env=environment
    )
    fits = []
    with (work / "fits.tsv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            test = next(t for t, r in TESTS.items() if r == row["test"])
            accepted = _sets(row["accepted"], others)
            empty_p = float(row["empty_p"])
            if max_parents is not None and empty_p > alpha:
                accepted = [[], *accepted]
            p_values = [None] * (len(others) + 1)
            for k, v in zip(others, _floats(row["pvalues"]), strict=True):
                p_values[k] = v
            reject = row["model_reject"] == "TRUE"
            fit = {
                "draw": int(row["draw"]),
                "test": test,
                "seconds": float(row["seconds"]),
                "accepted": accepted,
                "p_values": p_values,
                "parents": [] if reject else [k for k in others if p_values[k] <= alpha],
                "best": float(row["best"]),
                "model_rejected": reject,
                "upper": _floats(row["upper"]),
                "lower": _floats(row["lower"]),
            }
            if fixed:
                fit["sets"] = _set_table(work / f"sets_{row['draw']}.tsv", others)
                fit["coef"] = _coef_table(work / f"coef_{row['draw']}_{row['test']}.tsv", others)
            fits.append(fit)
    if keep is None:
        shutil.rmtree(work)
    return fits


def _set_table(path: Path, others: list[int]) -> list[dict]:
    with path.open() as handle:
        return [
            {
                "set": _sets(row["set"], others)[0],
                **{k: float(row[k]) for k in ("normal", "exact", "chow_package", "chow_paper")},
            }
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def _coef_table(path: Path, others: list[int]) -> list[dict]:
    if not path.exists() or not path.read_text().strip():
        return []
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [
        {
            "set": [others[int(k) - 1] for k in row["set"].split(",")],
            "coef": _floats(row["coef"]),
            "se": _floats(row["se"]),
        }
        for row in rows
    ]


# --- the campaign ----------------------------------------------------------------------------------


def run_scenario(task: tuple[str, int, str, Cell]) -> str:
    """Fit one scenario's datasets, Andrey's and R's, and write its task file."""
    name, index, out, cell = task
    sem = scenario(name, index, cell)
    truth = true_parents(sem)
    paired = r_scenario(cell, index)
    record = {
        "cell": name,
        "claim": cell.claim,
        "scenario": index,
        "nodes": int(sem.B.shape[0] - (sem.hidden is not None)),
        "rows": list(sem.rows),
        "environments": len(sem.rows),
        "target": sem.target,
        "truth": truth,
        "target_intervened": any(sem.target in c for c in sem.changes),
        "fits": [],
    }
    repeats = 3 if cell.kind == "speed" else 1
    r_data = []
    for i in range(cell.datasets):
        X, env = sample(sem, [658, CELL_IDS[name], index, i])
        if paired and i < cell.r_datasets:
            r_data.append((X, env))
        for test in ("approximate", "exact"):
            if repeats > 1:
                andrey_fit(X, env, sem.target, test, cell.max_parents)  # warm-up
            fits = [andrey_fit(X, env, sem.target, test, cell.max_parents) for _ in range(repeats)]
            fit = fits[0] | {"seconds": statistics.median(f["seconds"] for f in fits)}
            if not (paired and i < cell.r_datasets):
                fit = {k: fit[k] for k in ("parents", "best", "model_rejected", "seconds")}
            record["fits"].append({"solution": "andrey", "dataset": i, "test": test, **fit})
    if r_data:
        for fit in r_fits(r_data, sem.target, cell.r_tests, cell.max_parents):
            fit = {k: v for k, v in fit.items() if k not in ("upper", "lower")}
            record["fits"].append({"solution": "R", "dataset": fit.pop("draw"), **fit})
    path = Path(out) / "tasks" / name / f"{index:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_text(json.dumps(record))
    partial.replace(path)
    return f"{name}/{index:03d}"


def campaign(out: Path, cells: list[str], workers: int, resume: bool, smoke: bool) -> None:
    specs = {name: CELLS[name] for name in cells}
    if smoke:  # one scenario of each cell, with a few datasets
        specs = {
            name: replace(
                spec,
                scenarios=1,
                datasets=min(spec.datasets, 3),
                r_scenarios=min(spec.r_scenarios, 1),
                r_datasets=min(spec.r_datasets, 2),
            )
            for name, spec in specs.items()
        }
    meta = {
        "driver": Path(__file__).name,
        "alpha": ALPHA,
        "cells": {name: spec.__dict__ for name, spec in specs.items()},
        "machine_id": socket.gethostname(),
        "versions": _versions(),
    }
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "run_meta.json"
    if meta_path.exists() and not resume:
        raise SystemExit(f"{out} already holds a run; pass --resume to continue it")
    if resume and meta_path.exists():
        before = json.loads(meta_path.read_text())
        if before["cells"] != json.loads(json.dumps(meta["cells"])):
            raise SystemExit(f"--resume: {meta_path} lists other cells; pass the first run's")
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    # The longest scenarios start first, so the last ones to finish are short.
    first = ("speed-15", "D", "speed-12", "speed-12-large", "A")
    ordered = sorted(specs.items(), key=lambda item: (first + (item[0],)).index(item[0]))
    tasks = [
        (name, index, str(out), spec)
        for name, spec in ordered
        for index in range(spec.scenarios)
        if not (out / "tasks" / name / f"{index:03d}.json").exists()
    ]
    _log(f"=== ICP, Andrey and R -> {out}: {len(tasks)} scenarios to fit ===")
    start = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for done, label in enumerate(pool.imap_unordered(run_scenario, tasks), start=1):
            _log(f"[{done}/{len(tasks)}] {label} | elapsed {time.perf_counter() - start:.0f}s")


def _versions() -> dict:
    import andrey

    r = subprocess.run(
        [
            "Rscript",
            "-e",
            'cat(R.version.string, as.character(packageVersion("InvariantCausalPrediction")))',
        ],
        capture_output=True,
        text=True,
    )
    return {"andrey": andrey.__version__, "numpy": np.__version__, "R": r.stdout.strip()}


# --- the summary ----------------------------------------------------------------------------------


def _interval(hits: int, n: int) -> list[float]:
    """The two-sided 95% Clopper-Pearson interval of a share."""
    low = stats.beta.ppf(0.025, hits, n - hits + 1) if hits else 0.0
    high = stats.beta.ppf(0.975, hits + 1, n - hits) if hits < n else 1.0
    return [float(low), float(high)]


def summarize(out: Path) -> dict:
    """Each cell's accuracy, agreement with R, and fit times, from the task files."""
    summary: dict = {}
    for folder in sorted((out / "tasks").iterdir()):
        records = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
        cell: dict = {"claim": records[0]["claim"], "scenarios": len(records)}
        for test in ("approximate", "exact"):
            row = _accuracy(records, "andrey", test)
            if row:
                cell[f"andrey {test}"] = row
            row = _accuracy(records, "R", test)
            if row:
                cell[f"R {test}"] = row
                cell[f"agreement {test}"] = _agreement(records, test)
        summary[folder.name] = cell
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _accuracy(records: list[dict], solution: str, test: str) -> dict:
    false, success, found, rejected, seconds, n = 0, 0, [], 0, [], 0
    worst = []
    for record in records:
        truth = set(record["truth"])
        fits = [f for f in record["fits"] if f["solution"] == solution and f["test"] == test]
        wrong = sum(not set(f["parents"]) <= truth for f in fits)
        if fits:
            worst.append((wrong, len(fits), record["scenario"]))
        for fit in fits:
            parents = set(fit["parents"])
            false += not parents <= truth
            success += parents == truth
            rejected += fit["model_rejected"]
            seconds.append(fit["seconds"])
            if truth:
                found.append(len(parents & truth) / len(truth))
        n += len(fits)
    if not n:
        return {}
    # A scenario's FWER is above alpha with a one-sided binomial p-value below 0.05 / scenarios.
    above = [s for w, m, s in worst if stats.binom.sf(w - 1, m, ALPHA) < 0.05 / len(worst)]
    top = max(worst, key=lambda t: t[0] / t[1])
    return {
        "fits": n,
        "fwer": false / n,
        "fwer_interval": _interval(false, n),
        "largest_scenario_fwer": {"scenario": top[2], "fwer": top[0] / top[1], "fits": top[1]},
        "scenarios_above_alpha": above,
        "success": success / n,
        "share_found": statistics.fmean(found) if found else None,
        "model_rejected": rejected / n,
        "median_seconds": statistics.median(seconds),
    }


def _agreement(records: list[dict], test: str) -> dict:
    same_sets = same_parents = n = 0
    ratios, differing = [], []
    for record in records:
        r = {f["dataset"]: f for f in record["fits"] if f["solution"] == "R" and f["test"] == test}
        for fit in record["fits"]:
            other = r.get(fit["dataset"])
            if fit["solution"] != "andrey" or fit["test"] != test or other is None:
                continue
            n += 1
            sets = sorted(map(tuple, fit["accepted"])) == sorted(map(tuple, other["accepted"]))
            same_sets += sets
            same_parents += fit["parents"] == other["parents"]
            ratios.append(other["seconds"] / fit["seconds"])
            if not sets:
                differing.append([record["scenario"], fit["dataset"]])
    return {
        "paired_fits": n,
        "same_accepted_sets": same_sets,
        "same_parents": same_parents,
        "differing": differing,
        "median_time_ratio_r_over_andrey": statistics.median(ratios) if ratios else None,
    }


# --- tracing the differences ----------------------------------------------------------------------


def set_p_values(X, env, target, test, max_parents, rules) -> dict[tuple[int, ...], float]:
    """Every tested set's p-value from regressions on the rows, under R's rules or Andrey's.

    ``rules`` is ``"package"``, R's: the approximate test's correction is ``K - 1``, and the exact
    test's residual variance ``var(r) n_out / (n_out - p)``; or ``"paper"``, Andrey's: ``K``, 1
    with two environments, and ``RSS / (n_out - p)``. Chow's quadratic form is
    ``RSS_all - RSS_out``.
    """
    others = [j for j in range(X.shape[1]) if j != target]
    y, n = X[:, target], len(X)
    labels = np.unique(env)
    largest = len(others) if max_parents is None else min(max_parents, len(others))
    out = {}
    for size in range(largest + 1):
        for S in itertools.combinations(others, size):
            design = np.column_stack([np.ones(n), X[:, list(S)]])
            k = design.shape[1]
            coef = np.linalg.lstsq(design, y, rcond=None)[0]
            r = y - design @ coef
            p_env = []
            for e in labels:
                inside = env == e
                if test == "approximate":
                    a, b = r[inside], r[~inside]
                    p_mean = stats.ttest_ind(a, b, equal_var=False).pvalue
                    f = stats.f(len(a) - 1, len(b) - 1)
                    ratio = a.var(ddof=1) / b.var(ddof=1)
                    p_env.append(2 * min(p_mean, 2 * min(f.cdf(ratio), f.sf(ratio))))
                    continue
                n_in, n_out = int(inside.sum()), int((~inside).sum())
                fit = np.linalg.lstsq(design[~inside], y[~inside], rcond=None)[0]
                r_out = y[~inside] - design[~inside] @ fit
                rss_out = r_out @ r_out
                if rules == "package":
                    variance = r_out.var(ddof=1) * n_out / (n_out - k)
                else:
                    variance = rss_out / (n_out - k)
                statistic = max(r @ r - rss_out, 0.0) / (n_in * variance)
                p_env.append(stats.f.sf(statistic, n_in, n_out - k))
            K = len(labels)
            if test == "approximate":
                factor = K - 1 if rules == "package" else (1 if K == 2 else K)
            else:
                factor = K
            out[S] = min(1.0, factor * min(p_env))
    return out


def trace_fit(task: tuple) -> dict:
    """Whether R's rules give R's accepted sets and Andrey's rules Andrey's, for one paired fit."""
    name, index, dataset, test, mine, theirs = task
    cell = CELLS[name]
    sem = scenario(name, index)
    X, env = sample(sem, [658, CELL_IDS[name], index, dataset])
    found = {}
    for rules, accepted in (("package", theirs), ("paper", mine)):
        p = set_p_values(X, env, sem.target, test, cell.max_parents, rules)
        found[rules] = (p, sorted(S for S, v in p.items() if v > ALPHA) == sorted(accepted))
    straddle = [
        list(S)
        for S, v in found["paper"][0].items()
        if (v > ALPHA) != (found["package"][0][S] > ALPHA)
    ]
    cause = "the exact test's residual variance" if test == "exact" else "the correction"
    explained = found["package"][1] and found["paper"][1]
    return {
        "cell": name,
        "scenario": index,
        "dataset": dataset,
        "test": test,
        "environments": len(sem.rows),
        "explained": explained,
        "cause": cause if explained else None,
        "sets_straddling_alpha": straddle,
    }


def trace(out: Path, workers: int) -> dict:
    """Trace every paired fit whose accepted sets differ to its cause; writes ``trace.json``."""
    tasks = []
    for folder in sorted((out / "tasks").iterdir()):
        for path in sorted(folder.glob("*.json")):
            record = json.loads(path.read_text())
            r = {(f["dataset"], f["test"]): f for f in record["fits"] if f["solution"] == "R"}
            for fit in record["fits"]:
                other = r.get((fit["dataset"], fit["test"]))
                if fit["solution"] != "andrey" or other is None:
                    continue
                mine = sorted(tuple(s) for s in fit["accepted"])
                theirs = sorted(tuple(s) for s in other["accepted"])
                if mine != theirs:
                    tasks.append(
                        (folder.name, record["scenario"], fit["dataset"], fit["test"], mine, theirs)
                    )
    _log(f"=== tracing {len(tasks)} paired fits whose accepted sets differ ===")
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        rows = list(pool.imap_unordered(trace_fit, tasks))
    unexplained = [row for row in rows if not row["explained"]]
    result = {"traced": len(rows), "unexplained": unexplained, "fits": rows}
    (out / "trace.json").write_text(json.dumps(result, indent=2))
    return result


# --- the fixed cases --------------------------------------------------------------------------------


def _random_dag(rng, d, *, target_without_parents=False):
    """A random DAG over ``d`` nodes in a random order, edge probability ``2 / (d - 1)``, weights of
    magnitude in ``[0.3, 1]``; the target is the node with the most parents, the last in the order
    among ties, or the first node of the order."""
    order = rng.permutation(d)
    B = np.zeros((d, d))
    for a in range(d):
        for b in range(a + 1, d):
            if rng.random() < 2 / (d - 1):
                B[order[b], order[a]] = _weights(rng, 1, 0.3, 1.0)[0]
    most = max(range(d), key=lambda k: (np.count_nonzero(B[order[k]]), k))
    target = int(order[0] if target_without_parents else order[most])
    return B, tuple(int(j) for j in order), target


def _fixed_sem(case: str) -> tuple[SEM, int | None, float]:
    rng = np.random.default_rng([658, int(case[1:])])
    ones = np.ones
    if case in ("F1", "F7", "F9"):  # x2 -> x1 -> y -> x3; y is column 2
        B = np.zeros((4, 4))
        B[0, 1], B[2, 0], B[3, 2] = 0.8, 0.7, -0.6
        env2 = {1: ("do", 1.5, 1.0), 0: ("scale", 2.0, 2.0)}
        if case == "F7":
            env2 = {2: ("scale", 2.5, 2.5)}
        sem = SEM(B, (1, 0, 2, 3), ones(4), (300, 300), ({}, env2), (B, B), 2)
        return sem, None, 0.01 if case == "F9" else ALPHA
    if case in ("F2", "F8"):
        B, order, target = _random_dag(rng, 7)
        changes = {j: ("scale", 1.5, 3.0) for j in range(7) if j != target}
        sem = SEM(B, order, ones(7), (300, 300), ({}, changes), (B, B), target)
        return sem, 2 if case == "F8" else None, ALPHA
    if case == "F3":
        B, order, target = _random_dag(rng, 5)
        changes = {j: ("scale", 1.5, 3.0) for j in range(5) if j != target}
        return SEM(B, order, ones(5), (40, 25), ({}, changes), (B, B), target), None, ALPHA
    if case == "F4":
        B, order, target = _random_dag(rng, 7, target_without_parents=True)
        scale = {j: ("scale", 2.0, 3.0) for j in range(7) if j != target}
        shift = {j: ("do", 1.0, 1.0) for j in range(7) if j != target and j % 2}
        sem = SEM(B, order, ones(7), (200, 150, 250), ({}, scale, shift), (B, B, B), target)
        return sem, None, ALPHA
    if case == "F5":
        B, order, target = _random_dag(rng, 6)
        hit = rng.choice([j for j in range(6) if j != target], 3, replace=False)
        changes = ({}, *({int(j): ("do", 2.0, 1.0)} for j in hit))
        return SEM(B, order, ones(6), (200,) * 4, changes, (B,) * 4, target), None, ALPHA
    if case == "F6":  # x1 and x2 copy a hidden z, the target's parent, up to small noise
        B = np.zeros((5, 5))
        B[0, 4], B[1, 4], B[3, 4] = 1.0, 1.0, 1.0
        variances = np.array([0.0025, 0.0025, 1.0, 1.0, 1.0])
        sem = SEM(
            B,
            (4, 0, 1, 2, 3),
            variances,
            (300, 300),
            ({}, {4: ("scale", 3.0, 3.0)}),
            (B, B),
            3,
            hidden=4,
        )
        return sem, None, ALPHA
    raise KeyError(case)


FIXED = {
    "F1": "chain with a child, two environments",
    "F2": "random DAG of 7 nodes, noise interventions on every other variable",
    "F3": "random DAG of 5 nodes, 40 and 25 rows",
    "F4": "random DAG of 7 nodes, a target without parents, three environments",
    "F5": "do-interventions on 3 of 5 predictors, four environments",
    "F6": "two near-copies of the target's parent",
    "F7": "a noise intervention on the target (out of claim)",
    "F8": "F2 with max_parents=2",
    "F9": "F1 at alpha=0.01",
}


def expected(sets: list[dict], test: str, k: int, alpha: float, others: list[int]) -> dict:
    """Andrey's outputs implied by R's per-set p-values under Andrey's rules.

    The approximate test's p-value is R's times ``K / (K - 1)`` for three or more environments,
    the paper's Bonferroni factor over R's, and the exact test's is Chow's with the OLS residual
    variance (both capped at 1).
    """
    if test == "approximate":
        factor = 1.0 if k == 2 else k / (k - 1)
        p = [1.0 if s["normal"] >= 1.0 else min(1.0, s["normal"] * factor) for s in sets]
    else:
        p = [s["chow_paper"] for s in sets]
    accepted = [s["set"] for s, v in zip(sets, p, strict=True) if v > alpha]
    p_values = {
        j: max(v for s, v in zip(sets, p, strict=True) if j not in s["set"]) for j in others
    }
    if not accepted:
        p_values = dict.fromkeys(others, 1.0)
    parents = sorted(set(others).intersection(*map(set, accepted))) if accepted else []
    nearest = min(abs(v - alpha) for v in p)
    return {
        "accepted": accepted,
        "parents": parents,
        "p_values": p_values,
        "best": max(p),
        "nearest": nearest,
        "per_set": p,
    }


def _union(coef: list[dict], accepted: list[list[int]], others, quantile) -> dict:
    """The per-variable union of the accepted sets' intervals, with 0 for a set without it."""
    low = dict.fromkeys(others, np.inf)
    high = dict.fromkeys(others, -np.inf)
    by_set = {tuple(c["set"]): c for c in coef}
    for s in accepted:
        for j in others:
            if j in s:
                c = by_set[tuple(s)]
                at = c["set"].index(j)
                half = quantile(len(s)) * c["se"][at]
                low[j], high[j] = (
                    min(low[j], c["coef"][at] - half),
                    max(high[j], c["coef"][at] + half),
                )
            else:
                low[j], high[j] = min(low[j], 0.0), max(high[j], 0.0)
    return {j: (low[j], high[j]) for j in others}


def _close(a, b, rtol=1e-9, atol=1e-14) -> bool:
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=rtol, atol=atol))


def compare_fixed(out: Path) -> list[dict]:
    """Run F1-F9 in both packages and check every agreement 2.4 of ``comparisons/icp.md`` lists."""
    rows = []
    for case, label in FIXED.items():
        sem, max_parents, alpha = _fixed_sem(case)
        X, env = sample(sem, [658, int(case[1:]), 0])
        k = len(sem.rows)
        others = [j for j in range(X.shape[1]) if j != sem.target]
        both = ("approximate", "exact")
        fits = r_fits(
            [(X, env)], sem.target, both, max_parents, alpha=alpha, fixed=True, keep=out / case
        )
        r = {f["test"]: f for f in fits}
        for test in both:
            a = andrey_fit(X, env, sem.target, test, max_parents, alpha)
            rows.append(_check(case, label, test, k, alpha, a, r[test], others, len(X)))
    (out / "fixed.json").write_text(json.dumps(rows, indent=2))
    return rows


def _check(case, label, test, k, alpha, a, r, others, n) -> dict:
    sets = r["sets"]
    package = [s["normal" if test == "approximate" else "exact"] for s in sets]
    # The harness reads R correctly: ICP()'s accepted sets are the getpval sets above the level,
    # and the package-variance Chow p-values are getpval's exact ones.
    read = sorted(map(tuple, r["accepted"])) == sorted(
        tuple(s["set"]) for s, v in zip(sets, package, strict=True) if v > alpha
    )
    chow = _close([s["chow_package"] for s in sets], [s["exact"] for s in sets], rtol=1e-10)
    want = expected(sets, test, k, alpha, others)
    agree_sets = sorted(map(tuple, a["accepted"])) == sorted(map(tuple, want["accepted"]))
    agree_p = _close([a["p_values"][j] for j in others], [want["p_values"][j] for j in others])
    agree_best = _close(a["best"], want["best"])
    # Intervals: R's union, rebuilt with R's quantile, is R's ConfInt; Andrey's, rebuilt from R's
    # coefficients with the paper's quantile, is Andrey's, where the two accept the same sets.
    ci_read = ci_agree = None
    if not r["model_rejected"] and r["upper"]:
        z = stats.norm.ppf(1 - alpha / 4)
        theirs = _union(r["coef"], r["accepted"], others, lambda s: z)
        ci_read = _close(
            [theirs[j] for j in others],
            [(lo, up) for lo, up in zip(r["lower"], r["upper"], strict=True)],
        )
    if a["intervals"] is not None and sorted(map(tuple, a["accepted"])) == sorted(
        map(tuple, r["accepted"])
    ):
        mine = _union(
            r["coef"], a["accepted"], others, lambda s: stats.t.ppf(1 - alpha / (2 * s), n - s - 1)
        )
        ci_agree = _close([mine[j] for j in others], [a["intervals"][j] for j in others])
    raw_sets = sorted(map(tuple, a["accepted"])) == sorted(map(tuple, r["accepted"]))
    straddle = [
        s["set"]
        for s, mine, theirs in zip(sets, want["per_set"], package, strict=True)
        if (mine > alpha) != (theirs > alpha)
    ]
    return {
        "case": case,
        "label": label,
        "test": test,
        "environments": k,
        "alpha": alpha,
        "sets_tested": len(sets),
        "andrey_accepted": len(a["accepted"]),
        "r_accepted": len(r["accepted"]),
        "andrey_parents": a["parents"],
        "r_parents": r["parents"],
        "harness_reads_r": read and (chow if test == "exact" else True),
        "agrees_with_expected": agree_sets
        and agree_p
        and agree_best
        and a["parents"] == want["parents"],
        "intervals_read": ci_read,
        "intervals_agree": ci_agree,
        "same_as_r": raw_sets and a["parents"] == r["parents"],
        "sets_straddling_alpha": straddle,
        "nearest_p_to_alpha": want["nearest"],
    }


# --- main -----------------------------------------------------------------------------------------


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ICP, Andrey against R's InvariantCausalPrediction")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument("--fixed", action="store_true", help="run the fixed cases F1-F9 only")
    p.add_argument("--summary", action="store_true", help="fit nothing: write summary.json")
    p.add_argument(
        "--trace", action="store_true", help="trace the paired fits that differ: trace.json"
    )
    p.add_argument("--cells", default=",".join(CELLS), help="comma-separated cells (default: all)")
    p.add_argument("--workers", type=int, default=4, help="worker processes (default: 4)")
    p.add_argument("--resume", action="store_true", help="skip the scenarios already on disk")
    p.add_argument("--smoke", action="store_true", help="one scenario of each cell, few datasets")
    args = p.parse_args(argv)
    out = Path(args.out)
    if args.fixed:
        out.mkdir(parents=True, exist_ok=True)
        rows = compare_fixed(out)
        _log(
            f"{'case':4s} {'test':11s} {'K':>2s} {'sets':>4s} {'acc A/R':>8s} "
            f"{'expected':>8s} {'CI':>5s} {'same as R':>9s}  differing sets"
        )
        for row in rows:
            _log(
                f"{row['case']:4s} {row['test']:11s} {row['environments']:2d} "
                f"{row['sets_tested']:4d} {row['andrey_accepted']:3d}/{row['r_accepted']:<4d} "
                f"{str(row['agrees_with_expected'] and row['harness_reads_r']):>8s} "
                f"{str(row['intervals_agree']):>5s} {str(row['same_as_r']):>9s}  "
                f"{row['sets_straddling_alpha']}"
            )
        return 0
    if args.summary:
        _log(json.dumps(summarize(out), indent=2))
        return 0
    if args.trace:
        result = trace(out, args.workers)
        _log(f"traced {result['traced']}, unexplained {len(result['unexplained'])}")
        return 0
    cells = args.cells.split(",")
    unknown = set(cells) - set(CELLS)
    if unknown:
        p.error(f"unknown cells {sorted(unknown)}; the cells are {list(CELLS)}")
    campaign(out, cells, args.workers, args.resume, args.smoke)
    stamp = datetime.now(timezone.utc).isoformat()
    _log(f"finished {stamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
