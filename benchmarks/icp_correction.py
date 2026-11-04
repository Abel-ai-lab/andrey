#!/usr/bin/env python
"""Stress-test the approximate ICP test's correction: the factor K against R's K - 1.

Andrey's ``"icp_approximate"`` test compares each of the ``K`` environments' residuals with the
other rows' and multiplies the smallest p-value by ``K`` (by 1 when ``K = 2``); R's
InvariantCausalPrediction multiplies it by ``K - 1``. The ICP guarantee, that the estimated parents
are a subset of the true parents with probability at least ``1 - alpha``, rests on the test
rejecting the true parent set ``S*`` with probability at most ``alpha``, so the driver measures,
for each factor:

- the share of datasets whose ``S*`` the test rejects, ``p <= alpha``;
- the FWER of the full ICP fit: the share of fits whose parents are not a subset of ``S*``.

:class:`FactorInvariance` is Andrey's approximate test with the factor as a setting: its one-vs-rest
p-values are computed as in :class:`~andrey.core.invariance.ApproximateInvariance`, and at factor
``K`` it returns the same p-values bit for bit (:func:`checks`). The driver registers it under its
own names in :data:`~andrey.core.invariance.INVARIANCE_TESTS` and in the ICP engine's ``TESTS``, so
the engine's own set search runs with either factor. One fit at ``alpha = 0.05`` gives both levels:
a fit's per-column p-values and its best p-value do not depend on ``alpha``, and a column is a
parent at level ``a`` exactly when the best p-value is above ``a`` and its own is at most ``a``.

Cells (:data:`CELLS`): ``K`` in 3, 4, 5, 8, 10 environments, ``n`` rows per environment in 30, 100,
500, with equal sizes or with the observational environment at ``5 n`` rows; and the hard cell.
Each dataset comes from ``andrey.data`` (``SCM.sample`` with ``envs=``) on a fixed linear Gaussian
SEM whose weights and noise scales are pinned. In the grid (:data:`GRID`), environment 1 is
observational and each other environment intervenes on one non-target variable drawn uniformly,
with a do-intervention ``hard(2.0, sd=1.0)`` or a mean shift ``soft(shift=2.0)``, each with
probability 1/2, drawn anew per dataset. The hard cell (:data:`HARD`) makes a set with a child of
the target nearly invariant, so a rejection of ``S*`` turns into a false parent (``icp.md``).

    $ANDREY_BENCH_PYTHON benchmarks/icp_correction.py --seed 0 --out outputs/icp-correction
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402 - BLAS threads are set before NumPy loads
import itertools  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from icp_cpu import _interval, _pinned_specs  # noqa: E402
from scipy import stats  # noqa: E402

import andrey  # noqa: E402
from andrey.constraint import icp as icp_engine  # noqa: E402
from andrey.core.invariance import (  # noqa: E402
    INVARIANCE_TESTS,
    ApproximateInvariance,
    make_invariance_test,
)

ALPHAS = (0.05, 0.10)
FACTORS = ("K", "K-1")
DATASETS = 2000
CHUNK = 100  # datasets per worker task
CHECKED = 4  # datasets per K that the checks read

# --- the test with the factor switched --------------------------------------------------------------


class FactorInvariance(ApproximateInvariance):
    """Andrey's approximate test with the smallest p-value multiplied by ``K - offset``.

    The one-vs-rest p-values are those of ``ApproximateInvariance._p_values``, line for line; with
    two environments the factor stays 1.
    """

    name = "icp_approximate_factor_k"
    offset = 0

    def _p_values(self, grams, rows, pooled, beta, rss) -> np.ndarray:
        sizes, y = self._sizes, grams.shape[1] - 1
        A = grams[:, rows[:, :, None], rows[:, None, :]]
        total = grams[:, 0, y][:, None] - (grams[:, 0, rows] * beta).sum(axis=2)
        squares = (
            grams[:, y, y][:, None]
            - 2.0 * (grams[:, rows, y] * beta).sum(axis=2)
            + np.einsum("mi,kmij,mj->km", beta, A, beta)
        )
        n_in, n_out = sizes[:, None], sizes.sum() - sizes[:, None]
        mean_in = total / n_in
        mean_out = (total.sum(axis=0) - total) / n_out
        var_in = (squares - n_in * mean_in**2) / (n_in - 1.0)
        var_out = (squares.sum(axis=0) - squares - n_out * mean_out**2) / (n_out - 1.0)
        se_in, se_out = var_in / n_in, var_out / n_out
        t = (mean_in - mean_out) / np.sqrt(se_in + se_out)
        df = (se_in + se_out) ** 2 / (se_in**2 / (n_in - 1.0) + se_out**2 / (n_out - 1.0))
        p_mean = 2.0 * stats.t.sf(np.abs(t), df)
        ratio = var_in / var_out
        below = stats.f.cdf(ratio, n_in - 1.0, n_out - 1.0)
        above = stats.f.sf(ratio, n_in - 1.0, n_out - 1.0)
        p_variance = 2.0 * np.minimum(below, above)
        p_group = 2.0 * np.minimum(p_mean, p_variance)
        factor = 1.0 if len(sizes) == 2 else float(len(sizes) - self.offset)
        return np.minimum(1.0, factor * p_group.min(axis=0))


class KMinusOneInvariance(FactorInvariance):
    """The approximate test with R's factor, ``K - 1``."""

    name = "icp_approximate_k_minus_1"
    offset = 1


# The registry entries, and the engine's names for them: the engine builds its test by name through
# make_invariance_test, so its set search runs unchanged with either factor.
INVARIANCE_TESTS[FactorInvariance.name] = FactorInvariance
INVARIANCE_TESTS[KMinusOneInvariance.name] = KMinusOneInvariance
icp_engine.TESTS["approximate_factor_k"] = FactorInvariance.name
icp_engine.TESTS["approximate_k_minus_1"] = KMinusOneInvariance.name
ENGINE_TEST = {"K": "approximate", "K-1": "approximate_k_minus_1"}
SET_TEST = {"K": "icp_approximate", "K-1": KMinusOneInvariance.name}

# --- the data -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Model:
    """A linear Gaussian SEM over ``d`` columns: ``arcs[i] = (parent, child)`` with weight
    ``weights[i]``, noise standard deviations ``scales``, and the target with its parents."""

    arcs: tuple[tuple[int, int], ...]
    weights: tuple[float, ...]
    scales: tuple[float, ...]
    target: int
    parents: tuple[int, ...]


#: The grid's SEM: 0 -> 1, 2 -> 1, {0, 1} -> 3, 3 -> 4, {2, 4} -> 5, with target 3; 2 is a
#: non-descendant of the target outside its parents, 4 its child, and 5 a grandchild.
GRID = Model(
    arcs=((0, 1), (2, 1), (0, 3), (1, 3), (3, 4), (4, 5), (2, 5)),
    weights=(0.8, 0.6, 0.7, -0.9, 1.2, 0.8, -0.5),
    scales=(1.0,) * 6,
    target=3,
    parents=(0, 1),
)
#: The hard cell's SEM: 0 -> 1, {1, 2} -> 3, 3 -> 4, with target 3 and its child 4 a near copy of
#: it: ``x_4 = 2 x_3 + N(0, 0.5^2)``. Every column but the target's child is a non-descendant.
HARD = Model(
    arcs=((0, 1), (1, 3), (2, 3), (3, 4)),
    weights=(0.8, 1.0, -1.0, 2.0),
    scales=(1.0, 1.0, 1.0, 1.0, 0.5),
    target=3,
    parents=(1, 2),
)
#: The hard cell's shift of the target's child, in each interventional environment.
WEAK = 0.1


@dataclass(frozen=True)
class Cell:
    """``k`` environments of ``n`` rows each, or with the observational one at ``5 n`` rows."""

    k: int
    n: int
    sizes: str  # "equal" or "unequal"
    hard: bool = False

    @property
    def name(self) -> str:
        return "hard" if self.hard else f"K{self.k}-n{self.n}-{self.sizes}"

    @property
    def model(self) -> Model:
        return HARD if self.hard else GRID

    @property
    def rows(self) -> list[int]:
        first = 5 * self.n if self.sizes == "unequal" else self.n
        return [first] + [self.n] * (self.k - 1)


CELLS = [
    Cell(k, n, sizes)
    for sizes in ("equal", "unequal")
    for k in (3, 4, 5, 8, 10)
    for n in (30, 100, 500)
] + [Cell(3, 100, "equal", hard=True)]
CELL_IDS = {cell.name: index for index, cell in enumerate(CELLS)}

_SCMS: dict[Model, object] = {}


def _scm(model: Model):
    """The SCM of ``model``, with its weights and noise scales pinned, built once per process."""
    if model not in _SCMS:
        from andrey.data import SCM
        from andrey.data.graphs import FixedDAG

        linear, gaussian = _pinned_specs()
        d = len(model.scales)
        _SCMS[model] = SCM(
            graph=FixedDAG(
                nodes=tuple(str(j) for j in range(d)),
                arcs=tuple((str(a), str(b)) for a, b in model.arcs),
            ),
            functional=linear(pinned=model.weights),
            noise=gaussian(pinned=model.scales),
        )
    return _SCMS[model]


def environments(cell: Cell, rng: np.random.Generator) -> list[dict]:
    """Each environment's interventions: none in the first."""
    from andrey.data.interventions import hard, soft

    if cell.hard:
        # Strong changes of both parents, and a weak shift of the target's child.
        return [
            {},
            {1: hard(3.0, sd=1.0), 4: soft(shift=WEAK)},
            {2: soft(shift=3.0), 4: soft(shift=WEAK)},
        ]
    model = cell.model
    others = [j for j in range(len(model.scales)) if j != model.target]
    envs: list[dict] = [{}]
    for _ in range(cell.k - 1):
        j = int(rng.choice(others))
        envs.append({j: hard(2.0, sd=1.0) if rng.random() < 0.5 else soft(shift=2.0)})
    return envs


def draw(cell: Cell, seed: int, i: int) -> tuple[np.ndarray, np.ndarray]:
    """Dataset ``i`` of ``cell``: the data, in the SEM's column order, and each row's environment."""
    rng = np.random.default_rng([seed, CELL_IDS[cell.name], i])
    envs = environments(cell, rng)
    root = int(rng.integers(2**62))
    ds = _scm(cell.model).sample(n=cell.rows, seed=root, envs=envs, scale="raw", shuffle=False)
    return ds.data, ds.env


# --- fits --------------------------------------------------------------------------------------------


def engine_fit(X, env, target: int, factor: str, alpha: float = ALPHAS[0]):
    """ICP's engine with the approximate test at ``factor``."""
    k = int(env.max()) + 1
    return icp_engine.icp(
        X,
        env,
        target,
        alpha=alpha,
        invariance_test=ENGINE_TEST.get(factor, factor),
        max_parents=None,
        environments=list(range(k)),
    )


def parents_at(fit, level: float) -> frozenset[int]:
    """The parents of a fit made at a level at most ``level``, at the level ``level``."""
    if fit.best_p_value <= level:
        return frozenset()
    return frozenset(int(j) for j in np.flatnonzero(fit.p_values <= level))


def score(cell: Cell, seed: int, start: int, stop: int) -> dict:
    """Counts over datasets ``start..stop-1`` of ``cell``, per factor and level."""
    model = cell.model
    truth = frozenset(model.parents)
    keys = [f"{f}@{a}" for f in FACTORS for a in ALPHAS]
    counts = {
        key: {"reject_star": 0, "false": 0, "model_rejected": 0, "false_star_accepted": 0}
        for key in keys
    }
    # Each a count of datasets on which two routes to the same decision disagree.
    mismatch = {"star_rescaled": 0, "parents_rescaled": 0, "rerun_level": 0}
    for i in range(start, stop):
        X, env = draw(cell, seed, i)
        k = cell.k
        fits = {f: engine_fit(X, env, model.target, f) for f in FACTORS}
        p_star = {
            f: make_invariance_test(SET_TEST[f], X, env, indep_test="fisherz")(
                model.target, model.parents
            )
            for f in FACTORS
        }
        for f, a in itertools.product(FACTORS, ALPHAS):
            c = counts[f"{f}@{a}"]
            parents = parents_at(fits[f], a)
            rejected = p_star[f] <= a
            c["reject_star"] += rejected
            c["false"] += not parents <= truth
            c["model_rejected"] += fits[f].best_p_value <= a
            c["false_star_accepted"] += (not parents <= truth) and not rejected
        # The K - 1 decisions are the factor-K decisions at the level a K / (K - 1).
        for a in ALPHAS:
            wide = a * k / (k - 1)
            mismatch["star_rescaled"] += (p_star["K-1"] <= a) != (p_star["K"] <= wide)
            mismatch["parents_rescaled"] += parents_at(fits["K-1"], a) != parents_at(
                fits["K"], wide
            )
        if i < start + 2:  # a fit at the second level gives the parents read from the first
            for f in FACTORS:
                direct = engine_fit(X, env, model.target, f, alpha=ALPHAS[1])
                mismatch["rerun_level"] += frozenset(direct.parents) != parents_at(
                    fits[f], ALPHAS[1]
                )
    return {"cell": cell.name, "datasets": stop - start, "counts": counts, "mismatch": mismatch}


def _task(args) -> dict:
    return score(*args)


# --- checks ------------------------------------------------------------------------------------------


def _direct_p_value(X, env, target: int, condition: tuple[int, ...], factor: float) -> float:
    """The approximate test's p-value from the pooled residuals, with SciPy's two-sample tests."""
    Z = np.column_stack([np.ones(len(X)), X[:, list(condition)]])
    beta = np.linalg.lstsq(Z, X[:, target], rcond=None)[0]
    r = X[:, target] - Z @ beta
    smallest = 1.0
    for e in range(int(env.max()) + 1):
        inside, outside = r[env == e], r[env != e]
        p_mean = stats.ttest_ind(inside, outside, equal_var=False).pvalue
        ratio = inside.var(ddof=1) / outside.var(ddof=1)
        dfs = (len(inside) - 1, len(outside) - 1)
        p_variance = 2.0 * min(stats.f.cdf(ratio, *dfs), stats.f.sf(ratio, *dfs))
        smallest = min(smallest, 2.0 * min(p_mean, p_variance))
    return min(1.0, factor * smallest)


def checks(seed: int) -> dict:
    """That :class:`FactorInvariance` at factor ``K`` is Andrey's test, on ``CHECKED`` datasets of
    each ``K`` with 100 unequal rows: the same p-value for every set and the same ICP output, bit
    for bit, and SciPy's p-values from the pooled residuals within ``1e-9`` relative."""
    model = GRID
    d = len(model.scales)
    others = [j for j in range(d) if j != model.target]
    sets = [s for size in range(len(others) + 1) for s in itertools.combinations(others, size)]
    out = {"datasets": 0, "sets": 0, "same_p_values": 0, "same_icp": 0, "largest_scipy_rel": 0.0}
    for k in sorted({cell.k for cell in CELLS if not cell.hard}):
        cell = Cell(k, 100, "unequal")
        for i in range(CHECKED):
            X, env = draw(cell, seed, i)
            base = ApproximateInvariance(X, env)
            ours = FactorInvariance(X, env)
            for size in range(len(others) + 1):
                group = np.array(list(itertools.combinations(others, size)), dtype=np.intp)
                group = group.reshape(len(group), size)
                p_base = base.fit(model.target, group)[0]
                p_ours = ours.fit(model.target, group)[0]
                out["sets"] += len(group)
                out["same_p_values"] += int(np.sum(p_base == p_ours))
            plain = engine_fit(X, env, model.target, "approximate")
            switched = engine_fit(X, env, model.target, "approximate_factor_k")
            out["same_icp"] += (
                plain.parents == switched.parents
                and plain.accepted == switched.accepted
                and plain.best_p_value == switched.best_p_value
                and np.array_equal(plain.p_values, switched.p_values, equal_nan=True)
            )
            for s in sets:
                direct = _direct_p_value(X, env, model.target, s, float(k))
                ours_p = ours(model.target, s)
                rel = abs(direct - ours_p) / max(direct, 1e-300)
                out["largest_scipy_rel"] = max(out["largest_scipy_rel"], rel)
            out["datasets"] += 1
    return out


# --- the summary ----------------------------------------------------------------------------------


def summarize(rows: list[dict]) -> list[dict]:
    """Each cell's rates, with their 95% Clopper-Pearson intervals."""
    cells = []
    for cell in CELLS:
        mine = [row for row in rows if row["cell"] == cell.name]
        if not mine:
            continue
        n = sum(row["datasets"] for row in mine)
        mismatch = {key: sum(row["mismatch"][key] for row in mine) for key in mine[0]["mismatch"]}
        entries = []
        for f, a in itertools.product(FACTORS, ALPHAS):
            totals = {
                key: sum(row["counts"][f"{f}@{a}"][key] for row in mine)
                for key in mine[0]["counts"][f"{f}@{a}"]
            }
            entry = {"factor": f, "alpha": a, **totals}
            for key in ("reject_star", "false"):
                low, high = _interval(totals[key], n)
                entry[key + "_rate"] = totals[key] / n
                entry[key + "_ci"] = [low, high]
                entry[key + "_above_alpha"] = bool(low > a)
            entry["k_minus_1_bound"] = a * cell.k / (cell.k - 1) if f == "K-1" else None
            entries.append(entry)
        cells.append(
            {
                "cell": cell.name,
                "K": cell.k,
                "n": cell.n,
                "sizes": cell.sizes,
                "rows": cell.rows,
                "datasets": n,
                "mismatch": mismatch,
                "results": entries,
            }
        )
    return cells


def table(cells: list[dict]) -> str:
    """The Markdown table of the comparison page."""
    lines = [
        "| Cell | alpha | Reject S*, K | Reject S*, K - 1 | FWER, K | FWER, K - 1 |",
        "|---|---|---|---|---|---|",
    ]

    def show(entry, key):
        low, high = entry[key + "_ci"]
        mark = " (above)" if entry[key + "_above_alpha"] else ""
        return f"{entry[key + '_rate']:.4f} ({low:.4f}-{high:.4f}){mark}"

    for cell in cells:
        for a in ALPHAS:
            by = {e["factor"]: e for e in cell["results"] if e["alpha"] == a}
            lines.append(
                f"| {cell['cell']} | {a:.2f} | {show(by['K'], 'reject_star')} | "
                f"{show(by['K-1'], 'reject_star')} | {show(by['K'], 'false')} | "
                f"{show(by['K-1'], 'false')} |"
            )
    return "\n".join(lines)


def _machine() -> dict:
    model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"cpu": model or platform.processor(), "cpus": len(os.sched_getaffinity(0))}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument("--seed", type=int, default=0, help="the root seed (default: 0)")
    p.add_argument("--datasets", type=int, default=DATASETS, help="datasets per cell")
    p.add_argument("--cells", default=None, help="comma-separated cell names (default: all)")
    p.add_argument("--workers", type=int, default=4, help="worker processes (default: 4)")
    args = p.parse_args(argv)
    chosen = CELLS if args.cells is None else [c for c in CELLS if c.name in args.cells.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    checked = checks(args.seed)
    print(f"checks: {checked}", flush=True)
    tasks = [
        (cell, args.seed, lo, min(lo + CHUNK, args.datasets))
        for cell in chosen
        for lo in range(0, args.datasets, CHUNK)
    ]
    with multiprocessing.Pool(args.workers) as pool:
        rows = pool.map(_task, tasks, chunksize=1)
    cells = summarize(rows)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    ).stdout.strip()
    result = {
        "driver": Path(__file__).name,
        "seed": args.seed,
        "datasets": args.datasets,
        "alphas": ALPHAS,
        "weak_shift": WEAK,
        "versions": {
            "andrey": andrey.__version__,
            "commit": commit,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "machine": _machine(),
        "workers": args.workers,
        "seconds": time.perf_counter() - start,
        "checks": checked,
        "cells": cells,
    }
    path = out / "correction.json"
    path.write_text(json.dumps(result, indent=2))
    print(table(cells), flush=True)
    print(f"wrote {path} in {result['seconds']:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
