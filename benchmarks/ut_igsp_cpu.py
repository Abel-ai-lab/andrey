#!/usr/bin/env python
"""Benchmark UT-IGSP on CPU: Andrey against the authors' library, on interventional data.

The data follow Squires, Wang, and Uhler (2020, Section 5.1), drawn with ``andrey.data``
(:func:`dataset`): an Erdos-Renyi DAG on ``p`` variables with mean degree 1.5 (exactly
``round(1.5 p / 2)`` edges), weights uniform on ``[0.25, 1]`` in absolute value with a random sign,
standard normal noise in the generator's own units (``scale="raw"``), and its columns in an order
drawn from the seed. An observational environment and ``K`` settings of ``n`` rows each; setting
``k`` intervenes on one designed target, which the fit is told, and on ``l`` off-targets, which it
is not. The designed targets come from ``default_rng([seed, 654])`` and the off-targets of setting
``k`` from ``default_rng([seed, 654, l, k])`` among the other variables, so the cells of one block
share their DAGs and designed targets. :data:`KINDS` names the interventions.

:func:`grid` lists every cell: block A (seeds 1000-1019, Andrey alone, to choose its defaults), the
in-claim arms G1-G6 and the out-of-claim arms O1-O9 (seeds 0-19), each with its ``claim``. Per
dataset, each package runs in one child process that fits every algorithm of the package and every
draw of each (:mod:`andrey_bench.adapters.andrey_interventional`,
:mod:`andrey_bench.adapters.gml_interventional`), pinned to one core with one BLAS thread. A draw
over its per-draw cap is a ``timeout``, never scored; a killed child makes every draw it had not
recorded a timeout.

Scores (:func:`score`): each estimate is the interventional essential graph
(:func:`essential`) of the returned graph under the targets the method returned or was given, and
the truth that of the generating DAG under the true targets. ``shd`` counts the variable pairs
whose endpoint marks differ; ``target_error`` sums each setting's false and missed targets;
``target_floor`` is the ``target_error`` of the targets the fit was given; ``shd_true_targets`` is
the ``shd`` of the estimate's DAG re-read under the true targets, and ``in_true_imec`` whether it
is 0; ``empty_shd`` is the true graph's number of adjacent pairs.

``run`` writes ``records.jsonl`` (one line per draw), ``datasets.jsonl`` (one line per dataset:
the true DAG and targets), ``run_meta.json``, ``provenance.json``, and, once every dataset is
recorded, ``done.json``. ``summarize`` turns finished runs into summary entries that
``qa/reference_parity.py`` reads, and ``select`` applies block A's default-selection rule.

    benchmarks/start_utigsp.sh
    export ANDREY_BENCH_UTIGSP_PYTHON=$PWD/benchmarks/.venv-utigsp/bin/python
    $ANDREY_BENCH_PYTHON benchmarks/ut_igsp_cpu.py check-truth --out outputs/utigsp-truth
    $ANDREY_BENCH_PYTHON benchmarks/ut_igsp_cpu.py run --out outputs/utigsp-G1 --arm G1
    $ANDREY_BENCH_PYTHON benchmarks/ut_igsp_cpu.py summarize --runs outputs/utigsp-G1 \\
        --out outputs/utigsp-summary.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ANDREY_CHILD = HERE / "andrey_bench" / "adapters" / "andrey_interventional.py"
GML_CHILD = HERE / "andrey_bench" / "adapters" / "gml_interventional.py"
REFERENCE_PYTHON = "ANDREY_BENCH_UTIGSP_PYTHON"
RECORDS, DATASETS, DONE = "records.jsonl", "datasets.jsonl", "done.json"
SAMPLER = "andrey.data, scale='raw'"

#: The paper's level for both tests and the reference library's number of runs.
ALPHA, NRUNS = 1e-5, 5
#: Draws per dataset of a randomized method in claim and in block A; one elsewhere.
DRAWS = 5
#: First SCM seed of each block: A chooses Andrey's defaults, B compares, C replicates.
BLOCKS = {"A": 1000, "B": 0, "C": 2000}
#: Seconds a draw may take: Gaussian tests up to 50 variables, and the rest.
CAP_DRAW_S, CAP_DRAW_LONG_S = 1800.0, 3600.0
#: Bootstrap resamples of the paired interval.
RESAMPLES = 10_000

#: Each kind's intervention: ``(constructor, arguments)`` from ``andrey.data.interventions``.
KINDS: dict[str, tuple[str, dict]] = {
    "shift": ("soft", {"shift": 1.0}),
    "shift2": ("soft", {"shift": 2.0}),
    "perfect": ("hard", {"value": 1.0, "sd": 0.1**0.5}),
    "perfect_code": ("hard", {"value": 1.0, "sd": 0.1}),
    "variance": ("soft", {"noise_scale": 2.0}),
    "weight": ("soft", {"weight_scale": 0.5}),
    "weak0.25": ("soft", {"shift": 0.25}),
    "weak0.5": ("soft", {"shift": 0.5}),
    "discrete": ("hard", {}),
}

#: Andrey's matched configuration (J1) and the reference's.
J1_ANDREY = {"alpha": ALPHA, "n_starts": 5, "objective": "sum"}
J1_REFERENCE = {"alpha": ALPHA, "nruns": NRUNS}
#: Block A's step-1 candidates (design 4.4): alpha by objective at five starts.
STEP1 = [
    {"alpha": a, "n_starts": 5, "objective": o}
    for a in (1e-5, 1e-3, 1e-2)
    for o in ("sum", "lexicographic")
]


@dataclass(frozen=True)
class Cell:
    """One cell of the grid: a setting, its datasets, and whether it is in the method's claim."""

    arm: str
    kind: str
    off: int
    n: int
    p: int = 20
    degree: float = 1.5
    settings: int = 5
    n_obs: int | None = None
    functional: str = "linear"
    known: bool = True
    claim: str = "in"
    block: str = "B"
    datasets: int = 20

    @property
    def group(self) -> str:
        """The cell's name without its size: the summary entry its row belongs to."""
        name = f"{self.arm}_{self.kind}_l{self.off}"
        if self.degree != 1.5:
            name += f"_deg{self.degree:g}"
        if self.settings != 5:
            name += f"_K{self.settings}"
        if self.n_obs is not None:
            name += f"_obs{self.n_obs}"
        return name

    @property
    def name(self) -> str:
        return f"{self.group}_n{self.n}" + (f"_p{self.p}" if self.p != 20 else "")

    @property
    def seeds(self) -> list[int]:
        return list(range(BLOCKS[self.block], BLOCKS[self.block] + self.datasets))

    @property
    def cap_draw_s(self) -> float:
        return CAP_DRAW_LONG_S if self.p >= 100 or self.arm == "O6" else CAP_DRAW_S

    @property
    def draws(self) -> int:
        return DRAWS if self.claim == "in" or self.arm == "A" else 1


def grid() -> list[Cell]:
    """Every cell of the lane: block A, then the in-claim and out-of-claim arms."""
    cells = [
        Cell("A", kind, off, n, block="A")
        for kind in ("shift", "perfect")
        for off in (0, 3)
        for n in (1000, 3000, 5000)
    ]
    cells += [Cell("A", "shift", off, 300, block="A", claim="out") for off in (0, 3)]
    cells += [Cell("A", "shift", 3, 2000, p=50, block="A", claim="out", datasets=10)]
    for arm, kind in (("G1", "shift"), ("G2", "perfect")):
        cells += [Cell(arm, kind, off, n) for off in range(4) for n in range(1000, 5001, 1000)]
    cells += [Cell("G3", kind, off, 1000) for kind in ("shift2", "perfect_code") for off in (0, 3)]
    for arm, kind, known in (
        ("G4", "variance", True),
        ("G5", "shift", False),
        ("G6", "weight", True),
    ):
        cells += [Cell(arm, kind, off, n, known=known) for off in (0, 3) for n in (1000, 5000)]
    out = {"claim": "out"}
    cells += [Cell("O1", "shift", off, n, **out) for off in (0, 3) for n in (100, 300)]
    cells += [
        Cell("O2", kind, off, 1000, **out) for kind in ("weak0.25", "weak0.5") for off in (0, 3)
    ]
    cells += [
        Cell("O3", "perfect", 0, n, p=50, degree=5.0, settings=3, datasets=10, **out)
        for n in (500, 2000)
    ]
    cells += [Cell("O4", "shift", 3, 2000, p=p, datasets=10, **out) for p in (50, 100)]
    cells += [Cell("O5", "shift", off, 1000, functional="additive", **out) for off in (0, 3)]
    cells += [
        Cell("O6", "shift", off, n, p=10, functional="additive", **out)
        for off in (0, 3)
        for n in (200, 400)
    ]
    cells += [
        Cell("O7", "discrete", 0, n, p=10, functional="discrete", datasets=10, **out)
        for n in (1000, 5000)
    ]
    cells += [Cell("O8", "shift", off, 1000, settings=k, **out) for k in (1, 2) for off in (0, 3)]
    cells += [Cell("O9", "shift", off, 300, n_obs=2000, **out) for off in (0, 3)]
    return cells


def cell_named(name: str) -> Cell:
    for cell in grid():
        if cell.name == name:
            return cell
    raise KeyError(f"no cell named {name!r}")


# ------------------------------------------------------------------------------------------------
# Data and truth
# ------------------------------------------------------------------------------------------------


def _intervention(kind: str):
    from andrey.data.interventions import hard, soft

    maker, kwargs = KINDS[kind]
    if maker == "soft":
        return soft(**kwargs)
    kwargs = dict(kwargs)
    return hard(kwargs.pop("value", None), **kwargs)


def _scm(cell: Cell):
    import andrey.data as data

    functional = {
        "linear": data.functional.linear,
        "additive": data.functional.additive_noise,
    }[cell.functional](weight_range=(0.25, 1.0))
    return data.SCM(
        graph=data.graphs.erdos_renyi(d=cell.p, avg_degree=cell.degree),
        functional=functional,
        noise=data.noise.gaussian(),
    )


def has_parent(dag: np.ndarray) -> np.ndarray:
    """Whether each variable of a DAG's endpoint marks has a parent."""
    from andrey.core import ARROW

    return (dag == ARROW).any(axis=1)


def designed_and_off(cell: Cell, seed: int, eligible: np.ndarray) -> list[list[int]]:
    """Each setting's targets: its designed one first, then its off-targets."""
    if eligible.size < cell.settings:
        raise ValueError(
            f"{cell.name} seed {seed}: {eligible.size} eligible targets for {cell.settings} settings"
        )
    designed = np.random.default_rng([seed, 654]).choice(eligible, cell.settings, replace=False)
    targets = []
    for k, target in enumerate(designed):
        others = [int(v) for v in eligible if v != target]
        off = np.random.default_rng([seed, 654, cell.off, k]).choice(
            others, cell.off, replace=False
        )
        targets.append([int(target), *(int(v) for v in off)])
    return targets


def dataset(cell: Cell, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return ``(X, env, truth)`` for one dataset of ``cell``.

    ``truth`` holds the generating DAG's endpoint marks (``dag``), each setting's true targets
    (``true``), its designed target (``designed``), and the targets the fit is given (``given``:
    the designed ones, or none when the cell's targets are unknown).
    """
    rows = [cell.n_obs or cell.n] + [cell.n] * cell.settings
    spec = _intervention(cell.kind)
    if cell.functional == "discrete":
        from andrey.data import sample_discrete

        eligible = np.arange(cell.p)
        planned = designed_and_off(cell, seed, eligible)
        envs = [{}] + [{v: spec for v in t} for t in planned]
        ds = sample_discrete(d=cell.p, n=rows, seed=seed, envs=envs)
    else:
        scm = _scm(cell)
        eligible = np.arange(cell.p)
        if cell.kind == "weight":  # a weight change acts only on a variable with parents
            graph = scm.sample(n=2, seed=seed, scale="raw").graph.to_numpy()
            eligible = np.flatnonzero(has_parent(graph))
        planned = designed_and_off(cell, seed, eligible)
        envs = [{}] + [{v: spec for v in t} for t in planned]
        ds = scm.sample(n=rows, seed=seed, scale="raw", envs=envs)
    true = [sorted(int(v) for v in ds.targets[k]) for k in range(1, cell.settings + 1)]
    if true != [sorted(t) for t in planned]:
        raise RuntimeError(f"{cell.name} seed {seed}: the sampler moved the targets")
    designed = [[t[0]] for t in planned]
    truth = {
        "dag": np.asarray(ds.graph.to_numpy(), dtype=int).tolist(),
        "true": true,
        "designed": designed,
        "given": designed if cell.known else [[] for _ in designed],
    }
    return np.asarray(ds.data, dtype=np.float64), np.asarray(ds.env, dtype=np.int64), truth


def family(targets: list[list[int]], p: int) -> np.ndarray:
    """The 0/1 target mask of the observational setting and each setting of ``targets``."""
    mask = np.zeros((len(targets) + 1, p), dtype=bool)
    for row, variables in enumerate(targets, start=1):
        mask[row, list(variables)] = True
    return mask


def essential(dag: np.ndarray, targets: list[list[int]]) -> np.ndarray:
    """The interventional essential graph of ``dag`` under the observational setting and
    ``targets``, as endpoint marks."""
    from andrey.search.gies import cut_pairs, replace_unprotected

    dag = np.asarray(dag, dtype=np.int8)
    return replace_unprotected(dag.copy(), cut_pairs(family(targets, dag.shape[0])))


def dag_from_arcs(arcs: list, p: int) -> np.ndarray:
    from andrey.core import ARROW, TAIL

    dag = np.zeros((p, p), dtype=np.int8)
    for i, j in arcs:
        dag[i, j], dag[j, i] = TAIL, ARROW
    return dag


def pairs_differ(a: np.ndarray, b: np.ndarray) -> int:
    """Variable pairs whose endpoint marks differ: ``andrey.metrics.shd``'s structural count."""
    a, b = np.asarray(a), np.asarray(b)
    upper = np.triu_indices(a.shape[0], 1)
    return int(((a[upper] != b[upper]) | (a.T[upper] != b.T[upper])).sum())


def skeleton_differ(a: np.ndarray, b: np.ndarray) -> int:
    upper = np.triu_indices(np.asarray(a).shape[0], 1)
    return int(((np.asarray(a)[upper] != 0) != (np.asarray(b)[upper] != 0)).sum())


def target_floor(truth: dict) -> int:
    """The ``target_error`` of the given targets alone."""
    return sum(len(set(t) ^ set(g)) for t, g in zip(truth["true"], truth["given"], strict=True))


def score(line: dict, package: str, truth: dict) -> dict:
    """Every metric of one finished draw against ``truth``.

    The estimate is the returned graph's essential graph under the targets the method returned
    (Andrey's and the reference's UT-IGSP, JCI-GSP) or was given (GIES, IGSP); CD-NOD's graph is
    taken as returned. ``consistent`` is whether a returned essential graph equals the one the
    scorer rebuilds from it, a check on the output.
    """
    from andrey.core.orient import pdag2dag

    true_dag = np.asarray(truth["dag"], dtype=np.int8)
    p = true_dag.shape[0]
    truth_graph = essential(true_dag, truth["true"])
    algorithm, targets = line["algorithm"], line.get("targets")
    consistent, dag = None, None
    if package == "andrey":
        marks = np.asarray(line["marks"], dtype=np.int8)
        if algorithm == "cdnod":
            estimate = marks
        else:
            family_targets = targets if algorithm == "ut_igsp" else truth["given"]
            estimate = essential(pdag2dag(marks), family_targets)
            consistent = bool(np.array_equal(estimate, marks))
    else:
        dag = dag_from_arcs(line["arcs"], p)
        estimate = essential(dag, truth["given"] if algorithm == "igsp" else targets)
    try:  # a partially directed graph with no consistent extension has no member DAG
        member = dag if dag is not None else pdag2dag(estimate)
        shd_true_targets = pairs_differ(essential(member, truth["true"]), truth_graph)
    except ValueError:
        shd_true_targets = None
    out = {
        "shd": pairs_differ(estimate, truth_graph),
        "skeleton_shd": skeleton_differ(estimate, truth_graph),
        "empty_shd": int(np.count_nonzero(np.triu(truth_graph != 0))),
        "shd_true_targets": shd_true_targets,
        "target_floor": target_floor(truth),
        "consistent": consistent,
    }
    out["iess_exact"] = out["shd"] == 0
    out["in_true_imec"] = None if shd_true_targets is None else shd_true_targets == 0
    if algorithm == "cdnod":
        union = sorted(set().union(*map(set, truth["true"])))
        out["shd_class"] = pairs_differ(estimate, essential(true_dag, [union]))
    if targets is not None and algorithm != "igsp":
        fp = [len(set(e) - set(t)) for e, t in zip(targets, truth["true"], strict=True)]
        fn = [len(set(t) - set(e)) for e, t in zip(targets, truth["true"], strict=True)]
        tp = sum(len(set(e) & set(t)) for e, t in zip(targets, truth["true"], strict=True))
        designed = truth["designed"]
        out.update(
            target_error=sum(fp) + sum(fn),
            target_fp=sum(fp),
            target_fn=sum(fn),
            target_fn_off=sum(
                len(set(t) - set(e) - set(d))
                for e, t, d in zip(targets, truth["true"], designed, strict=True)
            ),
            target_precision=tp / (tp + sum(fp)) if tp + sum(fp) else None,
            target_recall=tp / (tp + sum(fn)) if tp + sum(fn) else None,
        )
    return out


# ------------------------------------------------------------------------------------------------
# Fits and children
# ------------------------------------------------------------------------------------------------


def config_name(config: dict) -> str:
    """``a<alpha>_<objective>_s<n_starts>``: an Andrey UT-IGSP configuration's name."""
    return f"a{config['alpha']:g}_{config['objective']}_s{config['n_starts']}"


def andrey_defaults() -> dict:
    """``ut_igsp``'s shipped alpha, number of starts, and objective."""
    import andrey
    import andrey.search.igsp as igsp

    params = inspect.signature(andrey.ut_igsp).parameters
    return {
        "alpha": params["alpha"].default,
        "n_starts": params["n_starts"].default,
        "objective": igsp.UT_IGSP_OBJECTIVE,
    }


def _ut_igsp_fit(config: dict, name: str, draws: list[int], **kwargs) -> dict:
    return {
        "algorithm": "ut_igsp",
        "config": name,
        "kwargs": {"alpha": config["alpha"], "n_starts": config["n_starts"], **kwargs},
        "objective": config["objective"],
        "draws": draws,
    }


def andrey_fits(cell: Cell, configs: list[dict] | None = None) -> list[dict]:
    """Andrey's fits on one dataset of ``cell``.

    Block A runs ``configs`` (its candidates). Elsewhere: ``ut_igsp`` at J1 and, when its shipped
    defaults differ, at them (J2); ``gies`` with the designed targets in G1, G2, G4, and G6;
    ``cdnod`` in G1 and G2.
    """
    draws = list(range(cell.draws))
    if cell.arm == "A":
        return [_ut_igsp_fit(c, config_name(c), draws) for c in configs or STEP1]
    extra = {"O6": {"indep_test": "kci"}, "O7": {"indep_test": "gsq"}}.get(cell.arm, {})
    fits = [_ut_igsp_fit(J1_ANDREY, "J1", draws, **extra)]
    defaults = andrey_defaults()
    if defaults != J1_ANDREY:
        fits.append(_ut_igsp_fit(defaults, "J2", draws, **extra))
    if cell.arm in ("G1", "G2", "G4", "G6"):
        fits.append({"algorithm": "gies", "config": "default", "kwargs": {}, "draws": [0]})
    if cell.arm in ("G1", "G2"):
        fits.append({"algorithm": "cdnod", "config": "default", "kwargs": {}, "draws": [0]})
    return fits


def reference_fits(cell: Cell, kci_conditional: bool = False) -> list[dict]:
    """The reference's fits on one dataset of ``cell``; none in block A and O7.

    UT-IGSP at J1 (alpha 1e-5, 5 runs) and, as context, at Andrey's shipped alpha (and 10 runs if
    Andrey ships 10 starts); IGSP in G1 and G2; JCI-GSP in G1, G2, and G5.
    """
    if cell.arm in ("A", "O7"):
        return []
    draws = list(range(cell.draws))
    tester = "kci" if cell.arm == "O6" else "gauss"
    base = {"algorithm": "ut_igsp", "tester": tester, "draws": draws}
    fits = [{**base, "config": "J1", **J1_REFERENCE}]
    if cell.arm == "O6" and kci_conditional:
        fits.append(
            {**base, "config": "J1_conditional", "tester": "kci_conditional", **J1_REFERENCE}
        )
    defaults = andrey_defaults()
    level = {"alpha": defaults["alpha"], "nruns": 10 if defaults["n_starts"] == 10 else NRUNS}
    if level != J1_REFERENCE:
        fits.append({**base, "config": "andrey_level", **level})
    if cell.arm in ("G1", "G2"):
        fits.append({**base, "algorithm": "igsp", "config": "J1", **J1_REFERENCE})
    if cell.arm in ("G1", "G2", "G5"):
        fits.append({**base, "algorithm": "jci_gsp", "config": "J1", **J1_REFERENCE})
    return fits


def child_env() -> dict[str, str]:
    """One BLAS thread and one Andrey worker per child, and a fixed hash seed: the reference's
    JCI-GSP iterates sets of text-named context nodes."""
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ANDREY_N_JOBS"):
        env[var] = "1"
    env["PYTHONHASHSEED"] = "0"
    return env


def run_child(
    python: str, script: Path, request: dict, cap_s: float, cpu: int | None, workdir: Path
) -> tuple[list[dict], bool, int | None, str]:
    """Run one child on ``request``; return its lines, whether it was killed, its exit code, and
    the tail of its standard error."""
    request_path = workdir / f"request-{threading.get_ident()}.npz"
    out_path = workdir / f"lines-{threading.get_ident()}.jsonl"
    np.savez(request_path, **request)
    out_path.unlink(missing_ok=True)
    cmd = [python, str(script), str(request_path), str(out_path)]
    if cpu is not None:
        cmd = ["taskset", "-c", str(cpu), *cmd]
    proc = subprocess.Popen(
        cmd,
        env=child_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    killed = False
    try:
        _, err = proc.communicate(timeout=cap_s)
    except subprocess.TimeoutExpired:
        killed = True
        os.killpg(proc.pid, signal.SIGKILL)
        _, err = proc.communicate()
    lines = (
        [json.loads(x) for x in out_path.read_text().splitlines() if x.strip()]
        if out_path.exists()
        else []
    )
    request_path.unlink(missing_ok=True)
    out_path.unlink(missing_ok=True)
    return lines, killed, proc.returncode, (err or "")[-2000:]


# ------------------------------------------------------------------------------------------------
# run
# ------------------------------------------------------------------------------------------------


def _read(path: Path) -> list[dict]:
    return (
        [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    )


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def run_dataset(cell: Cell, seed: int, packages: list[str], args, cpu: int | None) -> tuple:
    """Fit every package on one dataset; return its truth line and its records."""
    X, env, truth = dataset(cell, seed)
    p = X.shape[1]
    known = np.zeros(X.shape, dtype=np.uint8)
    for k, given in enumerate(truth["given"], start=1):
        known[np.ix_(env == k, given)] = 1
    truth_line = {
        "cell": cell.name,
        "seed": seed,
        **truth,
        "empty_shd": int(
            np.count_nonzero(np.triu(essential(np.array(truth["dag"]), truth["true"]) != 0))
        ),
        "target_floor": target_floor(truth),
        "moments": np.concatenate([X.mean(axis=0), X.std(axis=0)]).tolist(),
    }
    records = []
    for package in packages:
        if package == "andrey":
            fits = andrey_fits(cell, args.configs)
            python, script = args.andrey_python, ANDREY_CHILD
        else:
            fits = reference_fits(cell, args.kci_conditional)
            python, script = args.reference_python, GML_CHILD
        if args.smoke:
            fits = [{**f, "draws": f["draws"][:1]} for f in fits]
        if not fits:
            continue
        cap_draw = cell.cap_draw_s
        draws = sum(len(f["draws"]) for f in fits)
        request = {
            "mode": "fit",
            "X": X,
            "env": env,
            "known": known,
            "fits": json.dumps(fits),
            "warmup_rows": max(50, 2 * p),
        }
        lines, killed, code, err = run_child(
            python, script, request, (draws + 1) * cap_draw + 300, cpu, args.workdir
        )
        found = {(x["algorithm"], x["config"], x["draw"]): x for x in lines}
        for fit in fits:
            for draw in fit["draws"]:
                line = found.get((fit["algorithm"], fit["config"], draw))
                record = {
                    "cell": cell.name,
                    "block": cell.block,
                    "arm": cell.arm,
                    "claim": cell.claim,
                    "seed": seed,
                    "package": package,
                    "algorithm": fit["algorithm"],
                    "config": fit["config"],
                    "fit": {k: v for k, v in fit.items() if k != "draws"},
                    "draw": draw,
                    "cap_draw_s": cap_draw,
                    "machine_id": args.machine_id,
                }
                if line is None:
                    record.update(
                        status="timeout" if killed else "error",
                        error_text=None if killed else f"exit {code}: {err}",
                    )
                elif line["status"] == "ok" and line["wall_s"] > cap_draw:
                    record.update({**line, "status": "timeout"})
                    record.pop("marks", None)
                    record.pop("arcs", None)
                else:
                    record.update(line)
                if record["status"] == "ok":
                    record.update(score(record, package, truth))
                records.append(record)
    return truth_line, records


def parse_seeds(text: str) -> list[int]:
    """``"1000-1004"`` or ``"3,7"``: the seeds a run takes from its cells."""
    seeds: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        seeds += list(range(int(lo), int(hi or lo) + 1))
    return seeds


def run(args) -> int:
    cells = [c for c in grid() if c.arm == args.arm]
    if args.block:
        cells = [replace(c, block=args.block) for c in cells]
    if args.cells:
        wanted = set(args.cells.split(","))
        cells = [c for c in cells if c.name in wanted]
        if missing := wanted - {c.name for c in cells}:
            raise SystemExit(f"no cells {sorted(missing)} in arm {args.arm}")
    if not cells:
        raise SystemExit(f"arm {args.arm} has no cells")
    packages = args.packages.split(",")
    args.andrey_python = os.environ.get("ANDREY_BENCH_PYTHON") or sys.executable
    args.reference_python = os.environ.get(REFERENCE_PYTHON)
    needs_reference = "reference" in packages and any(reference_fits(c) for c in cells)
    if needs_reference and not args.reference_python:
        raise SystemExit(f"set {REFERENCE_PYTHON} to the reference interpreter (start_utigsp.sh)")
    args.configs = None
    if args.arm == "A":
        args.configs = STEP1 if args.step == 1 else step2_configs(args.winner)
    units = [
        (cell, seed)
        for cell in cells
        for seed in (cell.seeds[:1] if args.smoke else cell.seeds)
        if args.seeds is None or seed in args.seeds
    ]

    out = Path(args.out)
    if (out / RECORDS).exists() and not args.resume:
        raise SystemExit(f"{out} already holds a run; pass --resume to continue it")
    out.mkdir(parents=True, exist_ok=True)
    extra = json.loads(Path(args.provenance_extra).read_text()) if args.provenance_extra else {}
    run_meta = {
        "driver": Path(__file__).name,
        "sampler": SAMPLER,
        "arm": args.arm,
        "cells": [asdict(c) | {"name": c.name} for c in cells],
        "seeds": sorted({s for _, s in units}),
        "packages": packages,
        "step": args.step if args.arm == "A" else None,
        "andrey_fits": {c.name: andrey_fits(c, args.configs) for c in cells},
        "reference_fits": {c.name: reference_fits(c, args.kci_conditional) for c in cells},
        "andrey_defaults": andrey_defaults(),
        "smoke": args.smoke,
        "workers": args.workers,
        "machine_id": args.machine_id,
        "extra": extra,
    }
    meta_path = out / "run_meta.json"
    if args.resume and meta_path.exists() and json.loads(meta_path.read_text()) != run_meta:
        raise SystemExit(f"--resume: {meta_path} differs from these flags; pass the first run's")
    meta_path.write_text(json.dumps(run_meta, indent=2))
    from andrey_bench import rundir

    venvs = {}
    if needs_reference:
        venvs["utigsp"] = Path(args.reference_python).absolute().parent.parent
    prov = rundir.provenance_once(
        out, timestamp=datetime.now(timezone.utc).isoformat(), extra_venvs=venvs
    )
    _log(
        f"=== UT-IGSP lane, arm {args.arm}, {len(units)} datasets -> {out} (git {prov['git_sha']})"
    )

    done = {(r["cell"], r["seed"], r["package"]) for r in _read(out / RECORDS)}
    todo = [
        (cell, seed, [p for p in packages if (cell.name, seed, p) not in done])
        for cell, seed in units
    ]
    todo = [t for t in todo if t[2]]
    cpus = sorted(os.sched_getaffinity(0))
    free = list(cpus[: args.workers]) if len(cpus) >= args.workers else [None] * args.workers
    lock = threading.Lock()
    start = time.monotonic()
    skipped = []

    with tempfile.TemporaryDirectory(prefix="utigsp_") as tmp:
        args.workdir = Path(tmp)

        def task(item):
            cell, seed, pkgs = item
            if args.deadline_s and time.monotonic() - start > args.deadline_s:
                with lock:
                    skipped.append((cell.name, seed))
                return
            with lock:
                cpu = free.pop()
            try:
                truth_line, records = run_dataset(cell, seed, pkgs, args, cpu)
            finally:
                with lock:
                    free.append(cpu)
            with lock:
                with (out / DATASETS).open("a") as sink:
                    sink.write(json.dumps(truth_line) + "\n")
                with (out / RECORDS).open("a") as sink:
                    sink.writelines(json.dumps(r) + "\n" for r in records)
                ok = sum(r["status"] == "ok" for r in records)
                _log(
                    f"[{cell.name} seed {seed}] {ok}/{len(records)} ok | "
                    f"{time.monotonic() - start:.0f}s"
                )

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(task, todo))

    records = _read(out / RECORDS)
    if skipped:
        _log(f"deadline: {len(skipped)} datasets left for --resume")
        return 3
    counts = {s: sum(r["status"] == s for r in records) for s in ("ok", "timeout", "error")}
    (out / DONE).write_text(
        json.dumps({"datasets": len(units), "records": len(records), **counts}, indent=2)
    )
    _log(f"done: {counts}")
    return 0


def step2_configs(winner: str | None) -> list[dict]:
    """Block A's step-2 candidates: 1 and 10 starts at the step-1 winner ``a<alpha>_<objective>``."""
    if not winner:
        raise SystemExit("--step 2 needs --winner, the step-1 winner as a<alpha>_<objective>")
    alpha, objective = winner[1:].split("_", 1)
    return [{"alpha": float(alpha), "n_starts": s, "objective": objective} for s in (1, 10)]


# ------------------------------------------------------------------------------------------------
# check-truth
# ------------------------------------------------------------------------------------------------


def check_truth(args) -> int:
    """Compare :func:`essential` with graphical-models' ``interventional_cpdag`` on random pairs."""
    python = os.environ.get(REFERENCE_PYTHON)
    if not python:
        raise SystemExit(f"set {REFERENCE_PYTHON} to the reference interpreter (start_utigsp.sh)")
    from andrey.core import ARROW, TAIL

    rng = np.random.default_rng(654)
    pairs = []
    for _ in range(args.pairs):
        p = int(rng.integers(5, 21))
        order = rng.permutation(p)
        density = rng.uniform(0.1, 0.5)
        arcs = [
            [int(order[a]), int(order[b])]
            for a in range(p)
            for b in range(a + 1, p)
            if rng.random() < density
        ]
        settings = [
            sorted(int(v) for v in rng.choice(p, int(rng.integers(1, 5)), replace=False))
            for _ in range(int(rng.integers(1, 6)))
        ]
        pairs.append({"p": p, "arcs": arcs, "interventions": settings})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="utigsp_truth_") as tmp:
        request = {"mode": "icpdag", "pairs": json.dumps(pairs)}
        lines, killed, code, err = run_child(python, GML_CHILD, request, 1800, None, Path(tmp))
    if killed or len(lines) != len(pairs):
        raise SystemExit(
            f"the reference child answered {len(lines)} of {len(pairs)} (exit {code}): {err}"
        )
    disagree = []
    for index, (pair, line) in enumerate(zip(pairs, lines, strict=True)):
        p = pair["p"]
        theirs = np.zeros((p, p), dtype=np.int8)
        for i, j in line["arcs"]:
            theirs[i, j], theirs[j, i] = TAIL, ARROW
        for i, j in line["edges"]:
            theirs[i, j] = theirs[j, i] = TAIL
        mine = essential(dag_from_arcs(pair["arcs"], p), pair["interventions"])
        if not np.array_equal(mine, theirs):
            disagree.append(index)
    result = {"pairs": len(pairs), "agree": len(pairs) - len(disagree), "disagree": disagree}
    (out / "truth-check.json").write_text(json.dumps(result, indent=2))
    _log(f"truth check: {result['agree']} of {result['pairs']} agree")
    return 1 if disagree else 0


# ------------------------------------------------------------------------------------------------
# summarize and select
# ------------------------------------------------------------------------------------------------


def bootstrap_95(values: np.ndarray) -> list[float]:
    """The 95% percentile interval of the mean of ``values`` over :data:`RESAMPLES` resamples."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(0)
    means = values[rng.integers(0, values.size, size=(RESAMPLES, values.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def per_dataset(records: list[dict]) -> dict[tuple, dict]:
    """Each (cell, seed, package, algorithm, config): whether every draw finished, and the means
    over its draws of each metric, the median time, and the distinct answers over its draws."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        key = (r["cell"], r["seed"], r["package"], r["algorithm"], r["config"])
        groups.setdefault(key, []).append(r)
    out = {}
    for key, rows in groups.items():
        ok = [r for r in rows if r["status"] == "ok"]
        entry = {
            "complete": len(ok) == len(rows),
            "draws": len(rows),
            "timeouts": len(rows) - len(ok),
        }
        if entry["complete"]:
            for metric in (
                "shd",
                "target_error",
                "shd_true_targets",
                "skeleton_shd",
                "empty_shd",
                "target_floor",
            ):
                values = [r.get(metric) for r in ok]
                entry[metric] = None if None in values else float(np.mean(values))
            entry["wall_s"] = statistics.median(r["wall_s"] for r in ok)
            entry["distinct"] = len(
                {json.dumps([r.get("marks") or r.get("arcs"), r.get("targets")]) for r in ok}
            )
        out[key] = entry
    return out


def compare(mine: dict[int, dict], theirs: dict[int, dict], metric: str, datasets: int) -> dict:
    """The paired comparison of two arms on one cell and metric: Andrey's arm against another.

    ``mine`` and ``theirs`` map each seed to its :func:`per_dataset` entry. Pairs only the datasets
    complete for both; ``andrey_short`` counts those complete for the other arm and not for
    Andrey's.
    """
    seeds = sorted(s for s in mine if s in theirs and mine[s]["complete"] and theirs[s]["complete"])
    seeds = [
        s for s in seeds if mine[s].get(metric) is not None and theirs[s].get(metric) is not None
    ]
    D = np.array([mine[s][metric] - theirs[s][metric] for s in seeds], dtype=float)
    row = {
        "paired": len(seeds),
        "andrey_fewer_errors": int((D < 0).sum()),
        "comparator_fewer_errors": int((D > 0).sum()),
        "coverage": {
            "datasets": datasets,
            "andrey_complete": sum(e["complete"] for e in mine.values()),
            "comparator_complete": sum(e["complete"] for e in theirs.values()),
            "andrey_short": sum(
                1
                for s, e in theirs.items()
                if e["complete"] and not mine.get(s, {}).get("complete")
            ),
        },
    }
    if seeds:
        interval = bootstrap_95(D)
        row["multi_draw"] = {
            "draws": max(mine[s]["draws"] for s in seeds),
            "paired": len(seeds),
            f"andrey_{metric}_mean": float(np.mean([mine[s][metric] for s in seeds])),
            f"comparator_{metric}_mean": float(np.mean([theirs[s][metric] for s in seeds])),
            "mean_D": float(D.mean()),
            "bootstrap_95": interval,
            "above_zero": interval[0] > 0,
        }
        row["verdict"] = "worse" if interval[0] > 0 else "better" if interval[1] < 0 else "tie"
        ratios = [mine[s]["wall_s"] / theirs[s]["wall_s"] for s in seeds if theirs[s]["wall_s"]]
        row["speed_ratio_median"] = statistics.median(ratios) if ratios else None
    return row


def floor_interval(mine: dict[int, dict], metric: str, floor: str) -> list[float] | None:
    """The bootstrap interval of the mean of Andrey's ``metric`` minus ``floor`` over its complete
    datasets; ``None`` when the floor is 0 on every one of them."""
    done = [e for e in mine.values() if e["complete"] and e.get(metric) is not None]
    if not done or not any(e[floor] for e in done):
        return None
    return bootstrap_95(np.array([e[metric] - e[floor] for e in done]))


COMPARATORS = {
    ("reference", "ut_igsp", "J1"): "graphical-model-learning.unknown_target_igsp",
    (
        "reference",
        "ut_igsp",
        "andrey_level",
    ): "graphical-model-learning.unknown_target_igsp@andrey-level",
    (
        "reference",
        "ut_igsp",
        "J1_conditional",
    ): "graphical-model-learning.unknown_target_igsp@kci-conditional",
    ("reference", "igsp", "J1"): "graphical-model-learning.igsp",
    ("reference", "jci_gsp", "J1"): "graphical-model-learning.jci_gsp_fixed",
    ("andrey", "gies", "default"): "andrey.gies",
    ("andrey", "cdnod", "default"): "andrey.cdnod",
}


def _arms(records: list[dict]) -> dict[tuple, dict[int, dict]]:
    """:func:`per_dataset` keyed by (cell, package, algorithm, config), then by seed."""
    arms: dict[tuple, dict[int, dict]] = {}
    for (cell, seed, package, algorithm, config), entry in per_dataset(records).items():
        arms.setdefault((cell, package, algorithm, config), {})[seed] = entry
    return arms


def replication(mine: dict[int, dict], theirs: dict[int, dict], metric: str, datasets: int):
    """A judged row's rerun on fresh seeds, in the fields the parity check reads; ``None`` when
    no dataset pairs."""
    row = compare(mine, theirs, metric, datasets)
    if not row["paired"]:
        return None
    seeds = sorted(set(mine) | set(theirs))
    return {
        "seeds": f"{seeds[0]}-{seeds[-1]}",
        "paired": row["paired"],
        "coverage": row["coverage"],
        **{k: v for k, v in row["multi_draw"].items() if k not in ("paired",)},
    }


def summarize(records: list[dict], replications: list[dict] | None = None) -> dict:
    """The summary entries of finished runs (design 4.2), one per (cell group, judged row or
    context comparator, metric), with one row per cell size, and the escalation check (4.1).

    ``replications`` are the records of block C's reruns: a judged row whose cell they hold gets
    its ``replication``.
    """
    reruns = _arms(replications or [])
    by = per_dataset(records)
    cells = {c.name: c for c in grid()}
    arms: dict[tuple, dict[int, dict]] = {}
    for (cell, seed, package, algorithm, config), entry in by.items():
        arms.setdefault((cell, package, algorithm, config), {})[seed] = entry
    entries: dict[tuple, dict] = {}
    for (name, package, algorithm, config), theirs in sorted(arms.items()):
        cell = cells[name]
        if cell.arm == "A" or (package, algorithm, config) not in COMPARATORS:
            continue
        judged = (package, algorithm, config) == ("reference", "ut_igsp", "J1")
        rows = ["J1", "J2"] if judged else ["J1"]
        for row_name in rows:
            mine = arms.get((name, "andrey", "ut_igsp", row_name))
            if mine is None:
                continue
            metrics = ["shd"]
            if algorithm in ("ut_igsp", "jci_gsp"):
                metrics.append("target_error")
            for metric in metrics:
                regime = f"{cell.group}_{row_name}" + (
                    "_targets" if metric == "target_error" else ""
                )
                comparator = COMPARATORS[(package, algorithm, config)]
                entry = entries.setdefault(
                    (regime, comparator),
                    {
                        "method": "ut_igsp",
                        "comparator": comparator,
                        "regime": regime,
                        "metric": metric,
                        "claim": cell.claim,
                        **({} if judged else {"role": "context"}),
                        "rows": [],
                    },
                )
                row = {"mean_degree": cell.degree, "d": cell.p, "n": cell.n}
                row.update(compare(mine, theirs, metric, cell.datasets))
                rerun_mine = reruns.get((name, "andrey", "ut_igsp", row_name))
                rerun_theirs = reruns.get((name, package, algorithm, config))
                if judged and rerun_mine and rerun_theirs:
                    rerun = replication(rerun_mine, rerun_theirs, metric, cell.datasets)
                    if rerun is not None:
                        row["replication"] = rerun
                floor = "empty_shd" if metric == "shd" else "target_floor"
                interval = floor_interval(mine, metric, floor)
                if interval is not None:
                    row["floor_95" if metric == "shd" else "target_floor_95"] = interval
                entry["rows"].append(row)
    return {
        "parity_rule": "paired bootstrap",
        "methods": list(entries.values()),
        "escalation": escalation(arms, cells),
    }


def escalation(arms: dict, cells: dict) -> dict:
    """Design 4.1's escalation check over the G1 and G2 cells with off-targets: Andrey's J1 against
    Andrey's GIES and the reference's IGSP, both given the designed targets."""
    rivals = {"andrey.gies": ("andrey", "gies", "default"), "igsp": ("reference", "igsp", "J1")}
    verdicts: dict[str, dict[str, list[str]]] = {}
    for name, cell in cells.items():
        if cell.arm not in ("G1", "G2") or cell.off == 0:
            continue
        mine = arms.get((name, "andrey", "ut_igsp", "J1"))
        if mine is None:
            continue
        for metric in ("shd", "shd_true_targets", "skeleton_shd"):
            for rival, key in rivals.items():
                theirs = arms.get((name, *key))
                if theirs is None:
                    continue
                verdict = compare(mine, theirs, metric, cell.datasets).get("verdict", "unpaired")
                verdicts.setdefault(name, {}).setdefault(metric, []).append(verdict)
    shd_better = {n: "better" in v.get("shd", []) for n, v in verdicts.items()}
    shd_only = [
        n
        for n, v in verdicts.items()
        if "better" in v.get("shd", [])
        and "better" not in v.get("shd_true_targets", [])
        and "better" not in v.get("skeleton_shd", [])
    ]
    better_than_neither = sum(not b for b in shd_better.values())
    return {
        "cells": len(verdicts),
        "better_than_neither_on_shd": better_than_neither,
        "advantage_on_shd_alone": shd_only,
        "raise": better_than_neither > len(verdicts) / 2 or bool(shd_only),
        "verdicts": verdicts,
    }


def config_from_name(name: str) -> dict:
    alpha, objective, starts = name[1:].split("_")
    return {"alpha": float(alpha), "objective": objective, "n_starts": int(starts[1:])}


def select(records: list[dict], baseline: str) -> dict:
    """Block A's rule (design 4.4): each candidate against ``baseline``, cell by cell.

    A candidate is eligible when, over the cells, it is better than the baseline in more cells than
    it is worse on ``shd``, and not worse in more cells than it is better on ``target_error``. The
    eligible candidate with the largest (better - worse) count on ``shd`` wins; ties go to the
    larger count on ``target_error``, then to the candidate nearer the baseline (alpha nearer
    1e-5, then the sum, then fewer starts). With none eligible, the baseline stays.
    """
    by = per_dataset(
        [r for r in records if r["package"] == "andrey" and r["algorithm"] == "ut_igsp"]
    )
    arms: dict[tuple[str, str], dict[int, dict]] = {}
    for (cell, seed, _, _, config), entry in by.items():
        arms.setdefault((cell, config), {})[seed] = entry
    cells = sorted({c for c, _ in arms})
    configs = sorted({k for _, k in arms} - {baseline})
    datasets = {c.name: c.datasets for c in grid()}
    table = {}
    for config in configs:
        counts = {
            m: {"better": 0, "worse": 0, "tie": 0, "unpaired": 0} for m in ("shd", "target_error")
        }
        per_cell = {}
        for cell in cells:
            mine, base = arms.get((cell, config)), arms.get((cell, baseline))
            if mine is None or base is None:
                raise ValueError(f"cell {cell} lacks {config if mine is None else baseline}")
            per_cell[cell] = {}
            for metric in counts:
                row = compare(mine, base, metric, datasets[cell])
                verdict = row.get("verdict", "unpaired")
                counts[metric][verdict] += 1
                per_cell[cell][metric] = {
                    "verdict": verdict,
                    "mean_D": row.get("multi_draw", {}).get("mean_D"),
                    "bootstrap_95": row.get("multi_draw", {}).get("bootstrap_95"),
                    "coverage": row["coverage"],
                }
        shd, te = counts["shd"], counts["target_error"]
        table[config] = {
            "counts": counts,
            "eligible": shd["better"] > shd["worse"] and te["worse"] <= te["better"],
            "cells": per_cell,
        }

    def rank(config: str) -> tuple:
        c, counts = config_from_name(config), table[config]["counts"]
        return (
            -(counts["shd"]["better"] - counts["shd"]["worse"]),
            -(counts["target_error"]["better"] - counts["target_error"]["worse"]),
            abs(math.log10(c["alpha"]) - math.log10(ALPHA)),
            c["objective"] != "sum",
            c["n_starts"],
        )

    eligible = sorted((c for c in table if table[c]["eligible"]), key=rank)
    return {
        "baseline": baseline,
        "cells": cells,
        "winner": eligible[0] if eligible else baseline,
        "eligible": eligible,
        "candidates": table,
        "coverage": {
            config: {
                cell: sum(e["complete"] for e in arms[(cell, config)].values()) for cell in cells
            }
            for config in [baseline, *configs]
        },
    }


def _records(dirs: list[str]) -> list[dict]:
    records = []
    for d in dirs:
        records += _read(Path(d) / RECORDS)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="UT-IGSP, Andrey against the authors' library, CPU"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="fit one arm's cells")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument("--arm", required=True, help="A, G1-G6, or O1-O9")
    p.add_argument("--cells", default=None, help="only these cells of the arm, comma-separated")
    p.add_argument(
        "--seeds", type=parse_seeds, default=None, help='only these seeds, e.g. "1000-1009"'
    )
    p.add_argument("--packages", default="andrey,reference", help="andrey, reference, or both")
    p.add_argument("--block", choices=sorted(BLOCKS), default=None, help="seeds of this block")
    p.add_argument("--step", type=int, default=1, choices=(1, 2), help="block A's step")
    p.add_argument(
        "--winner", default=None, help="block A step 2: the step-1 winner, a<alpha>_<objective>"
    )
    p.add_argument(
        "--kci-conditional",
        action="store_true",
        help="O6: also the reference's regress=False testers",
    )
    p.add_argument("--workers", type=int, default=1, help="datasets fitted at once, one core each")
    p.add_argument(
        "--deadline-s", type=float, default=None, help="start no dataset after this many seconds"
    )
    p.add_argument("--machine-id", default=socket.gethostname(), help="stamped on every record")
    p.add_argument("--provenance-extra", default=None, help="a JSON file recorded in run_meta.json")
    p.add_argument(
        "--smoke", action="store_true", help="the first dataset of each cell, one draw per fit"
    )
    p.add_argument("--resume", action="store_true", help="continue the run already in --out")
    t = sub.add_parser("check-truth", help="compare the truth function with graphical-models'")
    t.add_argument("--out", required=True)
    t.add_argument("--pairs", type=int, default=100)
    s = sub.add_parser("summarize", help="summary entries of finished runs")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--replications", nargs="+", default=[], help="block C's runs")
    s.add_argument("--out", required=True)
    c = sub.add_parser("select", help="block A's default selection")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--baseline", default=config_name(J1_ANDREY))
    c.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    if args.command == "check-truth":
        return check_truth(args)
    if args.command == "summarize":
        summary = summarize(_records(args.runs), _records(args.replications))
        Path(args.out).write_text(json.dumps(summary, indent=2))
        _log(
            f"{len(summary['methods'])} entries -> {args.out}; escalation {summary['escalation']['raise']}"
        )
        return 0
    result = select(_records(args.runs), args.baseline)
    Path(args.out).write_text(json.dumps(result, indent=2))
    _log(f"winner {result['winner']}; eligible {result['eligible']}")
    for config, row in result["candidates"].items():
        _log(
            f"  {config:28s} shd {row['counts']['shd']} target_error {row['counts']['target_error']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
