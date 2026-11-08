#!/usr/bin/env python
"""Benchmark J-PCMCI+ on CPU: Andrey against tigramite's JPCMCIplus, on the paper's simulation.

The claimed cells of :data:`CELLS` (``claim="in"``) draw their datasets with
:func:`~andrey_bench.context_models.jpcmciplus2023`, the linear model of the J-PCMCI+ authors'
experiment code (Günther, Ninad, and Runge 2023, UAI): 5 system variables, 2 time and 2 space
contexts, ``tau_max=2``. They are the settings of the paper's Figures 2 and 3: every context
observed with 10 datasets of 10 to 200 steps, and one context of each kind observed with 10
datasets of 10 to 200 steps or with 5 to 200 datasets of 10 steps. That code gives every link
between system variables a positive lag (see the model's docstring), so in these cells the
same-step part of the error counts false links only. The cells outside the claim (``claim="out"``)
are the model of :func:`andrey.data.sample_context_series` at the sizes of its earlier comparison
(1 time and 2 space contexts, all observed, half the links at lag 0), the paper's model with 10 and
15 system variables, and its nonlinear model with GPDC, marked ``dry_run_first``.

Every arm fits the same datasets, one after another, each in a fresh interpreter on one core with
one BLAS thread (:func:`andrey_bench.runner.run_task`), at alpha 0.05, the paper's level, and 0.01,
both packages' default (:data:`ARMS`):

- ``andrey``: ``andrey.jpcmciplus`` with both dummies, every other argument at its default;
- ``tigramite``: tigramite's ``JPCMCIplus`` with both dummies and the stock ``ParCorrMult``, every
  argument but ``pc_alpha`` and ``tau_max`` at its default, with the fix of
  ``patches/tigramite-jpcmciplus-context-links.patch`` applied in memory
  (:mod:`andrey_bench.adapters.tigramite_jpcmciplus`);
- ``andrey.pooled``: ``andrey.pcmciplus`` on the system variables of the datasets one after
  another, the paper's baseline: reported, never judged;
- ``tigramite.stock``: tigramite without the fix, at 0.05, in the cells whose contexts are all
  observed: the check that the fix changes no fit that finishes without it.

A fit past its cap is a ``timeout`` with no scores, never an empty graph; an Andrey fit that raises
``ValueError`` is ``refused``. :func:`scores` computes the endpoint SHD over the links between system
variables at lags 0 to ``tau_max`` (``shd``: a missing or extra link costs its two marks, a wrong
mark one; :func:`andrey_bench.scoring.window_scores`), and, where every context is observed, the
links from the contexts into the system variables that a fit misses plus those it adds
(``context_error``); links between contexts are not scored. The empty graph is scored the same way.

One JSON line per dataset goes to ``<out>/datasets.jsonl`` and one per (cell, seed, arm, alpha) to
``<out>/records.jsonl``, each with the build that made it; ``--resume`` skips what is on disk.
``summarize`` pairs the records of one or more runs into rows: per row, the percentile bootstrap
95% interval (10,000 resamples) of the mean paired difference, Andrey minus tigramite. A claimed row,
at alpha 0.05, is judged where it has at least :data:`MIN_PAIRS` paired datasets and tigramite's own
interval against the empty graph lies below 0; Andrey must then beat the empty graph too, and an
Andrey fit that did not finish where tigramite's did fails the row. In the cells with hidden
contexts, the datasets where a hidden context confounds an observed one and a system variable
(Assumption 3 broken) form a row of their own, reported and not judged. A ``better`` or ``worse``
row is listed for a rerun on the cell's ``rerun_seeds``, and counts only if the rerun agrees.

    $ANDREY_BENCH_PYTHON benchmarks/jpcmciplus_cpu.py run --out outputs/jpcmciplus \\
        --cells all-M10-T100 half-M5-T10 --seeds 0-19
    $ANDREY_BENCH_PYTHON benchmarks/jpcmciplus_cpu.py summarize --runs outputs/jpcmciplus \\
        --out outputs/jpcmciplus-summary.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import socket
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from andrey_bench.adapters.tigramite_jpcmciplus import PYTHON_ENV_VAR, patch_id
from andrey_bench.context_models import ContextDraw, jpcmciplus2023, jpcmciplus2023_nonlinear
from andrey_bench.runner import run_task
from andrey_bench.scoring import window_scores
from andrey_bench.window import WindowGraph, marks_from_edges

TAU_MAX = 2
ALPHAS = (0.05, 0.01)

#: The level of the paper's experiments, the level of the claimed rows.
CLAIM_ALPHA = 0.05

#: Wall cap of one fit, imports and warm-up included.
CAP_WALL_S = 1800.0

#: Memory cap of one fit's process tree.
CAP_MEM_MB = 16000.0

BOOTSTRAP = 10_000
MIN_PAIRS = 10

#: The datasets of a cell with hidden contexts, split by the paper's Assumption 3.
HOLDS, BROKEN, ALL = "assumption 3 holds", "assumption 3 broken", "all"


@dataclass(frozen=True)
class Cell:
    """One setting: its model, its sizes, and the fraction of each kind of context observed.

    ``seeds`` and ``rerun_seeds`` are a cell's default seeds and those of its replication: 40 in the
    cells with hidden contexts, where about half the draws break Assumption 3 and leave the judged
    row, else 20.
    """

    name: str
    model: str  # "jpcmciplus2023", "nonlinear", or "sample_context_series"
    n_datasets: int
    n_steps: int
    observed: float
    claim: str
    n_system: int = 5
    test: str = "parcorr"  # "parcorr" or "gpdc"
    dry_run_first: bool = False

    @property
    def all_observed(self) -> bool:
        """Whether every context is observed, so the context links are scored."""
        return self.observed == 1.0

    @property
    def seeds(self) -> str:
        return "0-19" if self.all_observed else "0-39"

    @property
    def rerun_seeds(self) -> str:
        return "100-119" if self.all_observed else "100-139"

    def warmup(self, arm: str) -> int:
        """Warm-up fits before the timed one: none with GPDC or for tigramite with more than 5
        system variables, whose fits run for minutes, where a warm-up would double a fit that may
        run to the cap."""
        slow = self.test == "gpdc" or (arm.startswith("tigramite") and self.n_system > 5)
        return 0 if slow else 1


#: The time and space contexts of each model.
CONTEXTS = {"jpcmciplus2023": (2, 2), "nonlinear": (1, 1), "sample_context_series": (1, 2)}

STEPS = (10, 20, 30, 50, 100, 150, 200)
DATASETS = (5, 20, 30, 50, 100, 150, 200)

CELLS = {
    cell.name: cell
    for cell in (
        *(Cell(f"all-M10-T{t}", "jpcmciplus2023", 10, t, 1.0, "in") for t in STEPS),
        *(Cell(f"half-M10-T{t}", "jpcmciplus2023", 10, t, 0.5, "in") for t in STEPS),
        *(Cell(f"half-M{m}-T10", "jpcmciplus2023", m, 10, 0.5, "in") for m in DATASETS),
        *(
            Cell(f"current-M{m}-T{t}", "sample_context_series", m, t, 1.0, "out")
            for m, t in ((5, 100), (10, 50), (10, 100), (10, 200), (20, 50), (20, 100), (50, 10))
        ),
        *(Cell(f"all-N{n}-M10-T100", "jpcmciplus2023", 10, 100, 1.0, "out", n) for n in (10, 15)),
        *(
            Cell(f"gpdc-M{m}-T20", "nonlinear", m, 20, 1.0, "out", 2, "gpdc", True)
            for m in (5, 10, 20, 30, 50, 60)
        ),
    )
}


@dataclass(frozen=True)
class Arm:
    """One solution: its role and the levels it runs at."""

    name: str
    role: str  # "andrey", "reference", "context", or "check"
    alphas: tuple[float, ...] = ALPHAS


ARMS = {
    arm.name: arm
    for arm in (
        Arm("andrey", "andrey"),
        Arm("tigramite", "reference"),
        Arm("andrey.pooled", "context"),
        Arm("tigramite.stock", "check", (CLAIM_ALPHA,)),
    )
}

#: The Andrey arms, whose ``ValueError`` is a refusal.
ANDREY_ARMS = ("andrey", "andrey.pooled")


def cell_arms(cell: Cell) -> list[str]:
    """The arms a cell runs: the patch check only where every context is observed, and not with
    GPDC, whose model has one context of each kind."""
    arms = ["andrey", "tigramite", "andrey.pooled"]
    if cell.all_observed and cell.test != "gpdc":
        arms.append("tigramite.stock")
    return arms


# --- data and truth --------------------------------------------------------------------------------


@dataclass
class Dataset:
    """A cell's draw for one seed and the window truth of its system variables."""

    cell: Cell
    seed: int
    draw: ContextDraw
    truth: WindowGraph


def draw(cell: Cell, seed: int) -> Dataset:
    """Draw `cell`'s datasets for `seed`."""
    if cell.model == "jpcmciplus2023":
        d = jpcmciplus2023(cell.n_system, cell.n_datasets, cell.n_steps, cell.observed, seed)
    elif cell.model == "nonlinear":
        d = jpcmciplus2023_nonlinear(cell.n_datasets, cell.n_steps, seed)
    else:
        from andrey.data import sample_context_series

        n_time, n_space = CONTEXTS[cell.model]
        s = sample_context_series(
            cell.n_system,
            n_time,
            n_space,
            n_datasets=cell.n_datasets,
            n_steps=cell.n_steps,
            tau_max=TAU_MAX,
            observed=cell.observed,
            seed=seed,
        )
        contexts = s.time_contexts + s.space_contexts
        d = ContextDraw(
            datasets=np.stack(s.datasets),
            n_system=cell.n_system,
            time_contexts=s.time_contexts,
            space_contexts=s.space_contexts,
            links=s.links,
            context_links=s.context_links,
            breaks_assumption_3=False,
            model=(),
            observed=contexts,
            noise_sd=(),
        )
    n = d.n_system
    same_step = np.zeros((n, n), dtype=bool)
    lagged = np.zeros((TAU_MAX, n, n), dtype=bool)
    for cause, effect, lag in d.links:
        if lag == 0:
            same_step[cause, effect] = True
        else:
            lagged[lag - 1, cause, effect] = True
    return Dataset(cell, seed, d, WindowGraph(marks_from_edges(same_step, lagged), lag0="dag"))


def setting(data: Dataset) -> dict[str, Any]:
    """The setting every row of `data` records."""
    cell, d = data.cell, data.draw
    n_time, n_space = CONTEXTS[cell.model]
    return {
        "model": cell.model,
        "n_system": d.n_system,
        "n_time": n_time,
        "n_space": n_space,
        "observed_time": len(d.time_contexts),
        "observed_space": len(d.space_contexts),
        "n_datasets": cell.n_datasets,
        "n_steps": cell.n_steps,
        "tau_max": TAU_MAX,
        "test": cell.test,
    }


# --- arms ------------------------------------------------------------------------------------------


def adapter(arm: str):
    """A fresh adapter for `arm`."""
    if arm == "andrey":
        from andrey_bench.adapters.andrey_jpcmciplus import AndreyJPCMCIPlus

        return AndreyJPCMCIPlus()
    if arm == "andrey.pooled":
        from andrey_bench.adapters.andrey_jpcmciplus import AndreyPooledPCMCIPlus

        return AndreyPooledPCMCIPlus()
    if arm in ("tigramite", "tigramite.stock"):
        from andrey_bench.adapters.tigramite_jpcmciplus import tigramite_jpcmciplus_adapter

        return tigramite_jpcmciplus_adapter()
    raise KeyError(arm)


def params(arm: str, data: Dataset, alpha: float) -> dict[str, Any]:
    """The ``params`` of `arm` on `data` at `alpha`, over its default ones."""
    d = data.draw
    columns = {"time_contexts": list(d.time_contexts), "space_contexts": list(d.space_contexts)}
    gpdc = data.cell.test == "gpdc"
    if arm == "andrey":
        return {
            "tau_max": TAU_MAX,
            "alpha": alpha,
            "indep_test": "gpdc" if gpdc else None,
        } | columns
    if arm == "andrey.pooled":
        return {"tau_max": TAU_MAX, "alpha": alpha, "indep_test": "gpdc" if gpdc else None} | {
            "n_system": d.n_system
        }
    if arm in ("tigramite", "tigramite.stock"):
        return {
            "tau_max": TAU_MAX,
            "pc_alpha": alpha,
            "test": "gpdc" if gpdc else "parcorr_mult",
            "patch": arm == "tigramite",
            "n_system": d.n_system,
        } | columns
    raise KeyError(arm)


# --- scores ----------------------------------------------------------------------------------------


def mark_list(marks: np.ndarray) -> list[list[int]]:
    """``[lag, i, j, mark]`` for each nonzero window mark."""
    m = np.asarray(marks)
    return [[int(k), int(i), int(j), int(m[k, i, j])] for k, i, j in zip(*np.nonzero(m))]


def found_context_links(marks: np.ndarray, draw: ContextDraw) -> set[tuple[int, int, int]]:
    """The ``(context, lag, variable)`` links from contexts into system variables in `marks`."""
    n = draw.n_system
    return {
        (c, k, j)
        for c in draw.time_contexts + draw.space_contexts
        for k in range(marks.shape[0])
        for j in range(n)
        if marks[k, c, j] != 0
    }


def scores(marks: np.ndarray | None, data: Dataset, contexts: bool) -> dict[str, Any]:
    """The system scores of `marks` (``None``: the empty graph) and, with `contexts` where every
    context is observed, its context-link counts."""
    d = data.draw
    n = d.n_system
    if marks is None:
        marks = np.zeros((TAU_MAX + 1, n, n), dtype=np.int8)
    s = window_scores(WindowGraph(np.asarray(marks)[:, :n, :n], lag0="cpdag"), data.truth)
    out: dict[str, Any] = {
        "shd": s["endpoint_shd"],
        "shd_lag0": s["lag0_endpoint_shd"],
        "shd_lagged": s["lagged_endpoint_shd"],
        "adjacency_precision": s["adjacency_precision"],
        "adjacency_recall": s["adjacency_recall"],
        "adjacency_f1": s["adjacency_f1"],
        "wrong_marks": s["wrong_marks"],
        "links": s["n_est_links"],
    }
    if contexts and data.cell.all_observed:
        found = found_context_links(np.asarray(marks), d) if marks.shape[1] > n else set()
        true = set(d.context_links)
        out |= {
            "context_hits": len(found & true),
            "context_misses": len(true - found),
            "context_extras": len(found - true),
            "context_error": len(true - found) + len(found - true),
        }
    return out


# --- run -------------------------------------------------------------------------------------------


def _cpu_model() -> str:
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append(path: Path, record: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(record, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def _seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        seeds += list(range(int(lo), int(hi or lo) + 1))
    return seeds


def _git(args: list[str], cwd: Path) -> str | None:
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=60)
        if out.returncode == 0:
            return out.stdout.strip()
    return None


def build() -> dict[str, Any]:
    """The build every record carries: Andrey's version, commit, and uncommitted changes to its
    source, NumPy and Python in each fit environment, the CPU, tigramite's version, and the
    patch."""
    import andrey

    src = Path(andrey.__file__).resolve().parent
    out: dict[str, Any] = {
        "andrey": andrey.__version__,
        "andrey_commit": _git(["rev-parse", "HEAD"], src),
        "andrey_dirty": bool(_git(["status", "--porcelain", "--untracked-files=no", "."], src)),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "cpu_model": _cpu_model(),
        "tigramite": None,
        "tigramite_numpy": None,
        "tigramite_python": None,
        "tigramite_patch": patch_id(),
    }
    python = os.environ.get(PYTHON_ENV_VAR, "").strip()
    if python:
        probe = (
            "import importlib.metadata as m, platform, numpy; "
            "print(m.version('tigramite'), numpy.__version__, platform.python_version())"
        )
        with contextlib.suppress(OSError, subprocess.SubprocessError, ValueError):
            got = subprocess.run([python, "-c", probe], capture_output=True, text=True, timeout=120)
            tig, tig_numpy, tig_python = got.stdout.split()
            out |= {"tigramite": tig, "tigramite_numpy": tig_numpy, "tigramite_python": tig_python}
    return out


def _status(arm: str, result: dict) -> str:
    """The harness status, or ``refused`` for an Andrey fit that raised ``ValueError``."""
    lines = [line for line in str(result.get("stderr") or "").splitlines() if line.strip()]
    if result["status"] == "error" and arm in ANDREY_ARMS and lines:
        if lines[-1].startswith("ValueError"):
            return "refused"
    return result["status"]


def fit_arm(arm: str, data: Dataset, alpha: float, args, built: dict) -> dict[str, Any]:
    """Fit `arm` on `data` at `alpha` in a fresh interpreter; return its record."""
    solution = adapter(arm)
    fit_params = {**solution.params(), **params(arm, data, alpha)}
    t0 = time.perf_counter()
    result = run_task(
        solution,
        np.ascontiguousarray(data.draw.datasets),
        fit_params,
        cap_wall_s=args.cap,
        cap_mem_mb=args.cap_mem_mb,
        repeats=1,
        warmup=data.cell.warmup(arm),
        threads=1,
    )[0]
    tigramite = arm.startswith("tigramite")
    record: dict[str, Any] = {
        "arm": arm,
        "alpha": alpha,
        "role": ARMS[arm].role,
        "solution": solution.name,
        "package_version": str(getattr(solution, "package_version", "")),
        "params": fit_params,
        "status": _status(arm, result),
        "harness_status": result["status"],
        "cap_wall_s": args.cap,
        "warmup": data.cell.warmup(arm),
        "task_s": time.perf_counter() - t0,
        "setting": setting(data) | {"alpha": alpha},
        "build": {
            **built,
            "numpy": built["tigramite_numpy"] if tigramite else built["numpy"],
            "python": built["tigramite_python"] if tigramite else built["python"],
        },
    }
    for key in ("wall_s", "cpu_s", "peak_rss_mb", "elapsed_at_kill_s", "exit_code", "threads"):
        if key in result:
            record[key] = result[key]
    for key in ("error_text", "stderr"):
        if result.get(key):
            record[key] = str(result[key])[-1500:]
    if result["status"] != "ok":
        return record
    marks = np.asarray(result["adj"], dtype=np.int8)
    record["marks"] = mark_list(marks)
    record["marks_sha1"] = hashlib.sha1(np.ascontiguousarray(marks).tobytes()).hexdigest()
    return record | scores(marks, data, contexts=arm != "andrey.pooled")


def dataset_record(data: Dataset) -> dict[str, Any]:
    """The dataset line: its setting, its truth, Assumption 3, and the empty graph's scores."""
    d = data.draw
    return {
        "cell": data.cell.name,
        "seed": data.seed,
        "setting": setting(data),
        "claim": data.cell.claim,
        "rows_per_test": d.datasets.shape[0] * (d.datasets.shape[1] - 2 * TAU_MAX),
        "links": sorted(d.links),
        "context_links": sorted(d.context_links),
        "model": [list(link) for link in d.model],
        "observed": list(d.observed),
        "noise_sd": list(d.noise_sd),
        "breaks_assumption_3": d.breaks_assumption_3,
        "empty": scores(None, data, contexts=True),
        "data_sha1": hashlib.sha1(np.ascontiguousarray(d.datasets).tobytes()).hexdigest(),
    }


def run(args) -> int:
    """Fit every arm of each cell on each seed; append to ``datasets.jsonl`` and ``records.jsonl``."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec_path, ds_path = out / "records.jsonl", out / "datasets.jsonl"
    done = (
        {(r["cell"], r["seed"], r["arm"], r["alpha"]) for r in _read(rec_path)}
        if args.resume
        else set()
    )
    have = {(r["cell"], r["seed"]) for r in _read(ds_path)} if args.resume else set()
    if not args.resume:
        for path in (rec_path, ds_path):
            path.unlink(missing_ok=True)
    built = build()
    meta = {
        "argv": vars(args) | {"func": None},
        "build": built,
        "machine": args.machine_id or socket.gethostname(),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _append(out / "provenance.jsonl", meta)
    counts: dict[str, int] = {}
    for name in args.cells:
        cell = CELLS[name]
        arms = [a for a in cell_arms(cell) if not args.arms or a in args.arms]
        for seed in args.seeds or _seeds(cell.seeds):
            data = draw(cell, seed)
            if (name, seed) not in have:
                _append(ds_path, dataset_record(data))
            for arm in arms:
                for alpha in ARMS[arm].alphas:
                    if args.alphas and alpha not in args.alphas:
                        continue
                    if (name, seed, arm, alpha) in done:
                        continue
                    record = {"cell": name, "seed": seed} | fit_arm(arm, data, alpha, args, built)
                    _append(rec_path, record)
                    counts[record["status"]] = counts.get(record["status"], 0) + 1
                    print(
                        f"{name} seed {seed} {arm} alpha {alpha}: {record['status']} "
                        f"shd {record.get('shd')} context {record.get('context_error')} "
                        f"fit {record.get('wall_s', float('nan')):.2f} s "
                        f"task {record['task_s']:.1f} s",
                        flush=True,
                    )
    _append(
        out / "done.jsonl",
        {"finished": datetime.now(timezone.utc).isoformat(timespec="seconds"), "counts": counts},
    )
    return 0


# --- summarize -------------------------------------------------------------------------------------


def bootstrap(values: list[float], resamples: int = BOOTSTRAP) -> list[float] | None:
    """Percentile bootstrap 95% interval of the mean, ``default_rng(0)``."""
    if not values:
        return None
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(0)
    means = v[rng.integers(0, len(v), size=(resamples, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _statuses(records) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in records:
        key = "missing" if rec is None else rec["status"]
        counts[key] = counts.get(key, 0) + 1
    return counts


def _ok(rec: dict | None) -> bool:
    return rec is not None and rec["status"] == "ok"


def subsets(cell: Cell, seeds: list[int], datasets: dict) -> dict[str, list[int]]:
    """The seeds of each row: all of them where every context is observed, else split by
    Assumption 3."""
    if cell.all_observed:
        return {ALL: seeds}
    broken = {s for s in seeds if datasets[(cell.name, s)]["breaks_assumption_3"]}
    return {HOLDS: [s for s in seeds if s not in broken], BROKEN: sorted(broken)}


def paired_row(
    cell: Cell, alpha: float, metric: str, subset: str, seeds: list[int], recs: dict, datasets: dict
) -> dict[str, Any]:
    """The row of Andrey against tigramite in `cell` at `alpha` on `metric`, over `seeds`."""
    mine = {s: recs.get((cell.name, s, "andrey", alpha)) for s in seeds}
    theirs = {s: recs.get((cell.name, s, "tigramite", alpha)) for s in seeds}
    empty = {s: datasets[(cell.name, s)]["empty"][metric] for s in seeds}
    paired = [s for s in seeds if _ok(mine[s]) and _ok(theirs[s])]
    d = [mine[s][metric] - theirs[s][metric] for s in paired]
    a_empty = [mine[s][metric] - empty[s] for s in seeds if _ok(mine[s])]
    r_empty = [theirs[s][metric] - empty[s] for s in seeds if _ok(theirs[s])]
    interval = bootstrap(d)
    floor_a, floor_r = bootstrap(a_empty), bootstrap(r_empty)
    applies = floor_r is not None and floor_r[1] < 0
    claimed = cell.claim == "in" and alpha == CLAIM_ALPHA and subset != BROKEN
    judged = claimed and applies and len(paired) >= MIN_PAIRS
    verdict = None
    if interval is not None and len(paired) >= MIN_PAIRS:
        verdict = "worse" if interval[0] > 0 else "better" if interval[1] < 0 else "tie"
    failures = [s for s in seeds if _ok(theirs[s]) and not _ok(mine[s])]
    floor_pass = floor_a is not None and floor_a[1] < 0
    if not judged:
        result = "not judged"
    elif failures:
        result = "fail: Andrey did not finish where tigramite did"
    elif not floor_pass:
        result = "fail: not better than the empty graph"
    else:
        result = verdict
    ratios = [mine[s]["wall_s"] / theirs[s]["wall_s"] for s in paired if theirs[s].get("wall_s")]
    row: dict[str, Any] = {
        "regime": f"{cell.name}|{metric}|alpha={alpha}|{subset}",
        "cell": cell.name,
        "metric": metric,
        "alpha": alpha,
        "subset": subset,
        "setting": {k: v for k, v in asdict(cell).items() if k != "name"},
        "claim": "in" if claimed else "out",
        "judged": judged,
        "seeds": seeds,
        "paired": len(paired),
        "andrey_mean": _mean([mine[s][metric] for s in paired]),
        "comparator_mean": _mean([theirs[s][metric] for s in paired]),
        "empty_mean": _mean([empty[s] for s in paired]),
        "mean_D": _mean(d),
        "bootstrap_95": interval,
        "verdict": verdict,
        "result": result,
        "rerun": judged and result in ("better", "worse"),
        "failures": failures,
        "same_graph": sum(mine[s]["marks_sha1"] == theirs[s]["marks_sha1"] for s in paired),
        "andrey_fit_s_median": _median([mine[s]["wall_s"] for s in paired]),
        "comparator_fit_s_median": _median([theirs[s]["wall_s"] for s in paired]),
        "speed_ratio_median": _median(ratios),
        "coverage": {
            "datasets": len(seeds),
            "andrey_status": _statuses(mine.values()),
            "reference_status": _statuses(theirs.values()),
        },
        "floor": {
            "andrey_minus_empty": floor_a,
            "reference_minus_empty": floor_r,
            "applies": applies,
            "pass": floor_pass if applies else None,
        },
    }
    if claimed and not judged:
        row["reason"] = (
            "reference finished no dataset"
            if floor_r is None
            else "reference not better than the empty graph"
            if not applies
            else f"fewer than {MIN_PAIRS} paired datasets"
        )
    return row


def cell_rows(cell: Cell, recs: dict, datasets: dict) -> list[dict[str, Any]]:
    """The paired rows of `cell`: per level, the system SHD on each subset, and the context error
    where every context is observed."""
    seeds = sorted(s for (c, s) in datasets if c == cell.name)
    rows = []
    for alpha in ALPHAS:
        for subset, chosen in subsets(cell, seeds, datasets).items():
            if not chosen:
                continue
            metrics = ("shd", "context_error") if cell.all_observed else ("shd",)
            rows += [paired_row(cell, alpha, m, subset, chosen, recs, datasets) for m in metrics]
    return rows


def context_rows(cell: Cell, recs: dict, datasets: dict) -> list[dict[str, Any]]:
    """Each arm's mean scores and fit times in `cell` per level, and the empty graph's."""
    seeds = sorted(s for (c, s) in datasets if c == cell.name)
    rows = []
    keys = ("shd", "shd_lag0", "shd_lagged", "adjacency_f1", "wrong_marks", "context_error")
    for arm in cell_arms(cell):
        for alpha in ARMS[arm].alphas:
            got = [recs.get((cell.name, s, arm, alpha)) for s in seeds]
            ok = [g for g in got if _ok(g)]
            row: dict[str, Any] = {"cell": cell.name, "alpha": alpha, "arm": arm}
            row["status"] = _statuses(got)
            for key in keys:
                row[f"{key}_mean"] = _mean([g[key] for g in ok if g.get(key) is not None])
            row["fit_s_median"] = _median([g["wall_s"] for g in ok])
            row["task_s_median"] = _median([g["task_s"] for g in got if g is not None])
            rows.append(row)
    empties = [datasets[(cell.name, s)]["empty"] for s in seeds]
    rows.append(
        {"cell": cell.name, "arm": "empty", "datasets": len(empties)}
        | {f"{k}_mean": _mean([e[k] for e in empties if k in e]) for k in keys}
    )
    return rows


def patch_check(cell: Cell, recs: dict, datasets: dict) -> dict[str, Any] | None:
    """Whether every fit that finished without the patch gave the patched fit's marks."""
    if "tigramite.stock" not in cell_arms(cell):
        return None
    seeds = sorted(s for (c, s) in datasets if c == cell.name)
    stock = {s: recs.get((cell.name, s, "tigramite.stock", CLAIM_ALPHA)) for s in seeds}
    patched = {s: recs.get((cell.name, s, "tigramite", CLAIM_ALPHA)) for s in seeds}
    finished = [s for s in seeds if _ok(stock[s]) and _ok(patched[s])]
    differ = [s for s in finished if stock[s]["marks_sha1"] != patched[s]["marks_sha1"]]
    return {
        "cell": cell.name,
        "stock_status": _statuses(stock.values()),
        "patched_status": _statuses(patched.values()),
        "stock_finished": len(finished),
        "same_as_patched": len(finished) - len(differ),
        "differ": differ,
    }


def _load(dirs: list[str]) -> tuple[dict, dict]:
    """The records by (cell, seed, arm, alpha) and the datasets by (cell, seed). Runs split into
    parts by seeds or arms repeat a dataset's line, with the same data."""
    recs, datasets = {}, {}
    for d in dirs:
        for path in sorted(Path(d).rglob("records.jsonl")):
            for r in _read(path):
                recs[(r["cell"], r["seed"], r["arm"], r["alpha"])] = r
        for path in sorted(Path(d).rglob("datasets.jsonl")):
            for r in _read(path):
                key = (r["cell"], r["seed"])
                if key in datasets and datasets[key]["data_sha1"] != r["data_sha1"]:
                    raise ValueError(f"{key} has two different draws")
                datasets[key] = r
    return recs, datasets


#: The fields of a row's rerun kept on the row.
REPLICATION_KEYS = (
    "seeds",
    "paired",
    "andrey_mean",
    "comparator_mean",
    "mean_D",
    "bootstrap_95",
    "verdict",
    "failures",
    "coverage",
    "floor",
)


def summarize(args) -> int:
    """Pair the records of ``--runs`` into rows; ``--replications`` adds each row's rerun."""
    recs, datasets = _load(args.runs)
    again = _load(args.replications) if args.replications else None
    cells = [c for c in CELLS.values() if any(k[0] == c.name for k in datasets)]
    rows, context, patches = [], [], []
    for cell in cells:
        for row in cell_rows(cell, recs, datasets):
            if again is not None and any(k[0] == cell.name for k in again[1]):
                rep = {r["regime"]: r for r in cell_rows(cell, again[0], again[1])}.get(
                    row["regime"]
                )
                if rep is not None:
                    row["replication"] = {k: rep[k] for k in REPLICATION_KEYS}
                    if row["rerun"]:
                        replicated = rep["verdict"] == row["verdict"] and not rep["failures"]
                        row["result"] = row["verdict"] if replicated else "tie (did not replicate)"
                        row["rerun"] = False
            rows.append(row)
        context += context_rows(cell, recs, datasets)
        if (check := patch_check(cell, recs, datasets)) is not None:
            patches.append(check)
    rerun = [
        {"cell": r["cell"], "alpha": r["alpha"], "metric": r["metric"], "subset": r["subset"]}
        | {"seeds": CELLS[r["cell"]].rerun_seeds}
        for r in rows
        if r["rerun"]
    ]
    builds = {json.dumps(r["build"], sort_keys=True) for r in recs.values()}
    summary = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lane": "jpcmciplus",
        "parity_rule": "paired bootstrap",
        "bootstrap": {
            "resamples": BOOTSTRAP,
            "interval": "percentile 95%",
            "rng": "default_rng(0)",
        },
        "min_pairs": MIN_PAIRS,
        "claim_alpha": CLAIM_ALPHA,
        "datasets": len(datasets),
        "breaks_assumption_3": sum(d["breaks_assumption_3"] for d in datasets.values()),
        "builds": [json.loads(b) for b in sorted(builds)],
        "status": _statuses(recs.values()),
        "results": _statuses({"status": r["result"]} for r in rows if r["judged"]),
        "rows": rows,
        "rerun": rerun,
        "patch_check": patches,
        "context": context,
    }
    Path(args.out).write_text(json.dumps(summary, indent=1, default=_json_default) + "\n")
    for row in rows:
        print(
            f"{row['regime']:58s} {row['claim']:3s} n={row['paired']:2d} "
            f"A={row['andrey_mean']} R={row['comparator_mean']} E={row['empty_mean']} "
            f"D={row['mean_D']} {row['bootstrap_95']} {row['result']}"
        )
    for check in patches:
        print(
            f"patch check {check['cell']}: stock finished {check['stock_finished']}, "
            f"same {check['same_as_patched']}, stock {check['stock_status']}"
        )
    print(f"status {summary['status']}; judged {summary['results']}; reruns {len(rerun)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="fit every arm of the cells on the seeds")
    r.add_argument("--out", required=True)
    r.add_argument("--cells", nargs="+", choices=list(CELLS), required=True)
    r.add_argument("--seeds", type=_seeds, help="e.g. 0-19 or 0,5,7-9; default each cell's seeds")
    r.add_argument("--arms", nargs="+", choices=list(ARMS), help="only these arms")
    r.add_argument("--alphas", nargs="+", type=float, help="only these levels")
    r.add_argument("--cap", type=float, default=CAP_WALL_S, help="wall cap per fit, seconds")
    r.add_argument("--cap-mem-mb", type=float, default=CAP_MEM_MB)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--machine-id", default=None)
    r.set_defaults(func=run)
    s = sub.add_parser("summarize", help="pair the records into rows")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--replications", nargs="+")
    s.add_argument("--out", required=True)
    s.set_defaults(func=summarize)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
