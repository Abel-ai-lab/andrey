#!/usr/bin/env python
"""Benchmark the Markov blanket and parents-and-children learners on CPU: Andrey against bnlearn
and the authors' Causal Explorer.

Runs each learner of :data:`andrey_bench.adapters.andrey_blanket.LEARNERS` that the installed
Andrey has, beside the bnlearn function that runs its algorithm
(:data:`andrey_bench.adapters.bnlearn_blanket.COUNTERPART`), the Causal Explorer function that does
(:data:`andrey_bench.adapters.causal_explorer_blanket.COUNTERPART`, in target mode, when MATLAB and
the toolbox are there), bnlearn's ``pc.stable`` read for each kind of set, and the empty answer.
Every solution gets the same data, the same test, the same ``alpha``, and the same limit on the
conditioning set (``--max-depth``), all recorded in every row.
Two kinds of data, as :data:`CELLS` writes them:

- ``net:<name>:<n>``: ``n`` rows sampled from a published network's own tables, discrete, with
  Andrey's ``--indep-test`` (``gsq`` or ``chisq``) against bnlearn's ``mi`` or ``x2`` form
  (``--discrete-test``, below);
- ``gauss:<d>:<n>[:<degree>]``: ``n`` rows of a linear-Gaussian graph on ``d`` variables with
  ``degree`` expected edges per variable (2 by default), standardized, with Fisher's z (bnlearn's
  ``zf``).

The columns come in an order drawn from the seed. ``--mode`` picks what a fit answers:

- ``target`` (the default): the set of each target, ``--targets`` of them (``all``, or a number
  drawn from the seed), as a Markov blanket for a learner of blankets and as parents and children
  for a learner of those. A fit answers for every target, and its time is the time for all of
  them; the tests of each search are in the row where the package reports them.
- ``graph``: the graph the sets give, each of ``--structures`` that a learner offers: the moral
  graph, the skeleton, and the CPDAG.

``--discrete-test`` picks bnlearn's form of the test: ``adf`` for ``mi-adf`` and ``x2-adf``, which
assume independence on a table with fewer than 5 rows per cell on average and take empty rows and
columns out of the degrees of freedom, and ``plain`` for ``mi`` and ``x2``, which do neither.
Andrey's tests have no such rule. See :mod:`andrey_bench.adapters.bnlearn_blanket`.

Each fit runs isolated, under a per-fit wall cap; a fit past the cap is a ``timeout`` with no
scores. A set is scored against the true DAG's (:func:`andrey_bench.blanket_lane.set_scores`):
precision, recall, F1, and the distance of (recall, precision) to (1, 1), per target, and a graph
by its adjacencies (:func:`andrey_bench.blanket_lane.pair_scores`) or, for the CPDAG, by
``andrey.metrics``. The empty answer is the floor. One JSON line per fit goes to
``<out>/records.jsonl`` with the answer, so ``--rescore`` recomputes the scores without fitting;
``<out>/provenance.json`` records the build and the machine.

Load R with bnlearn 5.2.1 as the README's R lane describes before running bnlearn's solutions, and
for Causal Explorer's, put ``matlab`` on the ``PATH`` and run ``benchmarks/start_causal_explorer.sh``
(see :mod:`andrey_bench.adapters.causal_explorer_blanket`); without them the lane leaves that arm
out.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/blanket_cpu.py --out outputs/blanket-cpu
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from andrey import GraphStructure
from andrey_bench import datasets, rsession, rundir, scoring
from andrey_bench.adapters import andrey_blanket, bnlearn_blanket, causal_explorer_blanket
from andrey_bench.adapters.andrey_blanket import AndreyBlanket
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.bnlearn_blanket import BnlearnBlanket
from andrey_bench.adapters.causal_explorer_blanket import CausalExplorerBlanket
from andrey_bench.blanket_lane import (
    NO_COUNT,
    STRUCTURES,
    TARGETS_PARAM,
    pair_scores,
    pooled_set_scores,
    set_scores,
    sets_of,
    true_graph,
    true_set,
    unpack_answer,
)
from andrey_bench.runner import available_cores, run_task, split_cores

RECORDS = "records.jsonl"

#: The default cells: three published networks and two Gaussian sizes, each at two sample sizes.
CELLS = (
    "net:child:1000",
    "net:child:5000",
    "net:insurance:1000",
    "net:insurance:5000",
    "net:alarm:1000",
    "net:alarm:5000",
    "gauss:20:1000",
    "gauss:20:5000",
    "gauss:50:1000",
    "gauss:50:5000",
)

#: One small network, one seed, and a short cap: touches every solution.
SMOKE = {"cells": ["net:asia:500"], "seeds": 1, "cap_wall_s": 300.0, "targets": "all"}

#: The test every Gaussian cell runs; the discrete tests are ``--indep-test``.
CONTINUOUS_TEST = "fisherz"


@dataclass(frozen=True)
class Cell:
    """One data setting: a network's rows, or a linear-Gaussian graph's."""

    kind: str  # "network" or "gaussian"
    name: str  # the network's name, "" for a Gaussian graph
    d: int  # variables of a Gaussian graph, 0 for a network
    n: int
    degree: float = datasets.CANONICAL_DENSITY

    @classmethod
    def parse(cls, text: str) -> "Cell":
        """``net:<name>:<n>`` or ``gauss:<d>:<n>[:<degree>]``."""
        parts = text.split(":")
        try:
            if parts[0] == "net" and len(parts) == 3:
                return cls("network", parts[1], 0, int(parts[2]))
            if parts[0] == "gauss" and len(parts) in (3, 4):
                degree = float(parts[3]) if len(parts) == 4 else datasets.CANONICAL_DENSITY
                return cls("gaussian", "", int(parts[1]), int(parts[2]), degree)
        except ValueError:
            pass
        raise ValueError(f"a cell is net:<name>:<n> or gauss:<d>:<n>[:<degree>], got {text!r}")

    @property
    def label(self) -> str:
        """The cell as ``parse`` reads it: ``net:<name>:<n>``, ``gauss:<d>:<n>:<degree>``."""
        if self.kind == "network":
            return f"net:{self.name}:{self.n}"
        return f"gauss:{self.d}:{self.n}:{self.degree:g}"

    def key(self, seed: int) -> datasets.DatasetKey:
        if self.kind == "network":
            return datasets.network_key(self.name, self.n, seed)
        topology, functional, noise, standardize = datasets.REGIMES["linear_gauss_er"]
        return datasets.DatasetKey(
            topology=topology,
            functional=functional,
            noise=noise,
            standardize=standardize,
            density=self.degree,
            d=self.d,
            n=self.n,
            seed=seed,
        )

    def indep_test(self, discrete_test: str) -> str:
        return discrete_test if self.kind == "network" else CONTINUOUS_TEST


@dataclass(frozen=True)
class Settings:
    """What every solution of a run shares."""

    discrete_test: str  # Andrey's name: gsq or chisq
    sparse_rule: bool  # bnlearn's -adf form
    alpha: float
    max_depth: int | None


def targets_of(d: int, spec: str, seed: int) -> list[int]:
    """The column positions to answer for: every column, or ``spec`` of them drawn from `seed`."""
    if spec == "all" or int(spec) >= d:
        return list(range(d))
    return sorted(int(t) for t in np.random.default_rng(seed).choice(d, int(spec), replace=False))


def field(
    cell: Cell, settings: Settings, structure: str | None, causal_explorer: bool | None = None
) -> list:
    """The solutions of a cell: Andrey's learners, the bnlearn function and, in target mode, the
    Causal Explorer function of each, bnlearn's ``pc.stable`` for each kind of set, and the empty
    answer last.

    A whole-graph run (`structure` named) takes only the learners that offer that graph.
    `causal_explorer` says whether to run Causal Explorer's functions; ``None`` runs them when
    MATLAB and the toolbox are there (:func:`causal_explorer_blanket.available`).
    """
    test = cell.indep_test(settings.discrete_test)
    shared = {"indep_test": test, "alpha": settings.alpha, "max_depth": settings.max_depth}
    if causal_explorer is None:
        causal_explorer = causal_explorer_blanket.available()
    solutions: list = []
    kinds: set[str] = set()
    for learner in andrey_blanket.available():
        offered = andrey_blanket.LEARNERS[learner]
        if structure is not None and structure not in offered.structures:
            continue
        solutions.append(AndreyBlanket(learner, structure=structure, **shared))
        kinds.add(offered.kind)
        function = bnlearn_blanket.COUNTERPART.get(learner)
        if function and (
            structure is None or structure in bnlearn_blanket.FUNCTIONS[function].structures
        ):
            solutions.append(
                BnlearnBlanket(
                    function, structure=structure, sparse_rule=settings.sparse_rule, **shared
                )
            )
        function = causal_explorer_blanket.COUNTERPART.get(learner)
        if (
            causal_explorer
            and structure is None
            and function
            and test in causal_explorer_blanket.TESTS[function]
        ):
            solutions.append(CausalExplorerBlanket(function, **shared))
    if structure is None:
        for kind in sorted(kinds):
            solutions.append(
                BnlearnBlanket("pc.stable", kind=kind, sparse_rule=settings.sparse_rule, **shared)
            )
    elif structure in bnlearn_blanket.FUNCTIONS["pc.stable"].structures:
        solutions.append(
            BnlearnBlanket(
                "pc.stable", structure=structure, sparse_rule=settings.sparse_rule, **shared
            )
        )
    solutions.append(BaselineEmpty())
    return solutions


def kinds_of(solutions: list) -> list[str]:
    """The kinds of set the field's solutions learn, sorted."""
    return sorted({a.kind for a in solutions if getattr(a, "kind", None)})


def reference_name(name: str, kind: str | None) -> str | None:
    """The bnlearn solution that runs the algorithm of the Andrey solution `name`, if any."""
    if not name.startswith("andrey."):
        return None
    function = bnlearn_blanket.COUNTERPART.get(name.removeprefix("andrey."))
    return None if function is None else "bnlearn." + function.replace(".", "-")


def authors_name(name: str) -> str | None:
    """The Causal Explorer solution that runs the algorithm of the Andrey solution `name`, if any."""
    if not name.startswith("andrey."):
        return None
    function = causal_explorer_blanket.COUNTERPART.get(name.removeprefix("andrey."))
    return None if function is None else CausalExplorerBlanket(function, indep_test="gsq").name


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def graph_adjacency(marks: np.ndarray) -> np.ndarray:
    """The 0/1 adjacency of a graph's endpoint marks, a pair joined by either mark."""
    return ((np.asarray(marks) != 0) | (np.asarray(marks) != 0).T).astype(np.int8)


def score_sets(answer: np.ndarray, truth: np.ndarray, targets: list[int], kind: str) -> dict:
    """A target answer's scores: each target's, and the pool over them."""
    found = sets_of(answer, targets)
    per_target = {t: set_scores(found[t], true_set(truth, t, kind)) for t in targets}
    return {
        "sets": {str(t): found[t] for t in targets},
        "per_target": {str(t): s for t, s in per_target.items()},
        **{f"pooled_{k}": v for k, v in pooled_set_scores(list(per_target.values())).items()},
    }


def score_graph(marks: np.ndarray, truth: np.ndarray, structure: str) -> dict:
    """A whole-graph answer's scores against the true DAG `truth` (endpoint marks)."""
    out: dict[str, Any] = {"marks": np.asarray(marks, dtype=int).tolist()}
    if structure == "cpdag":
        dag = GraphStructure.from_numpy(truth, kind="dag")
        estimate = GraphStructure.from_numpy(np.asarray(marks, dtype=np.int8), kind="cpdag")
        empty = GraphStructure.from_numpy(np.zeros_like(truth), kind="cpdag")
        metrics = scoring.score_result(estimate, dag)
        out.update({k: float(metrics[k]) for k in ("shd", "arrowhead_f1")})
        out.update({k: float(metrics[f"skeleton_{k}"]) for k in ("precision", "recall", "f1")})
        out["empty_shd"] = float(scoring.score_result(empty, dag)["shd"])
        return out
    out.update(pair_scores(graph_adjacency(marks), true_graph(truth, structure)))
    return out


def make_record(base: dict, measure: dict) -> dict:
    """A fit's record before scoring: its coordinates and what the runner measured."""
    return {
        **base,
        "status": measure.get("status"),
        "wall_s": measure.get("wall_s"),
        "cpu_s": measure.get("cpu_s"),
        "peak_rss_mb": measure.get("peak_rss_mb"),
        # The text of a finished fit is the packages' own warnings; only a failure is worth keeping.
        "error_text": (
            None
            if measure.get("status") == "ok"
            else measure.get("error_text") or measure.get("stderr")
        ),
    }


def score(record: dict, adapter: Any, measure: dict, truth: np.ndarray) -> None:
    """Add the answer, its scores, and its test counts to the record of a finished fit."""
    if measure.get("status") != "ok" or measure.get("adj") is None:
        return
    native = np.asarray(measure["adj"])
    answer, counts, total = unpack_answer(native)
    if record["structure"] is None:
        targets = record["targets"]
        record.update(score_sets(answer, truth, targets, record["kind"]))
        record["n_tests"] = {
            str(t): None if counts[t] == NO_COUNT else int(counts[t]) for t in targets
        }
    else:
        marks = adapter.to_structure(native).to_numpy()
        record.update(score_graph(marks, truth, record["structure"]))
    record["total_tests"] = total


def rescore(out: Path) -> list[dict]:
    """Recompute the scores of every record in `out` from its stored answer; rewrite the file."""
    records = _read(out)
    truths: dict[tuple[str, int], np.ndarray] = {}
    for record in records:
        if record["status"] != "ok":
            continue
        cell = Cell.parse(record["cell"])
        if (record["cell"], record["seed"]) not in truths:
            truths[record["cell"], record["seed"]] = datasets.generate_one(
                cell.key(record["seed"])
            )[1]
        truth = truths[record["cell"], record["seed"]]
        if record["mode"] == "target":
            targets = [int(t) for t in record["sets"]]
            answer = np.zeros((truth.shape[0],) * 2, dtype=np.int8)
            for t, members in record["sets"].items():
                answer[int(t), members] = 1
            record.update(score_sets(answer, truth, targets, record["kind"]))
        else:
            record.update(score_graph(np.asarray(record["marks"]), truth, record["structure"]))
    scratch = out / f".{RECORDS}.tmp"
    scratch.write_text("".join(json.dumps(r, default=str) + "\n" for r in records))
    scratch.replace(out / RECORDS)
    return records


def _read(out: Path) -> list[dict]:
    path = out / RECORDS
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _record_key(record: dict) -> tuple:
    return (
        record["cell"],
        record["seed"],
        record["solution"],
        record["mode"],
        record.get("kind"),
        record.get("structure"),
        record["repeat"],
    )


def parse_max_depth(text: str) -> int | None:
    """``none`` or an integer of at least 1, the least bnlearn's ``max.sx`` takes."""
    if text.lower() == "none":
        return None
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("--max-depth is none or at least 1 (bnlearn's max.sx)")
    return value


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Markov blanket learners, Andrey against bnlearn, CPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--cells",
        nargs="+",
        default=list(CELLS),
        help="data settings, net:<name>:<n> or gauss:<d>:<n>[:<degree>] (default: ten cells)",
    )
    p.add_argument("--mode", choices=("target", "graph"), default="target")
    p.add_argument(
        "--structures",
        nargs="+",
        choices=STRUCTURES,
        default=list(STRUCTURES),
        help="graph mode: the graphs to learn (default: every one a learner offers)",
    )
    p.add_argument(
        "--targets",
        default="all",
        help="target mode: 'all', or the number of targets drawn from the seed (default: all)",
    )
    p.add_argument("--indep-test", choices=("gsq", "chisq"), default="gsq", help="discrete cells")
    p.add_argument(
        "--discrete-test",
        choices=("adf", "plain"),
        default="adf",
        help="bnlearn's form of the discrete test: mi-adf and x2-adf, with the sparse-table rule "
        "and adjusted degrees of freedom, or mi and x2, with neither (default: adf)",
    )
    p.add_argument("--alpha", type=float, default=0.05, help="significance level (default: 0.05)")
    p.add_argument(
        "--max-depth",
        type=parse_max_depth,
        default=None,
        help="largest conditioning set for both sides: none (default) or at least 1",
    )
    p.add_argument("--seeds", type=int, default=20, help="datasets per cell (default: 20)")
    p.add_argument("--first-seed", type=int, default=0, help="the first seed (default: 0)")
    p.add_argument("--part", type=int, default=0, help="run every --parts-th cell from this one")
    p.add_argument("--parts", type=int, default=1, help="parts the cells are split into")
    p.add_argument("--repeats", type=int, default=1, help="timed repeats per fit (default: 1)")
    p.add_argument("--warmup", type=int, default=1, help="untimed fits per repeat (default: 1)")
    p.add_argument("--cap-wall-s", type=float, default=1800.0, help="wall cap per repeat, seconds")
    p.add_argument("--cap-mem-mb", type=float, default=None, help="memory cap per repeat, MB")
    p.add_argument(
        "--cores",
        type=int,
        default=available_cores(),
        help="core budget per fit (default: every core this job is allowed)",
    )
    p.add_argument(
        "--solutions",
        default=None,
        help="run only these solution names, comma-separated (default: every solution)",
    )
    p.add_argument(
        "--machine-id",
        default=socket.gethostname(),
        help="machine id stamped on every record (default: this host's name)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="run asia, 500 rows, one seed, every target, and a 300 s cap, replacing those flags",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue the run already in --out: skip the fits whose records are on disk",
    )
    p.add_argument(
        "--rescore",
        action="store_true",
        help="fit nothing: recompute the scores of the run already in --out from its stored answers",
    )
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE)
    try:
        cells = [Cell.parse(text) for text in args.cells][args.part :: args.parts]
        if args.targets != "all" and int(args.targets) < 1:
            raise ValueError("--targets is 'all' or a positive number")
    except ValueError as error:
        p.error(str(error))

    out = Path(args.out)
    if args.rescore:
        if not (out / RECORDS).exists():
            raise SystemExit(f"--rescore: {out / RECORDS} holds no records")
        records = rescore(out)
        _log(f"rescored    : {len(records)} records -> {out / RECORDS}")
        _summarize(records)
        return 0
    if (out / RECORDS).exists() and not args.resume:
        raise SystemExit(f"{out} already holds a run; pass --resume to continue it")

    settings = Settings(args.indep_test, args.discrete_test == "adf", args.alpha, args.max_depth)
    structures = [None] if args.mode == "target" else list(args.structures)
    chosen = None if args.solutions is None else set(args.solutions.split(","))
    fields = {
        (cell.label, structure): [
            a for a in field(cell, settings, structure) if chosen is None or a.name in chosen
        ]
        for cell in cells
        for structure in structures
    }
    known = {a.name for solutions in fields.values() for a in solutions}
    if chosen is not None and chosen - known:
        p.error(f"unknown solutions {sorted(chosen - known)}; the solutions are {sorted(known)}")
    names = sorted(known)
    needs_r = any(a.package == "bnlearn" for solutions in fields.values() for a in solutions)
    r_environment = {}
    if needs_r:
        r_environment = rsession.check_environment("bnlearn")
        bnlearn_blanket.check_version()
    needs_matlab = any(
        a.package == "causal-explorer" for solutions in fields.values() for a in solutions
    )
    if not needs_matlab:
        _log("causal-explorer: matlab or the toolbox is absent; its arm is left out")

    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "run_meta.json"
    run_meta = {
        "driver": Path(__file__).name,
        "mode": args.mode,
        "structures": [s for s in structures if s],
        "solution list": names,
        "params": {
            f"{label}/{structure}": {a.name: a.params() for a in solutions}
            for (label, structure), solutions in fields.items()
        },
        "indep_test": {"discrete": args.indep_test, "continuous": CONTINUOUS_TEST},
        "discrete_test": args.discrete_test,
        "sparse_rule": args.discrete_test == "adf",
        "alpha": args.alpha,
        "max_depth": args.max_depth,
        "targets": args.targets,
        "r environment": r_environment,
        "causal explorer": causal_explorer_blanket.environment() if needs_matlab else {},
        "cells": [c.label for c in cells],
        "seeds": args.seeds,
        "first_seed": args.first_seed,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "cap_wall_s": args.cap_wall_s,
        "cap_mem_mb": args.cap_mem_mb,
        "cores": args.cores,
        "machine_id": args.machine_id,
    }
    if args.resume and meta_path.exists() and json.loads(meta_path.read_text()) != run_meta:
        raise SystemExit(f"--resume: {meta_path} differs from these flags; pass the first run's")
    meta_path.write_text(json.dumps(run_meta, indent=2))
    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())
    _log(f"=== Markov blanket learners, Andrey and bnlearn -> {out} ===")
    for name, value in run_meta.items():
        if name != "params":
            _log(f"{name:12s}: {value}")
    _log(f"{'git_sha':12s}: {prov['git_sha']}")
    _log("")

    done = {_record_key(r) for r in _read(out)}
    start = time.perf_counter()
    with (out / RECORDS).open("a") as sink:
        for cell in cells:
            for seed in range(args.first_seed, args.first_seed + args.seeds):
                key = cell.key(seed)
                X, truth, _, _ = datasets.generate_one(key)
                targets = targets_of(X.shape[1], args.targets, seed)
                walls: dict[str, list[float]] = {}
                for structure in structures:
                    solutions = fields[cell.label, structure]
                    kinds = kinds_of(solutions)
                    for adapter in solutions:
                        # The empty answer is one fit, scored as each kind of set.
                        scored_as = (
                            kinds
                            if adapter.package == "baseline" and structure is None
                            else [getattr(adapter, "kind", None)]
                        )
                        wanted = [
                            (cell.label, seed, adapter.name, args.mode, kind, structure, repeat)
                            for kind in scored_as
                            for repeat in range(args.repeats)
                        ]
                        if all(w in done for w in wanted):
                            continue
                        params = adapter.params()
                        if structure is None:
                            params = {**params, TARGETS_PARAM: targets}
                        threads, n_jobs = split_cores(args.cores, adapter.mode)
                        measures = run_task(
                            adapter,
                            X,
                            params,
                            cap_wall_s=args.cap_wall_s,
                            cap_mem_mb=args.cap_mem_mb,
                            repeats=args.repeats,
                            warmup=args.warmup,
                            threads=threads,
                            n_jobs=n_jobs,
                        )
                        test = cell.indep_test(settings.discrete_test)
                        base = {
                            "cell": cell.label,
                            "data": cell.kind,
                            "d": int(X.shape[1]),
                            "n": cell.n,
                            "seed": seed,
                            "dataset": key.digest,
                            "mode": args.mode,
                            "structure": structure,
                            "targets": targets if structure is None else None,
                            "solution": adapter.name,
                            "package": adapter.package,
                            "package_version": adapter.package_version,
                            "indep_test": test,
                            "bnlearn_test": bnlearn_blanket.TESTS[(test, settings.sparse_rule)],
                            "sparse_rule": settings.sparse_rule and test != CONTINUOUS_TEST,
                            "alpha": settings.alpha,
                            "max_depth": settings.max_depth,
                            "params": params,
                            "machine_id": args.machine_id,
                        }
                        for repeat, m in enumerate(measures):
                            for kind in scored_as:
                                label = (cell.label, seed, adapter.name, args.mode, kind)
                                if (*label, structure, repeat) in done:
                                    continue
                                record = make_record({**base, "kind": kind, "repeat": repeat}, m)
                                score(record, adapter, m, truth)
                                if record["status"] == "ok":
                                    walls.setdefault(adapter.name, []).append(m["wall_s"])
                                sink.write(json.dumps(record, default=str) + "\n")
                                sink.flush()
                cost = "  ".join(f"{n}={statistics.median(w):.3g}s" for n, w in walls.items())
                _log(
                    f"[{cell.label} seed={seed}] {cost} | elapsed {time.perf_counter() - start:.0f}s"
                )
    _summarize(_read(out))
    return 0


def _summarize(records: list[dict]) -> None:
    """Print, per cell and solution, the finished fits, median time, the scores, and agreement.

    Target mode: pooled precision and recall over the datasets' targets, the mean F1 and distance
    over the targets where they are defined, ``exact`` the share of targets whose set is the true
    one, ``same`` the targets where the solution's set is the one its bnlearn counterpart found,
    and ``authors`` the targets where it is the one its Causal Explorer counterpart found. Graph
    mode: pooled precision and recall of the adjacencies, the mean ``shd`` against the empty
    graph's, and ``same`` the datasets with the counterpart's graph.
    """
    if not records:
        return
    ok = [r for r in records if r["status"] == "ok"]
    _log("")
    _log(
        f"{'solution':24s} {'set':>9s} {'ok/all':>7s} {'wall_s':>8s} {'prec':>6s} {'recall':>6s} "
        f"{'f1':>6s} {'dist':>6s} {'exact':>6s} {'tests':>8s} {'same':>9s} {'authors':>9s}"
    )
    groups = sorted({(r["cell"], r["mode"], r["kind"], r["structure"]) for r in records}, key=str)
    for cell, mode, kind, structure in groups:
        in_group = [
            r
            for r in records
            if (r["cell"], r["mode"], r["kind"], r["structure"]) == (cell, mode, kind, structure)
        ]
        _log(f"-- {cell} {mode} {structure or kind}")
        for name in sorted({r["solution"] for r in in_group}):
            mine = [r for r in in_group if r["solution"] == name]
            fit = [r for r in mine if r["status"] == "ok"]
            if not fit:
                _log(f"{name:24s} {structure or kind:>9s} {0:3d}/{len(mine):<3d}")
                continue
            wall = statistics.median(r["wall_s"] for r in fit)

            def fits_of(reference: str | None) -> dict[int, dict]:
                return {
                    r["seed"]: r
                    for r in ok
                    if r["solution"] == reference
                    and (r["cell"], r["mode"], r["kind"], r["structure"])
                    == (cell, mode, kind, structure)
                }

            theirs = fits_of(reference_name(name, kind))
            if mode == "target":
                agree = []
                for reference in (theirs, fits_of(authors_name(name))):
                    same = total = 0
                    for r in fit:
                        other = reference.get(r["seed"])
                        if other is not None:
                            total += len(r["sets"])
                            same += sum(r["sets"][t] == other["sets"][t] for t in r["sets"])
                    agree.append(f"{same}/{total}" if total else "-")
                tests = [c for r in fit for c in r["n_tests"].values() if c is not None]
                _log(
                    f"{name:24s} {structure or kind:>9s} {len(fit):3d}/{len(mine):<3d} {wall:8.3g} "
                    f"{_pool(fit, 'precision'):6.3f} {_pool(fit, 'recall'):6.3f} "
                    f"{_mean(fit, 'f1'):6.3f} {_mean(fit, 'distance'):6.3f} "
                    f"{_pool(fit, 'exact'):6.3f} "
                    f"{(statistics.mean(tests) if tests else float('nan')):8.4g} "
                    f"{agree[0]:>9s} {agree[1]:>9s}"
                )
            else:
                same = total = 0
                for r in fit:
                    other = theirs.get(r["seed"])
                    if other is not None:
                        total += 1
                        same += int(r["marks"] == other["marks"])
                shd = statistics.mean(r["shd"] for r in fit)
                empty = statistics.mean(r["empty_shd"] for r in fit)
                _log(
                    f"{name:24s} {structure:>9s} {len(fit):3d}/{len(mine):<3d} {wall:8.3g} "
                    f"{_mean_of(fit, 'precision'):6.3f} {_mean_of(fit, 'recall'):6.3f} "
                    f"{_mean_of(fit, 'f1'):6.3f} {'':6s} {'':6s} {'':8s} "
                    f"{(f'{same}/{total}' if total else '-'):>9s}  shd {shd:.3g} (empty {empty:.3g})"
                )


def _mean_of(fit: list[dict], name: str) -> float:
    """The mean of a graph score over the fits where it is defined."""
    values = [r[name] for r in fit if r[name] == r[name]]
    return statistics.mean(values) if values else float("nan")


def _pool(fit: list[dict], name: str) -> float:
    """A ratio over the counts of every target of every fit; ``exact`` over the targets."""
    if name == "exact":
        flags = [s["exact"] for r in fit for s in r["per_target"].values()]
        return sum(flags) / len(flags) if flags else float("nan")
    tp = sum(s["tp"] for r in fit for s in r["per_target"].values())
    est = sum(s["est"] for r in fit for s in r["per_target"].values())
    true = sum(s["true"] for r in fit for s in r["per_target"].values())
    den = est if name == "precision" else true
    return tp / den if den else float("nan")


def _mean(fit: list[dict], name: str) -> float:
    """The mean of a per-target ratio over the targets where it is defined."""
    values = [s[name] for r in fit for s in r["per_target"].values() if s[name] == s[name]]
    return statistics.mean(values) if values else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
