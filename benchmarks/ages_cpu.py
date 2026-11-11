#!/usr/bin/env python
"""Benchmark AGES on CPU: Andrey against pcalg, on linear Gaussian data with weak edges.

Runs Andrey's `ages`, pcalg's `ages`, and Andrey's `ges` at the same penalty on the simulation of
Eigenmann, Nandy, and Maathuis (2017, Section 4.1) (:func:`sample`):

- a DAG on :data:`D` variables: each pair of variables, in a causal order, holds a strong edge with
  probability ``qs``, a weak edge with probability ``qw``, and no edge otherwise;
- edge weights of magnitude ``U(0.8, 1.2)`` on strong edges and ``U(0.1, 0.3)`` on weak ones, each
  with a random sign, and Gaussian errors of variance ``U(0.5, 1.5)``;
- ``n`` rows, the columns in an order drawn from the seed.

:data:`CELLS` holds the paper's settings: the 13 pairs ``(qs, qw)`` from 0.1, 0.3, 0.5, and 0.7
with ``qs + qw <= 1``, then ``qs`` from 0.1 to 1.0 without weak edges; :data:`ROWS` its sample
sizes.

Each fit runs isolated, under a per-fit wall cap; a fit past the cap is a ``timeout`` with no
scores. A graph is scored against the generating DAG (:func:`scores`):

- ``correct``, ``directed``, and ``true_edges`` give the paper's measures: precision is the share
  of the graph's directed edges that the true DAG directs the same way, and recall the share of
  the true DAG's edges the graph directs correctly;
- ``shd`` counts the variable pairs where the graph differs from every graph between the true
  CPDAG and the true DAG. The graph AGES estimates has the true skeleton, directs every edge the
  true CPDAG directs, and directs no edge against the true DAG, so an edge left undirected where
  the CPDAG leaves it undirected is no error, and neither is one directed as the DAG directs it;
- ``shd_cpdag`` and ``shd_dag`` count the pairs that differ from the true CPDAG and from the true
  DAG; ``empty_shd`` is the empty graph's count on all three, the true edge count.

One JSON line per fit goes to ``<out>/records.jsonl`` with the graph's endpoint marks, so
``--rescore`` recomputes the scores without fitting; ``<out>/provenance.json`` records the build and
the machine. ``ages_summary.py`` compares the solutions per setting.

Load R with pcalg as the README's R lane describes before running pcalg's solution.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/ages_cpu.py --out outputs/ages-cpu
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from andrey_bench import rsession, rundir
from andrey_bench.adapters.andrey_ages import AndreyAGES
from andrey_bench.adapters.andrey_ges import AndreyGES
from andrey_bench.adapters.andrey_hc import _andrey_version
from andrey_bench.adapters.pcalg_ages import PcalgAGES
from andrey_bench.runner import available_cores, run_task, split_cores

#: Variables of every dataset, the paper's ``p``.
D = 10

#: The solutions: Andrey, the reference, and GES with the score and penalty AGES starts from (BIC,
#: no parent limit below ``D - 1``), whose CPDAG is the first graph of Andrey's path.
SOLUTIONS = [
    AndreyAGES(),
    PcalgAGES(),
    AndreyGES(
        name="andrey.ges.numpy.bic",
        backend="numpy",
        mode="serial",
        package_version=_andrey_version(),
        max_parents=D - 1,
    ),
]

#: The paper's settings ``(qs, qw)``: the shares of strong and of weak edges among the pairs.
CELLS: tuple[tuple[float, float], ...] = (
    *((qs, qw) for qs in (0.1, 0.3, 0.5, 0.7) for qw in (0.1, 0.3, 0.5, 0.7) if qs + qw <= 1.01),
    *((round(0.1 * k, 1), 0.0) for k in range(1, 11)),
)

#: The paper's sample sizes: the main text's, then the supplement's two.
ROWS = (10000, 1000, 100)

#: Magnitude ranges of the strong and the weak edge weights, and the range of the error variances.
STRONG, WEAK, VARIANCE = (0.8, 1.2), (0.1, 0.3), (0.5, 1.5)

#: The generator a run's records come from; a run recorded with another cannot be rescored.
SAMPLER = "ages_cpu.sample, numpy.random.default_rng([10 qs, 10 qw, n, seed])"

#: Smallest run that touches every solution: one small dataset.
SMOKE = {"cells": ["0.3:0.7"], "rows": [1000], "seeds": 1, "cap_wall_s": 300.0}

RECORDS = "records.jsonl"

# Andrey endpoint marks.
_TAIL, _ARROW = 1, 2


def sample(qs: float, qw: float, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(samples, dag)`` for one dataset; ``dag`` holds the DAG's endpoint marks."""
    rng = np.random.default_rng([round(10 * qs), round(10 * qw), n, seed])
    upper = np.triu_indices(D, 1)
    draw = rng.random(upper[0].size)
    strong, weak = draw < qs, (draw >= qs) & (draw < qs + qw)
    magnitude = np.where(strong, rng.uniform(*STRONG, draw.size), rng.uniform(*WEAK, draw.size))
    sign = rng.choice([-1.0, 1.0], draw.size)
    weights = np.zeros((D, D))  # weights[i, j] is the coefficient of i in the equation of j
    weights[upper] = np.where(strong | weak, magnitude * sign, 0.0)
    errors = rng.standard_normal((n, D)) * np.sqrt(rng.uniform(*VARIANCE, D))
    samples = errors @ np.linalg.inv(np.eye(D) - weights)
    order = rng.permutation(D)
    weights = weights[np.ix_(order, order)]
    dag = np.zeros((D, D), dtype=np.int8)
    dag[weights != 0] = _TAIL
    dag.T[weights != 0] = _ARROW
    return samples[:, order], dag


def _differs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per variable pair ``i < j``: whether the edge of ``a`` differs from the edge of ``b``."""
    upper = np.triu_indices(a.shape[0], 1)
    return ((a != b) | (a.T != b.T))[upper]


def scores(marks: np.ndarray, dag: np.ndarray) -> dict:
    """The paper's direction counts and the three pair counts of ``marks`` against ``dag``."""
    from andrey.core.orient import dag2cpdag

    marks = np.asarray(marks, dtype=np.int8)
    cpdag = np.asarray(dag2cpdag(dag.copy()), dtype=np.int8)
    directed = (marks == _TAIL) & (marks.T == _ARROW)
    truth = (dag == _TAIL) & (dag.T == _ARROW)
    from_dag, from_cpdag = _differs(marks, dag), _differs(marks, cpdag)
    return {
        "correct": int((directed & truth).sum()),
        "directed": int(directed.sum()),
        "true_edges": int(truth.sum()),
        "shd": int((from_dag & from_cpdag).sum()),
        "shd_cpdag": int(from_cpdag.sum()),
        "shd_dag": int(from_dag.sum()),
        "empty_shd": int(truth.sum()),
    }


def parse_cell(text: str) -> tuple[float, float]:
    """The setting ``"qs:qw"`` names, which must be one of :data:`CELLS`."""
    qs, _, qw = text.partition(":")
    cell = (float(qs), float(qw))
    if cell not in CELLS:
        raise ValueError(f"{text!r} is not a setting of the paper; the settings are {CELLS}")
    return cell


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _read(out: Path) -> list[dict]:
    path = out / RECORDS
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def rescore(out: Path, meta: dict) -> list[dict]:
    """Recompute every record's scores from its stored marks and rewrite the records."""
    if meta.get("sampler") != SAMPLER:
        raise SystemExit(
            f"--rescore: the run's data came from {meta.get('sampler', 'an earlier sampler')}, "
            f"not {SAMPLER}; rescore it with the build that recorded it"
        )
    records = _read(out)
    for record in records:
        if record.get("marks") is None:
            continue
        _, dag = sample(record["qs"], record["qw"], record["n"], record["seed"])
        record.update(scores(np.asarray(record["marks"], dtype=np.int8), dag))
    (out / RECORDS).write_text("".join(json.dumps(r) + "\n" for r in records))
    return records


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AGES, Andrey against pcalg, CPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--cells",
        nargs="+",
        default=[f"{qs}:{qw}" for qs, qw in CELLS],
        help="settings as qs:qw (default: the paper's 23)",
    )
    p.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=list(ROWS),
        help="sample sizes (default: the paper's)",
    )
    p.add_argument("--seeds", type=int, default=20, help="datasets per setting (default: 20)")
    p.add_argument("--first-seed", type=int, default=0, help="the first seed (default: 0)")
    p.add_argument("--part", type=int, default=0, help="run every --parts-th setting from this one")
    p.add_argument("--parts", type=int, default=1, help="parts the settings are split into")
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
        help="run one setting, 1,000 rows, one seed, and a 300 s cap, replacing those flags",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue the run already in --out: skip the fits whose records are on disk",
    )
    p.add_argument(
        "--rescore",
        action="store_true",
        help="fit nothing: recompute the scores of the run already in --out from its stored marks",
    )
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE)
    try:
        cells = [parse_cell(text) for text in args.cells][args.part :: args.parts]
    except ValueError as error:
        p.error(str(error))

    field = SOLUTIONS
    if args.solutions is not None:
        chosen = set(args.solutions.split(","))
        known = {a.name for a in SOLUTIONS}
        if chosen - known:
            p.error(
                f"unknown solutions {sorted(chosen - known)}; the solutions are {sorted(known)}"
            )
        field = [a for a in SOLUTIONS if a.name in chosen]

    out = Path(args.out)
    meta_path = out / "run_meta.json"
    if args.rescore:
        if not (out / RECORDS).exists():
            raise SystemExit(f"--rescore: {out / RECORDS} holds no records")
        records = rescore(out, json.loads(meta_path.read_text()))
        _log(f"rescored    : {len(records)} records -> {out / RECORDS}")
        _summarize(records)
        return 0
    if (out / RECORDS).exists() and not args.resume:
        raise SystemExit(f"{out} already holds a run; pass --resume to continue it")
    out.mkdir(parents=True, exist_ok=True)

    needs_r = any(a.package == "pcalg" for a in field)
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in field],
        "params": {a.name: dict(a.params()) for a in field},
        "r environment": rsession.check_environment("pcalg") if needs_r else {},
        "sampler": SAMPLER,
        "d": D,
        "strong": list(STRONG),
        "weak": list(WEAK),
        "variance": list(VARIANCE),
        "cells": [list(cell) for cell in cells],
        "rows": args.rows,
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
    _log(f"=== AGES, Andrey and pcalg -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:12s}: {value}")
    _log(f"{'git_sha':12s}: {prov['git_sha']}")
    _log("")

    done = {(r["qs"], r["qw"], r["n"], r["seed"], r["solution"], r["repeat"]) for r in _read(out)}
    start = time.perf_counter()
    with (out / RECORDS).open("a") as sink:
        for (qs, qw), n in ((cell, n) for cell in cells for n in args.rows):
            walls: dict[str, list[float]] = {a.name: [] for a in field}
            for seed in range(args.first_seed, args.first_seed + args.seeds):
                X, dag = sample(qs, qw, n, seed)
                for adapter in field:
                    key = (qs, qw, n, seed, adapter.name)
                    if all((*key, r) in done for r in range(args.repeats)):
                        continue
                    params = adapter.params()
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
                    for repeat, m in enumerate(measures):
                        if (*key, repeat) in done:
                            continue
                        record = {
                            "qs": qs,
                            "qw": qw,
                            "d": D,
                            "n": n,
                            "seed": seed,
                            "solution": adapter.name,
                            "package": adapter.package,
                            "package_version": adapter.package_version,
                            "repeat": repeat,
                            "status": m.get("status"),
                            "wall_s": m.get("wall_s"),
                            "cpu_s": m.get("cpu_s"),
                            "peak_rss_mb": m.get("peak_rss_mb"),
                            "error_text": m.get("error_text") or m.get("stderr"),
                            "params": params,
                            "machine_id": args.machine_id,
                            "marks": None,
                        }
                        if m.get("status") == "ok" and m.get("adj") is not None:
                            marks = adapter.to_structure(np.asarray(m["adj"])).to_numpy()
                            record["marks"] = np.asarray(marks, dtype=int).tolist()
                            record.update(scores(marks, dag))
                            walls[adapter.name].append(m["wall_s"])
                        sink.write(json.dumps(record, default=str) + "\n")
                        sink.flush()
            cost = "  ".join(
                f"{name}={statistics.median(w):.3g}s" if w else f"{name}=no new fit finished"
                for name, w in walls.items()
            )
            _log(f"[qs={qs} qw={qw} n={n}] {cost} | elapsed {time.perf_counter() - start:.0f}s")
    _summarize(_read(out))
    return 0


def _summarize(records: list[dict]) -> None:
    """Print, per setting and solution, finished fits, median time, the measures, and agreement.

    Precision and recall pool the counts of a setting's datasets; ``same`` counts the datasets
    where the solution returned the graph pcalg returned.
    """
    if not records:
        return
    ok = [r for r in records if r["status"] == "ok"]
    reference = {
        (r["qs"], r["qw"], r["n"], r["seed"]): r["marks"]
        for r in ok
        if r["solution"] == "pcalg.ages"
    }
    _log("")
    _log(
        f"{'solution':22s} {'qs':>4s} {'qw':>4s} {'n':>6s} {'ok/all':>7s} {'wall_s':>8s} "
        f"{'prec':>6s} {'recall':>6s} {'shd':>6s} {'empty':>6s} {'same':>6s}"
    )
    for qs, qw, n in sorted({(r["qs"], r["qw"], r["n"]) for r in records}):
        for name in sorted({r["solution"] for r in records}):
            mine = [
                r for r in records if (r["qs"], r["qw"], r["n"], r["solution"]) == (qs, qw, n, name)
            ]
            fit = [r for r in mine if r["status"] == "ok"]
            if not fit:
                continue
            same = sum(reference.get((qs, qw, n, r["seed"])) == r["marks"] for r in fit)
            correct, directed, true_edges = (
                sum(r[k] for r in fit) for k in ("correct", "directed", "true_edges")
            )
            _log(
                f"{name:22s} {qs:4.1f} {qw:4.1f} {n:6d} {len(fit):3d}/{len(mine):<3d} "
                f"{statistics.median(r['wall_s'] for r in fit):8.3g} "
                f"{correct / directed if directed else float('nan'):6.3f} "
                f"{correct / true_edges if true_edges else float('nan'):6.3f} "
                f"{statistics.mean(r['shd'] for r in fit):6.3g} "
                f"{statistics.mean(r['empty_shd'] for r in fit):6.3g} {same:3d}/{len(fit):<2d}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
