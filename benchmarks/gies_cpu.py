#!/usr/bin/env python
"""Benchmark GIES on CPU: Andrey against pcalg, on interventional data.

Runs Andrey's `gies` and pcalg's `gies` on linear-Gaussian data that mixes observation with
interventions, simulated as in Hauser and Buhlmann (2012, Section 5.2.1) (:func:`sample`):

- a random DAG on ``d`` variables whose vertex degrees have mean :data:`DEGREE`;
- edge weights drawn from ``U([-1, -0.1] U [0.1, 1])`` and error variances from ``U(0.5, 1)``,
  rescaled so every variable has unit variance without interventions;
- ``round(target_share * d)`` single-variable intervention targets, chosen at random; each target's
  rows set its variable from ``N(2, 0.2^2)``;
- ``rows_per_variable * d`` rows, split evenly between no intervention and each target.

Each fit gets the samples and the 0/1 intervention matrix side by side
(:func:`andrey_bench.adapters.andrey_gies.stack`) and runs isolated, under a per-fit wall cap; a fit
past the cap is a ``timeout`` with no scores. Its essential graph is scored against the true one,
the interventional essential graph of the generating DAG: ``shd`` counts the variable pairs whose
edge differs. ``bic`` is the graph's BIC on its data in Andrey's deviance units, lower is better,
and ``truth_bic`` the generating DAG's; ``empty_shd`` is the empty graph's ``shd``, the true edge
count. One JSON line per fit goes to ``<out>/records.jsonl`` with the graph's endpoint marks, so
``--rescore`` recomputes the scores without fitting; ``<out>/provenance.json`` records the build and
the machine.

Load R with pcalg as the README's R lane describes before running pcalg's solution.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/gies_cpu.py --out outputs/gies-cpu
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
from andrey_bench.adapters.andrey_gies import AndreyGIES, stack
from andrey_bench.adapters.pcalg_gies import PcalgGIES
from andrey_bench.runner import available_cores, run_task, split_cores

#: The solutions: Andrey, then the reference.
SOLUTIONS = [AndreyGIES(), PcalgGIES()]

#: Mean vertex degree of the generating DAG.
DEGREE = 2.0

#: The level of an intervened variable: mean and standard deviation.
LEVEL = (2.0, 0.2)

#: Smallest run that touches every solution: one small dataset.
SMOKE = {"ladder": [10], "seeds": 1, "cap_wall_s": 300.0}

RECORDS = "records.jsonl"


def sample(
    d: int, seed: int, *, rows_per_variable: int, target_share: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(samples, interventions, dag)`` for one dataset.

    ``interventions[r, j]`` is True when row ``r`` set variable ``j`` by intervention; ``dag`` holds
    the generating DAG's endpoint marks.
    """
    rng = np.random.default_rng([d, seed])
    # The causal order is a uniformly random permutation of the columns and the targets are drawn
    # uniformly, so the column order carries nothing of the graph and needs no shuffle step.
    order = rng.permutation(d)
    B = np.zeros((d, d))  # B[i, j]: the weight of j -> i
    for a in range(d):
        for b in range(a + 1, d):
            if rng.random() < DEGREE / max(d - 1, 1):
                B[order[b], order[a]] = rng.uniform(0.1, 1.0) * rng.choice((-1.0, 1.0))
    variance = rng.uniform(0.5, 1.0, d)
    inverse = np.linalg.inv(np.eye(d) - B)
    scale = 1.0 / np.sqrt(np.diag(inverse @ np.diag(variance) @ inverse.T))
    B = scale[:, None] * B / scale[None, :]
    variance = scale**2 * variance

    n = rows_per_variable * d
    targets = rng.choice(d, size=round(target_share * d), replace=False)
    setting = np.arange(n) % (len(targets) + 1)  # 0: no intervention; k: the k-th target
    interventions = np.zeros((n, d), dtype=bool)
    for k, target in enumerate(targets, start=1):
        interventions[setting == k, target] = True
    X = np.zeros((n, d))
    for j in order:
        X[:, j] = X @ B[j] + rng.normal(0.0, np.sqrt(variance[j]), n)
        hit = interventions[:, j]
        X[hit, j] = rng.normal(*LEVEL, hit.sum())

    dag = np.zeros((d, d), dtype=np.int8)
    child, parent = np.nonzero(B)
    dag[parent, child], dag[child, parent] = 1, 2  # parent -> child
    return X, interventions, dag


def essential_graph(dag: np.ndarray, interventions: np.ndarray) -> np.ndarray:
    """The interventional essential graph of ``dag`` under the rows' intervention targets."""
    from andrey.search.gies import cut_pairs, replace_unprotected

    return replace_unprotected(dag.copy(), cut_pairs(interventions))


def graph_bic(marks: np.ndarray, X: np.ndarray, interventions: np.ndarray) -> float:
    """BIC of an essential graph in deviance units, through any DAG of its class."""
    from andrey.core.orient import pdag2dag
    from andrey.search.gies import InterventionalBICScore

    dag = pdag2dag(marks)
    score = InterventionalBICScore(X, interventions)
    return float(sum(score.score(j, np.nonzero(dag[j] == 2)[0]) for j in range(dag.shape[0])))


def scores(marks: np.ndarray, X: np.ndarray, interventions: np.ndarray, dag: np.ndarray) -> dict:
    """``shd`` against the true essential graph, ``bic``, ``truth_bic``, and ``empty_shd``."""
    from andrey import GraphStructure
    from andrey.metrics import shd

    truth = essential_graph(dag, interventions)
    estimate = GraphStructure.from_numpy(np.asarray(marks, dtype=np.int8), kind="cpdag")
    return {
        "shd": int(shd(estimate, GraphStructure.from_numpy(truth, kind="cpdag"))),
        "bic": graph_bic(marks, X, interventions),
        "truth_bic": graph_bic(dag, X, interventions),
        "empty_shd": int(np.count_nonzero(np.triu(truth != 0))),
    }


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _read(out: Path) -> list[dict]:
    path = out / RECORDS
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def rescore(out: Path, meta: dict) -> list[dict]:
    """Recompute every record's scores from its stored marks and rewrite the records."""
    records = _read(out)
    for record in records:
        if record.get("marks") is None:
            continue
        X, interventions, dag = sample(
            record["d"],
            record["seed"],
            rows_per_variable=meta["rows_per_variable"],
            target_share=meta["target_share"],
        )
        record.update(scores(np.asarray(record["marks"], dtype=np.int8), X, interventions, dag))
    (out / RECORDS).write_text("".join(json.dumps(r) + "\n" for r in records))
    return records


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="GIES, Andrey against pcalg, CPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--ladder",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100],
        help="variable counts d (default: 10 20 50 100)",
    )
    p.add_argument("--seeds", type=int, default=5, help="datasets per rung (default: 5)")
    p.add_argument(
        "--rows-per-variable", type=int, default=10, help="rows per variable (default: 10)"
    )
    p.add_argument(
        "--target-share",
        type=float,
        default=0.4,
        help="intervention targets per variable, one variable each (default: 0.4)",
    )
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
        help="run d=10, one seed, and a 300 s cap, replacing those flags",
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
        "degree": DEGREE,
        "level": list(LEVEL),
        "ladder": args.ladder,
        "seeds": args.seeds,
        "rows_per_variable": args.rows_per_variable,
        "target_share": args.target_share,
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
    _log(f"=== GIES, Andrey and pcalg -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:12s}: {value}")
    _log(f"{'git_sha':12s}: {prov['git_sha']}")
    _log("")

    done = {(r["d"], r["seed"], r["solution"], r["repeat"]) for r in _read(out)}
    start = time.perf_counter()
    with (out / RECORDS).open("a") as sink:
        for d in args.ladder:
            walls: dict[str, list[float]] = {a.name: [] for a in field}
            for seed in range(args.seeds):
                X, interventions, dag = sample(
                    d,
                    seed,
                    rows_per_variable=args.rows_per_variable,
                    target_share=args.target_share,
                )
                for adapter in field:
                    if all((d, seed, adapter.name, r) in done for r in range(args.repeats)):
                        continue
                    params = adapter.params()
                    threads, num_workers = split_cores(args.cores, adapter.mode)
                    measures = run_task(
                        adapter,
                        stack(X, interventions),
                        params,
                        cap_wall_s=args.cap_wall_s,
                        cap_mem_mb=args.cap_mem_mb,
                        repeats=args.repeats,
                        warmup=args.warmup,
                        threads=threads,
                        num_workers=num_workers,
                    )
                    for repeat, m in enumerate(measures):
                        if (d, seed, adapter.name, repeat) in done:
                            continue
                        record = {
                            "d": d,
                            "n": X.shape[0],
                            "seed": seed,
                            "targets": int(interventions.any(axis=0).sum()),
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
                            record.update(scores(marks, X, interventions, dag))
                            walls[adapter.name].append(m["wall_s"])
                        sink.write(json.dumps(record, default=str) + "\n")
                        sink.flush()
            cost = "  ".join(
                f"{name}={statistics.median(w):.3g}s" if w else f"{name}=no new fit finished"
                for name, w in walls.items()
            )
            _log(f"[d={d}] {cost} | elapsed {time.perf_counter() - start:.0f}s")
    _summarize(_read(out))
    return 0


def _summarize(records: list[dict]) -> None:
    """Print, per rung and solution, finished fits, median time and errors, and agreement.

    ``same`` counts the datasets where the solution returned the graph pcalg returned.
    """
    if not records:
        return
    ok = [r for r in records if r["status"] == "ok"]
    reference = {(r["d"], r["seed"]): r["marks"] for r in ok if r["solution"] == "pcalg.gies"}
    _log("")
    _log(
        f"{'solution':20s} {'d':>4s} {'ok/all':>7s} {'wall_s':>8s} {'shd':>6s} {'empty':>6s} "
        f"{'bic-truth':>10s} {'same':>6s}"
    )
    for d in sorted({r["d"] for r in records}):
        for name in sorted({r["solution"] for r in records}):
            mine = [r for r in records if r["d"] == d and r["solution"] == name]
            fit = [r for r in mine if r["status"] == "ok"]
            if not mine:
                continue
            same = sum(reference.get((r["d"], r["seed"])) == r["marks"] for r in fit)
            wall, shd, empty, gap = (
                statistics.median(values) if fit else float("nan")
                for values in (
                    [r["wall_s"] for r in fit],
                    [r["shd"] for r in fit],
                    [r["empty_shd"] for r in fit],
                    [r["bic"] - r["truth_bic"] for r in fit],
                )
            )
            _log(
                f"{name:20s} {d:4d} {len(fit):3d}/{len(mine):<3d} {wall:8.3g} {shd:6.3g} "
                f"{empty:6.3g} {gap:10.3g} {same:3d}/{len(fit):<2d}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
