#!/usr/bin/env python
"""Benchmark lasso Granger causality on CPU: Andrey against causal-learn.

Runs Andrey's ``granger``, serially and on worker threads, and causal-learn's
``Granger.granger_lasso`` on vector-autoregressive
time series with known lagged edges: ``x_t = A_1 x_{t-1} + ... + A_p x_{t-p} + e_t`` with Gaussian
noise and no instantaneous effects, drawn by ``andrey.data`` (:func:`sample`). Every variable keeps
its own lags, and each other variable enters a lag with probability ``LAG_DEGREE / (d - 1)``. The
lag matrices are scaled so the process is stationary. Each variable is z-scored before the fits,
so both packages fit the same standardized series, and both fit with the same ``maxlag`` and ``cv``.

Each fit runs isolated, with the variables' count ``d`` as the ladder, under a per-fit wall cap; a
fit past the cap is recorded as ``timeout`` and has no scores. A fit's lagged edges, the nonzero
coefficients at lags 1 to ``maxlag``, are scored against the true ones (:func:`lag_scores`). One
JSON line per fit goes to ``<out>/records.jsonl``, with the fitted coefficients, so ``--rescore``
recomputes the scores without fitting.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/granger_var.py --out outputs/granger-var
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

from andrey.data.functional import var
from andrey.data.graphs import erdos_renyi
from andrey.data.noise import gaussian
from andrey.data.scm import SCM
from andrey_bench.adapters.andrey_granger import andrey_granger_adapters
from andrey_bench.adapters.causal_learn_granger import CausalLearnGranger
from andrey_bench.runner import available_cores, run_task, split_cores

LANE = [*andrey_granger_adapters(), CausalLearnGranger()]

#: Lags of the true process; the solutions fit ``maxlag`` lags, the same number by default.
LAGS = 2

#: Expected number of other variables entering each lag of a variable.
LAG_DEGREE = 2.0

#: What both packages are given: each variable z-scored.
INPUT = "standardized"

#: Smallest run that touches every solution: one small series.
SMOKE = {"ladder": [5], "seeds": 1, "n": 300, "cap_wall_s": 300.0}

RECORDS = "records.jsonl"


def sample(d: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(data, truth)`` for one series: ``n`` time steps of ``d`` variables.

    ``data`` is z-scored per variable: causal-learn fits the series as given and Andrey
    standardizes it first, so both get the same standardized input. ``truth[k - 1, i, j]`` is true
    when ``x_i(t - k) -> x_j(t)``, for ``k`` from 1 to :data:`LAGS`; scaling a variable keeps it.
    """
    scm = SCM(
        graph=erdos_renyi(d, avg_degree=0.0),  # no instantaneous effects: lagged edges only
        functional=var(n_lags=LAGS, lag_density=min(1.0, LAG_DEGREE / (d - 1))),
        noise=gaussian(),
    )
    ds = scm.sample(n=n, seed=seed)
    weights = np.asarray(ds.graph.lag_weights)  # (1 + LAGS, d, d), [cause, effect], lag 0 first
    if weights[0].any():
        raise AssertionError("the process has instantaneous effects")
    data = np.asarray(ds.data, dtype=np.float64)
    return (data - data.mean(axis=0)) / data.std(axis=0), weights[1:] != 0


def lag_support(coeff: np.ndarray) -> np.ndarray:
    """Return the lagged edges of a ``(d, d * maxlag)`` coefficient matrix as a truth-shaped stack.

    ``coeff[i, j + d * (k - 1)]`` is the weight of ``x_j(t - k)`` on ``x_i(t)``, causal-learn's
    layout; the stack is ``[cause, effect]`` per lag, like :func:`sample`'s truth.
    """
    d = coeff.shape[0]
    return np.stack([coeff[:, d * k : d * (k + 1)].T for k in range(coeff.shape[1] // d)]) != 0


#: The columns :func:`lag_scores` returns, in order.
SCORE_COLUMNS = (
    "n_true",
    "n_est",
    "precision",
    "recall",
    "f1",
    "cross_n_true",
    "cross_precision",
    "cross_recall",
    "cross_f1",
)


def lag_scores(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Score the estimated lagged edges against the true ones.

    Both are ``(lags, d, d)`` boolean stacks, padded with empty lags to the longer one. The
    ``cross_*`` scores leave out each variable's edges from its own past, which every true process
    holds. A ratio with no denominator is ``NaN``, not 1.
    """
    lags = max(len(estimate), len(truth))
    est, true = (
        np.concatenate([s, np.zeros((lags - len(s), *s.shape[1:]), bool)])
        for s in (estimate, truth)
    )
    scores: dict[str, float] = {"n_est": float(est.sum())}
    cross = ~np.eye(true.shape[1], dtype=bool)
    for prefix, mask in (("", np.ones_like(cross)), ("cross_", cross)):
        e, t = est & mask, true & mask
        hits, n_est, n_true = float((e & t).sum()), float(e.sum()), float(t.sum())
        scores[f"{prefix}n_true"] = n_true
        scores[f"{prefix}precision"] = hits / n_est if n_est else np.nan
        scores[f"{prefix}recall"] = hits / n_true if n_true else np.nan
        scores[f"{prefix}f1"] = 2 * hits / (n_est + n_true) if n_est + n_true else np.nan
    return {c: scores[c] for c in SCORE_COLUMNS}


def _log(msg: str = "") -> None:
    """Print and flush batch output immediately."""
    print(msg, flush=True)


def _read(out: Path) -> list[dict]:
    """Return the run's records, oldest first; none when it has not started."""
    path = out / RECORDS
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def rescore(out: Path) -> list[dict]:
    """Recompute every record's scores from its stored coefficients and rewrite the records."""
    meta = json.loads((out / "run_meta.json").read_text())
    if (meta["driver"], meta["lags"], meta["lag_degree"], meta.get("input")) != (
        Path(__file__).name,
        LAGS,
        LAG_DEGREE,
        INPUT,
    ):
        raise SystemExit(
            f"--rescore: {out} was drawn with other process settings than this script's"
        )
    records = _read(out)
    for record in records:
        for column in SCORE_COLUMNS:
            record.pop(column, None)
        if record["status"] == "ok":
            _, truth = sample(record["d"], record["n"], record["seed"])
            record.update(lag_scores(lag_support(np.asarray(record["coeff"])), truth))
    scratch = out / f".{RECORDS}.tmp"
    scratch.write_text("".join(json.dumps(r) + "\n" for r in records))
    os.replace(scratch, out / RECORDS)
    return records


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Lasso Granger, Andrey against causal-learn, CPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--ladder",
        type=int,
        nargs="+",
        default=[5, 10, 20, 40, 80],
        help="variable counts d (default: 5 10 20 40 80)",
    )
    p.add_argument("--seeds", type=int, default=5, help="series per rung (default: 5)")
    p.add_argument("--n", type=int, default=1000, help="time steps per series (default: 1000)")
    p.add_argument("--maxlag", type=int, default=LAGS, help=f"lags fitted (default: {LAGS})")
    p.add_argument("--cv", type=int, default=5, help="cross-validation folds (default: 5)")
    p.add_argument("--repeats", type=int, default=1, help="timed repeats per fit (default: 1)")
    p.add_argument("--warmup", type=int, default=0, help="untimed fits per repeat (default: 0)")
    p.add_argument("--cap-wall-s", type=float, default=1800.0, help="wall cap per repeat, seconds")
    p.add_argument("--cap-mem-mb", type=float, default=None, help="memory cap per repeat, MB")
    p.add_argument(
        "--cores",
        type=int,
        default=available_cores(),
        help="core budget per fit (default: every core this job is allowed)",
    )
    p.add_argument(
        "--machine-id",
        default=socket.gethostname(),
        help="machine id stamped on every record (default: this host's name)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="run d=5, one seed, 300 time steps, and a 300 s cap, replacing those flags",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue the run already in --out: skip the fits whose records are on disk",
    )
    p.add_argument(
        "--rescore",
        action="store_true",
        help="fit nothing: recompute the scores of the run already in --out from its stored "
        "coefficients",
    )
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE)

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
    out.mkdir(parents=True, exist_ok=True)
    params = {"maxlag": args.maxlag, "cv": args.cv}
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in LANE],
        "lags": LAGS,
        "lag_degree": LAG_DEGREE,
        "input": INPUT,
        "ladder": args.ladder,
        "seeds": args.seeds,
        "n": args.n,
        "params": params,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "cap_wall_s": args.cap_wall_s,
        "cap_mem_mb": args.cap_mem_mb,
        "cores": args.cores,
        "machine_id": args.machine_id,
    }
    meta_path = out / "run_meta.json"
    if args.resume and meta_path.exists() and json.loads(meta_path.read_text()) != run_meta:
        raise SystemExit(f"--resume: {meta_path} differs from these flags; pass the first run's")
    meta_path.write_text(json.dumps(run_meta, indent=2))
    _log(f"=== Lasso Granger, Andrey and causal-learn -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:12s}: {value}")
    _log("")

    done = {(r["d"], r["seed"], r["solution"], r["repeat"]) for r in _read(out)}
    start = time.perf_counter()
    with (out / RECORDS).open("a") as sink:
        for d in args.ladder:
            walls: dict[str, list[float]] = {a.name: [] for a in LANE}
            for seed in range(args.seeds):
                data, truth = sample(d, args.n, seed)
                for adapter in LANE:
                    if all((d, seed, adapter.name, r) in done for r in range(args.repeats)):
                        continue
                    threads, num_workers = split_cores(args.cores, adapter.mode)
                    measures = run_task(
                        adapter,
                        data,
                        params,
                        cap_wall_s=args.cap_wall_s,
                        cap_mem_mb=args.cap_mem_mb,
                        repeats=args.repeats,
                        warmup=args.warmup,
                        threads=threads,
                        num_workers=num_workers,
                    )
                    for repeat, m in enumerate(measures):
                        if (d, seed, adapter.name, repeat) in done:  # kept from an earlier run
                            continue
                        record = {
                            "d": d,
                            "n": args.n,
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
                        }
                        if m.get("status") == "ok" and m.get("adj") is not None:
                            coeff = np.asarray(m["adj"], dtype=np.float64)
                            record["coeff"] = coeff.tolist()
                            record.update(lag_scores(lag_support(coeff), truth))
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
    """Print, per rung and solution, the finished fits, their median time, and mean scores.

    ``f1`` scores every lagged edge, ``cross f1`` the edges between two different variables.
    ``agree`` is the share of series, among those every solution finished, on which they all
    return the same lagged edges.
    """
    if not records:
        return
    # A run with no finished fit has no coefficient or score columns; they read as missing.
    columns = ["d", "seed", "repeat", "solution", "status", "wall_s", "coeff", *SCORE_COLUMNS]
    table = pd.DataFrame(records).reindex(columns=columns)
    ok = table[table["status"] == "ok"].copy()
    ok["support"] = [str(lag_support(np.asarray(c)).tolist()) for c in ok["coeff"]]
    _log("")
    _log(
        f"{'d':>4s} {'solution':24s} {'ok/all':>7s} {'wall_s':>8s} {'prec':>6s} {'recall':>6s} "
        f"{'f1':>6s} {'cross f1':>8s} {'agree':>6s}"
    )
    for d, rung in table.groupby("d", sort=True):
        mine = ok[ok["d"] == d]
        same = mine.groupby(["seed", "repeat"])["support"].agg(["nunique", "size"])
        both = same[same["size"] == len(LANE)]
        agree = (both["nunique"] == 1).mean() if len(both) else float("nan")
        for name, rows in rung.groupby("solution", sort=True):
            fits = mine[mine["solution"] == name]
            _log(
                f"{d:4d} {name:24s} {len(fits):3d}/{len(rows):<3d} {fits['wall_s'].median():8.3g} "
                f"{fits['precision'].mean():6.3f} {fits['recall'].mean():6.3f} "
                f"{fits['f1'].mean():6.3f} {fits['cross_f1'].mean():8.3f} {agree:6.3f}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
