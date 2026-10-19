#!/usr/bin/env python
"""Benchmark the nonlinear CI tests query by query: Andrey's against the reference packages'.

Every contestant answers the same queries on the same seeded draws. A draw has columns ``x, y,
z_1..z_k`` (the draw of ``qa/ci_calibration.py``): ``x = tanh(2 s) + e`` and ``y = s^2 + e'`` with
``s`` the scaled sum of the ``z``, so ``x`` is independent of ``y`` given ``z`` (with ``k = 0``,
``x`` and ``y`` are independent noise); one noise term added to both (``--shared``) makes them
dependent. Each query tests ``x`` against ``y`` given every ``z``, on ``--null`` independent draws
and ``--alt`` dependent ones, for each row count and ``k``.

``records.jsonl`` holds one line per contestant and draw: the p-value, the seconds of the test
alone (building it and answering the query, without file input and output), and ``status``
(``ok``, or ``timeout`` for draws a cell's cap left unanswered). ``summary.csv`` gives, per
contestant, row count, and ``k``: the level (the share of independent draws rejected at
``--alpha``), the power (the share of dependent draws rejected), the Kolmogorov-Smirnov distance
of the independent draws' p-values to the uniform, the share of draws on which a reference
decides as Andrey's test of the same kind, the median absolute p-value difference to it, and the
median seconds per query. A reference is measured, not a target.

Contestants and their environments:

- ``andrey.<test>``: Andrey's ``kci``, ``rcot``, ``cmiknn``, and ``gpdc`` in this process;
- ``causal-learn.kci``, ``.rcot``, ``.fastkci``: causal-learn in this interpreter, through
  ``andrey_bench/ci_query_worker.py``;
- ``tigramite.cmiknn``, ``.gpdc``: tigramite through the same worker in the interpreter
  ``ANDREY_BENCH_TIGRAMITE_PYTHON`` names (``benchmarks/start_tigramite.sh``);
- ``rcit.rcot``, ``rcit.kcit``: RCIT's RCoT and KCIT through ``andrey_bench/ci_query_worker.R``,
  with the R that ``benchmarks/start_r.sh --env`` sets up.

Every test runs on one thread at Andrey's default settings; the workers map them to each package's
names.

    $ANDREY_BENCH_PYTHON benchmarks/nonlinear_ci_queries.py --out outputs/ci-queries \\
        --contestants andrey.rcot,causal-learn.rcot,rcit.rcot --rows 500 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import time
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np

#: Contestant -> (package, test). Andrey's run in process; the others in a worker.
CONTESTANTS: dict[str, tuple[str, str]] = {
    "andrey.kci": ("andrey", "kci"),
    "andrey.rcot": ("andrey", "rcot"),
    "andrey.cmiknn": ("andrey", "cmiknn"),
    "andrey.gpdc": ("andrey", "gpdc"),
    "causal-learn.kci": ("causal-learn", "kci"),
    "causal-learn.rcot": ("causal-learn", "rcot"),
    "causal-learn.fastkci": ("causal-learn", "fastkci"),
    "tigramite.cmiknn": ("tigramite", "cmiknn"),
    "tigramite.gpdc": ("tigramite", "gpdc"),
    "rcit.rcot": ("RCIT", "rcot"),
    "rcit.kcit": ("RCIT", "kcit"),
}

#: The Andrey test each contestant's decisions are compared with.
ANDREY_PEER = {"kci": "kci", "rcot": "rcot", "fastkci": "kci", "kcit": "kci"}

TIGRAMITE_PYTHON = "ANDREY_BENCH_TIGRAMITE_PYTHON"
HERE = Path(__file__).resolve().parent
ONE_THREAD = {
    v: "1"
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
}

#: Smallest run that reaches every contestant.
SMOKE = {"rows": [100], "k": [0, 2], "null": 3, "alt": 2, "cap_cell_s": 300.0}


def draw(seed: int, n: int, k: int, shared: float) -> np.ndarray:
    """One draw with columns ``x, y, z_1..z_k``; ``shared`` scales the noise ``x`` and ``y``
    share. The same generator as ``qa/ci_calibration.py``."""
    rng = np.random.default_rng([seed, n, k])
    z = rng.standard_normal((n, k))
    s = z.sum(axis=1) / np.sqrt(k) if k else np.zeros(n)
    u = rng.standard_normal(n)
    x = np.tanh(2.0 * s) + 0.4 * rng.standard_normal(n) + shared * u
    y = s**2 + 0.4 * rng.standard_normal(n) + shared * u
    return np.column_stack([x, y, z])


def _andrey(test: str, draws: list[np.ndarray], cap_s: float) -> list[dict]:
    """Answer each draw's query with Andrey's ``test`` in this process, one BLAS thread."""
    from threadpoolctl import threadpool_limits

    import andrey
    from andrey.core.ci import make_indep_test

    rows, spent = [], 0.0
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        warnings.simplefilter("ignore", andrey.SampleSizeWarning)
        for data in draws:
            if spent > cap_s:
                rows.append({"status": "timeout"})
                continue
            start = time.perf_counter()
            p = make_indep_test(test, data)(0, 1, tuple(range(2, data.shape[1])))
            seconds = time.perf_counter() - start
            spent += seconds
            rows.append({"status": "ok", "p": float(p), "seconds": seconds})
    return rows


def _worker(package: str, test: str, draws: list[np.ndarray], cell: Path, cap_s: float) -> list:
    """Answer each draw's query in a worker process for ``package``; draws it did not reach
    within ``cap_s`` seconds are timeouts."""
    cell.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **ONE_THREAD}
    if package == "RCIT":
        for i, data in enumerate(draws):
            names = ["x", "y", *(f"z{j}" for j in range(1, data.shape[1] - 1))]
            np.savetxt(cell / f"d{i}.csv", data, delimiter=",", header=",".join(names), comments="")
        out = cell / f"{package}.{test}.csv"
        cmd = ["Rscript", "--vanilla", str(HERE / "andrey_bench" / "ci_query_worker.R"), test]
        cmd += [str(cell), str(out)]
    else:
        np.savez(cell / "draws.npz", **{f"d{i}": data for i, data in enumerate(draws)})
        out = cell / f"{package}.{test}.jsonl"
        python = os.environ[TIGRAMITE_PYTHON] if package == "tigramite" else sys.executable
        cmd = [python, str(HERE / "andrey_bench" / "ci_query_worker.py"), package, test]
        cmd += [str(cell / "draws.npz"), str(out)]
    try:
        subprocess.run(cmd, env=env, timeout=cap_s, check=True, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        pass
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{package} {test} failed:\n{exc.stderr[-2000:]}") from None
    answered = {}
    if out.exists():
        if package == "RCIT":
            with open(out) as f:
                answered = {int(r["draw"]): r for r in csv.DictReader(f)}
        else:
            answered = {r["draw"]: r for r in map(json.loads, out.read_text().splitlines())}
    return [
        {"status": "ok", "p": float(answered[i]["p"]), "seconds": float(answered[i]["seconds"])}
        if i in answered
        else {"status": "timeout"}
        for i in range(len(draws))
    ]


def versions(contestants: list[str]) -> dict[str, str]:
    """The version of each package the contestants use, read without importing it here."""
    out = {}
    for package in sorted({CONTESTANTS[c][0] for c in contestants}):
        if package == "tigramite":
            code = "from importlib import metadata; print(metadata.version('tigramite'))"
            run = subprocess.run([os.environ[TIGRAMITE_PYTHON], "-c", code], capture_output=True)
            out[package] = run.stdout.decode().strip() or "unknown"
        elif package == "RCIT":
            from andrey_bench import rsession

            out[package] = rsession.package_version("RCIT")
        else:
            out[package] = metadata.version("andrey-core" if package == "andrey" else package)
    return out


def summary(records: list[dict], alpha: float) -> list[dict]:
    """One row per contestant, row count, and ``k``; see the module docstring."""
    from scipy.stats import kstest

    by_key: dict[tuple, dict] = {}
    for r in records:
        by_key.setdefault((r["contestant"], r["n"], r["k"]), {})[(r["kind"], r["seed"])] = r
    rows = []
    for (name, n, k), cell in sorted(by_key.items()):
        ok = {key: r for key, r in cell.items() if r["status"] == "ok"}
        null = [r["p"] for (kind, _), r in ok.items() if kind == "null"]
        alt = [r["p"] for (kind, _), r in ok.items() if kind == "alt"]
        row = {
            "contestant": name,
            "n": n,
            "k": k,
            "answered": len(ok),
            "draws": len(cell),
            "level": float(np.mean(np.asarray(null) <= alpha)) if null else None,
            "power": float(np.mean(np.asarray(alt) <= alpha)) if alt else None,
            "ks_uniform": float(kstest(null, "uniform").statistic) if len(null) > 1 else None,
            "seconds": statistics.median(r["seconds"] for r in ok.values()) if ok else None,
            "agreement": None,
            "median_abs_dp": None,
        }
        package, test = CONTESTANTS[name]
        peer = by_key.get((f"andrey.{ANDREY_PEER.get(test, test)}", n, k), {})
        pairs = [
            (r["p"], peer[key]["p"])
            for key, r in ok.items()
            if package != "andrey" and peer.get(key, {}).get("status") == "ok"
        ]
        if pairs:
            mine, theirs = np.asarray(pairs).T
            row["agreement"] = float(np.mean((mine <= alpha) == (theirs <= alpha)))
            row["median_abs_dp"] = float(np.median(np.abs(mine - theirs)))
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--contestants",
        default=",".join(CONTESTANTS),
        help="comma-separated contestant names (default: every one)",
    )
    p.add_argument("--rows", type=int, nargs="+", default=[200, 500, 1000, 2000])
    p.add_argument("--k", type=int, nargs="+", default=[0, 1, 2, 3], help="conditioning sizes")
    p.add_argument("--null", type=int, default=100, help="independent draws per cell")
    p.add_argument("--alt", type=int, default=20, help="dependent draws per cell")
    p.add_argument("--shared", type=float, default=0.5, help="shared noise of dependent draws")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument(
        "--cap-cell-s", type=float, default=1800.0, help="wall cap per contestant and cell"
    )
    p.add_argument("--smoke", action="store_true", help="tiny cells: " + json.dumps(SMOKE))
    p.add_argument("--resume", action="store_true", help="skip cells with records on disk")
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE)
    chosen = args.contestants.split(",")
    if unknown := sorted(set(chosen) - set(CONTESTANTS)):
        p.error(f"unknown contestants {unknown}; the contestants are {list(CONTESTANTS)}")
    packages = {CONTESTANTS[c][0] for c in chosen}
    if "tigramite" in packages and not os.environ.get(TIGRAMITE_PYTHON):
        p.error(f"tigramite's contestants need {TIGRAMITE_PYTHON} (benchmarks/start_tigramite.sh)")
    if "RCIT" in packages and not shutil.which("Rscript"):
        p.error("RCIT's contestants need Rscript: export what benchmarks/start_r.sh --env prints")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / "records.jsonl"
    done = set()
    if args.resume and records_path.exists():
        done = {(r["contestant"], r["n"], r["k"]) for r in map(json.loads, open(records_path))}
    elif records_path.exists():
        p.error(f"{records_path} exists; pass --resume to continue that run")
    cpu = next(
        (ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if "model name" in ln), ""
    )
    meta = {
        "driver": Path(__file__).name,
        "args": vars(args),
        "versions": versions(chosen),
        "host": socket.gethostname(),
        "cpu": cpu,
        "cores": len(os.sched_getaffinity(0)),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)

    for n in args.rows:
        for k in args.k:
            seeds = [("null", s, 0.0) for s in range(args.null)]
            seeds += [("alt", s, args.shared) for s in range(args.alt)]
            draws = [draw(s, n, k, shared) for _, s, shared in seeds]
            for name in chosen:
                if (name, n, k) in done:
                    continue
                package, test = CONTESTANTS[name]
                start = time.perf_counter()
                if package == "andrey":
                    answers = _andrey(test, draws, args.cap_cell_s)
                else:
                    cell = out / "cells" / f"n{n}_k{k}"
                    answers = _worker(package, test, draws, cell, args.cap_cell_s)
                with open(records_path, "a") as f:
                    for (kind, seed, _), answer in zip(seeds, answers):
                        record = {"contestant": name, "n": n, "k": k, "kind": kind, "seed": seed}
                        f.write(json.dumps({**record, **answer}) + "\n")
                print(f"[n={n} k={k}] {name}: {time.perf_counter() - start:.1f}s", flush=True)

    records = [json.loads(line) for line in open(records_path)]
    rows = summary(records, args.alpha)
    with open(out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            " ".join(
                f"{key}={value:.3g}" if isinstance(value, float) else f"{key}={value}"
                for key, value in row.items()
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
