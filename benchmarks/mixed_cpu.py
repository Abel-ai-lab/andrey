#!/usr/bin/env python
"""Benchmark the mixed-data tests and scores on CPU: Andrey against Tetrad and bnlearn.

Runs PC with the degenerate and the conditional Gaussian test, and GES and hill climbing with the
degenerate and the conditional Gaussian score, from Andrey; Tetrad's PC and FGES with the same four
(through its causal-cmd jar); bnlearn's ``pc.stable`` with the conditional Gaussian test and its
hill climbing with the conditional linear Gaussian BIC; and the empty graph. Each pair runs one
algorithm: PC with PC, GES with FGES, hill climbing with hill climbing. The data come from the
``mixed_er`` regime: a conditional linear Gaussian network on an Erdos-Renyi DAG, with the
``--categorical-share`` of the variables categorical, each with between the two ``--categories``
counts of categories. Every solution is serial and reads the categorical columns by one rule:
every column of at most ten distinct whole numbers.

Every estimate is scored against the true DAG's CPDAG. A CPDAG row's error is ``shd``; bnlearn's
hill climbing returns a DAG, whose comparable error is ``mec_shd``. ``--rescore`` recomputes the
score columns from the stored estimates. A timeout is a timeout: it has no scores.

    ANDREY_BENCH_PYTHON=... ANDREY_BENCH_TETRAD_JAR=... $ANDREY_BENCH_PYTHON \\
        benchmarks/mixed_cpu.py --out outputs/mixed-cpu --cap-mem-mb 32000
"""

from __future__ import annotations

import argparse
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from discrete_cpu import SCORE_COLUMNS, rescore

from andrey_bench import datasets, memcap, rundir
from andrey_bench.adapters.andrey_mixed import AndreyGESMixed, AndreyHCMixed, AndreyPCMixed
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.bnlearn_mixed import BnlearnHCCondGauss, BnlearnPCStableCondGauss
from andrey_bench.adapters.tetrad_cmd import TetradFGES, TetradPC
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores

REGIME = "mixed_er"

#: The score columns ``--rescore`` recomputes: the categorical lane's, then the DAG rows' MEC pair.
MIXED_SCORE_COLUMNS = (*SCORE_COLUMNS, "mec_shd", "mec_arrowhead_f1")

_PC_DG, _PC_CG = AndreyPCMixed("degen_gauss"), AndreyPCMixed("cond_gauss")
_GES_DG = AndreyGESMixed("local_score_degen_gauss")
_GES_CG = AndreyGESMixed("local_score_cond_gauss")
_HC_DG = AndreyHCMixed("local_score_degen_gauss")
_HC_CG = AndreyHCMixed("local_score_cond_gauss")
_TETRAD_PC_DG, _TETRAD_PC_CG = TetradPC("degen_gauss"), TetradPC("cond_gauss")
_FGES_DG = TetradFGES("local_score_degen_gauss")
_FGES_CG = TetradFGES("local_score_cond_gauss")
_BNLEARN_PC, _BNLEARN_HC = BnlearnPCStableCondGauss(), BnlearnHCCondGauss()

#: Andrey's run of one algorithm with one test or score, and a comparator's run of that algorithm,
#: by ``"<algorithm> <setting> <package>"``. bnlearn's hill climbing has the conditional linear
#: Gaussian BIC only, so it is the comparator for both of Andrey's hill-climbing scores.
PAIRS = {
    "pc degen_gauss tetrad": (_PC_DG, _TETRAD_PC_DG),
    "pc cond_gauss tetrad": (_PC_CG, _TETRAD_PC_CG),
    "pc cond_gauss bnlearn": (_PC_CG, _BNLEARN_PC),
    "ges degen_gauss tetrad": (_GES_DG, _FGES_DG),
    "ges cond_gauss tetrad": (_GES_CG, _FGES_CG),
    "hc degen_gauss bnlearn": (_HC_DG, _BNLEARN_HC),
    "hc cond_gauss bnlearn": (_HC_CG, _BNLEARN_HC),
}
SOLUTIONS = [
    _PC_DG,
    _TETRAD_PC_DG,
    _PC_CG,
    _TETRAD_PC_CG,
    _GES_DG,
    _FGES_DG,
    _GES_CG,
    _FGES_CG,
    _BNLEARN_PC,
    _BNLEARN_HC,
    _HC_DG,
    _HC_CG,
    BaselineEmpty(),
]

#: Smallest run that touches every solution: one tiny dataset.
SMOKE = {"ladder": [5], "seeds": 1, "n": 500, "cap_wall_s": 300.0}
WARMUP = 1


def regime_slice(categorical_share: float, categories: tuple[int, int]) -> dict:
    """Return the :class:`DatasetKey` coordinates of the mixed regime."""
    topology, functional, noise, standardize = datasets.REGIMES[REGIME]
    return {
        "topology": topology,
        "functional": functional,
        "noise": noise,
        "standardize": standardize,
        "density": datasets.CANONICAL_DENSITY,
        "categorical_share": categorical_share,
        "categories": categories,
    }


def ladder_keys(
    ladder: list[int], seeds: int, n: int, categorical_share: float, categories: tuple[int, int]
) -> list[DatasetKey]:
    """Return one dataset of `n` rows per ``(d, seed)``."""
    coordinates = regime_slice(categorical_share, categories)
    return [DatasetKey(**coordinates, d=d, n=n, seed=seed) for d in ladder for seed in range(seeds)]


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Mixed-data tests and scores: Andrey, Tetrad, bnlearn")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--ladder", type=int, nargs="+", default=[10, 20, 40], help="d sizes (default: 10 20 40)"
    )
    p.add_argument("--seeds", type=int, default=5, help="datasets per rung (default: 5)")
    p.add_argument("--n", type=int, default=2000, help="rows per dataset (default: 2000)")
    p.add_argument(
        "--categorical-share",
        type=float,
        default=0.5,
        help="share of the variables that are categorical (default: 0.5)",
    )
    p.add_argument(
        "--categories",
        type=int,
        nargs=2,
        default=[2, 3],
        metavar=("LEAST", "MOST"),
        help="categories per categorical variable (default: 2 3)",
    )
    p.add_argument(
        "--solutions",
        nargs="+",
        default=None,
        help="run only these solution names (default: every solution)",
    )
    p.add_argument("--repeats", type=int, default=1, help="timed repeats per fit (default: 1)")
    p.add_argument("--warmup", type=int, default=WARMUP, help="untimed fits per repeat")
    p.add_argument("--cap-wall-s", type=float, default=1800.0, help="wall cap per repeat, seconds")
    p.add_argument("--cap-mem-mb", type=float, default=None, help="memory cap per repeat, MB")
    p.add_argument("--mem-route", choices=memcap.ROUTES, default=None, help="cap enforcement route")
    p.add_argument("--cores", type=int, default=available_cores(), help="core budget per fit")
    p.add_argument("--machine-id", default=socket.gethostname(), help="machine id on every record")
    p.add_argument("--smoke", action="store_true", help="d=5, one seed, 500 rows, 300 s cap")
    p.add_argument("--resume", action="store_true", help="continue the run already in --out")
    p.add_argument("--rescore", action="store_true", help="recompute the score columns only")
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE)
    categories = (int(args.categories[0]), int(args.categories[1]))
    solutions = SOLUTIONS
    if args.solutions is not None:
        unknown = sorted(set(args.solutions) - {a.name for a in SOLUTIONS})
        if unknown:
            raise SystemExit(f"unknown solutions {unknown}; known: {[a.name for a in SOLUTIONS]}")
        solutions = [a for a in SOLUTIONS if a.name in args.solutions]

    out = Path(args.out)
    keys = {
        key.digest: key
        for key in ladder_keys(args.ladder, args.seeds, args.n, args.categorical_share, categories)
    }
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in solutions],
        "slice": regime_slice(args.categorical_share, categories),
        "generator settings": datasets.MIXED_SETTINGS,
        "ladder": args.ladder,
        "n": args.n,
        "seeds": args.seeds,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "cap_wall_s": args.cap_wall_s,
        "cap_mem_mb": args.cap_mem_mb,
        "mem_route": memcap.resolve_route(args.cap_mem_mb, args.mem_route),
        "cores": args.cores,
        "machine_id": args.machine_id,
    }
    if args.rescore:
        records_dir = out / rundir.RECORDS_DIRNAME
        if not any(records_dir.glob("*.parquet")):
            raise SystemExit(f"--rescore: {records_dir} holds no records")
        rescore(out, records_dir, keys, MIXED_SCORE_COLUMNS)
        table = rundir.assemble(records_dir)
        table.to_parquet(out / "runs.parquet", index=False)
        _log(f"rescored    : {len(table)} rows  -> {out / 'runs.parquet'}")
        _summarize(table)
        return 0
    records_dir = rundir.start_run(out, run_meta, resume=args.resume)
    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())

    _log(f"=== Mixed-data tests and scores, Andrey, Tetrad, bnlearn -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:12s}: {value}")
    _log(f"{'git_sha':12s}: {prov['git_sha']}")
    _log("")

    errors: list[str] = []
    skipped = 0
    measured = 0
    start = time.perf_counter()
    for d in args.ladder:
        result = run_batch(
            datasets.store_dir(out),
            solutions,
            ladder_keys([d], args.seeds, args.n, args.categorical_share, categories),
            machine_id=args.machine_id,
            cap_wall_s=args.cap_wall_s,
            cap_mem_mb=args.cap_mem_mb,
            mem_route=args.mem_route,
            repeats=args.repeats,
            warmup=args.warmup,
            cores=args.cores,
            records_dir=records_dir,
            run_dir=out,
        )
        measured += len(result.records)
        errors.extend(result.errors)
        skipped += result.skipped
        ran = []
        for adapter in solutions:
            mine = [r for r in result.records if r.name == adapter.name]
            walls = [r.wall_s for r in mine if r.status == "ok" and r.wall_s is not None]
            statuses = sorted({r.status for r in mine})
            cost = (
                f"{statistics.median(walls):.3g}s" if walls else "/".join(statuses) or "no-records"
            )
            ran.append(f"{adapter.name}={cost}")
        _log(f"[d={d}] {'  '.join(ran)} | elapsed {time.perf_counter() - start:.0f}s")

    table = rundir.assemble(records_dir)
    table.to_parquet(out / "runs.parquet", index=False)
    rundir.finish_run(out)
    _log("")
    _log(f"elapsed     : {time.perf_counter() - start:.1f}s")
    _log(f"measured    : {measured} records, {skipped} units skipped")
    _log(f"table       : {len(table)} rows  -> {out / 'runs.parquet'}")
    _log(f"errors      : {len(errors)}")
    for err in errors:
        _log(f"  ! {err}")
    _summarize(table)
    return 1 if errors else 0


def graph_error(row: pd.Series) -> float:
    """A row's comparable graph error: ``shd`` for a CPDAG, ``mec_shd`` for a DAG."""
    return float(row["mec_shd"] if row["output_type"] == "dag" else row["shd"])


def _summarize(table: pd.DataFrame) -> None:
    """Print each solution's finished fits, median wall time, and median error per ``d``, then each
    pair's speed ratio: the comparator's median wall time over Andrey's where both finished."""
    if table.empty:
        return
    ok = table[table["status"] == "ok"].copy()
    ok["error"] = ok.apply(graph_error, axis=1) if not ok.empty else []
    _log("")
    _log(f"{'solution':32s} {'d':>3s} {'ok/all':>7s} {'wall_s':>8s} {'error':>6s}")
    for (name, d), rows in table.groupby(["name", "d"], sort=True):
        done = ok[(ok["name"] == name) & (ok["d"] == d)]
        _log(
            f"{name:32s} {d:3d} {len(done):3d}/{len(rows):<3d} {done['wall_s'].median():8.3g} "
            f"{done['error'].median():6.3g}"
        )
    _log("")
    _log(f"{'pair':24s} {'d':>3s} {'paired':>6s} {'speed-up':>8s}")
    for method, (mine, theirs) in PAIRS.items():
        for d in sorted(ok["d"].unique()):
            a = ok[(ok["name"] == mine.name) & (ok["d"] == d)].set_index("dataset_id")["wall_s"]
            b = ok[(ok["name"] == theirs.name) & (ok["d"] == d)].set_index("dataset_id")["wall_s"]
            both = a.index.intersection(b.index)
            if not both.empty:
                _log(
                    f"{method:24s} {d:3d} {len(both):6d} {b[both].median() / a[both].median():8.3g}"
                )


__all__ = [
    "MIXED_SCORE_COLUMNS",
    "PAIRS",
    "SOLUTIONS",
    "graph_error",
    "ladder_keys",
    "regime_slice",
]

if __name__ == "__main__":
    raise SystemExit(main())
