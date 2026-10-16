#!/usr/bin/env python
"""Benchmark the categorical tests and scores on CPU: Andrey against causal-learn, bnlearn, pgmpy.

Runs PC with the chi-square and the G-square test, GES with the BDeu score, and hill climbing with
the BDeu score and the multinomial BIC, on the ``discrete_er`` regime: category codes sampled from a
discrete network with Dirichlet tables on an Erdos-Renyi DAG (:data:`datasets.DISCRETE_SETTINGS`
gives the categories per node and the concentration). Each pair runs one method with one test or
score and the same settings, and every solution is serial:

- causal-learn's PC and GES, against Andrey's: ``alpha=0.05`` for PC, BDeu's default priors and at
  most four parents for GES. Andrey's GES runs on the numpy backend, so no compile lands in a fit.
- bnlearn's ``pc.stable`` and ``hc``, against Andrey's PC and hill climbing; pgmpy's hill climbing,
  against Andrey's. Hill climbing runs BDeu with an equivalent sample size of 1 and a structure
  prior of one expected parent per variable, or the multinomial BIC with the standard penalty, and
  no parent cap. ``bnlearn_discrete.py`` and ``pgmpy_hc.py`` document each setting.

The empty graph is the quality baseline. Every estimate is scored against the true DAG's CPDAG
(``shd`` and the skeleton and arrowhead columns), and a DAG's comparable columns are ``mec_shd`` and
``mec_arrowhead_f1``; ``--rescore`` recomputes those columns from the stored estimates. A timeout is
a timeout: it has no scores. ``--solutions`` runs only the named solutions: give a run of new
solutions a fresh ``--out``, and share the datasets of the earlier run through
``ANDREY_BENCH_DATA``.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/discrete_cpu.py \\
        --out outputs/discrete-cpu --cap-mem-mb 32000
"""

from __future__ import annotations

import argparse
import os
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

from andrey import GraphStructure
from andrey_bench import datasets, memcap, rsession, rundir, scoring
from andrey_bench.adapters import pgmpy_hc
from andrey_bench.adapters.andrey_ges import andrey_ges_bdeu
from andrey_bench.adapters.andrey_hc import andrey_hc_categorical
from andrey_bench.adapters.andrey_pc import AndreyPCChisq, AndreyPCGsq
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.bnlearn_discrete import BnlearnHCCategorical, BnlearnPCStableCategorical
from andrey_bench.adapters.causal_learn_ges import CausalLearnGESBDeu
from andrey_bench.adapters.causal_learn_pc import CausalLearnPCChisq, CausalLearnPCGsq
from andrey_bench.adapters.pgmpy_hc import PgmpyHC
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores

REGIME = "discrete_er"

_PC_CHISQ, _PC_GSQ = AndreyPCChisq(), AndreyPCGsq()
_HC_BDEU = andrey_hc_categorical("local_score_BDeu")
_HC_BIC = andrey_hc_categorical("local_score_BIC_categorical")

#: Andrey's and causal-learn's run of one method with one setting, by method.
PAIRS = {
    "pc chisq": (_PC_CHISQ, CausalLearnPCChisq()),
    "pc gsq": (_PC_GSQ, CausalLearnPCGsq()),
    "ges bdeu": (andrey_ges_bdeu(), CausalLearnGESBDeu()),
}

#: Andrey's and a comparator's run of one method with one test or score and the same settings, by
#: ``"<method> <setting> <package>"``. One Andrey solution is paired with each package that runs
#: its setting.
COMPARATOR_PAIRS = {
    "pc chisq bnlearn": (_PC_CHISQ, BnlearnPCStableCategorical("chisq")),
    "pc gsq bnlearn": (_PC_GSQ, BnlearnPCStableCategorical("gsq")),
    "hc bdeu bnlearn": (_HC_BDEU, BnlearnHCCategorical("local_score_BDeu")),
    "hc bdeu pgmpy": (_HC_BDEU, PgmpyHC("local_score_BDeu")),
    "hc bic bnlearn": (_HC_BIC, BnlearnHCCategorical("local_score_BIC_categorical")),
    "hc bic pgmpy": (_HC_BIC, PgmpyHC("local_score_BIC_categorical")),
}


#: The solutions: the causal-learn pairs, then the comparators', each solution once, then the empty
#: graph.
SOLUTIONS = [
    *{a.name: a for pair in (*PAIRS.values(), *COMPARATOR_PAIRS.values()) for a in pair}.values(),
    BaselineEmpty(),
]

#: Smallest run that touches every solution: one tiny dataset.
SMOKE = {"ladder": [5], "seeds": 1, "n": 500, "cap_wall_s": 300.0}

#: Untimed fits before each measurement, so imports and first-call caches stay out of the timing.
#: The per-repeat wall cap covers warm-up and fit together.
WARMUP = 1


def regime_slice() -> dict:
    """Return the :class:`DatasetKey` coordinates of the categorical regime."""
    topology, functional, noise, standardize = datasets.REGIMES[REGIME]
    return {
        "topology": topology,
        "functional": functional,
        "noise": noise,
        "standardize": standardize,
        "density": datasets.CANONICAL_DENSITY,
    }


def ladder_keys(ladder: list[int], seeds: int, n: int) -> list[DatasetKey]:
    """Return one dataset of `n` rows per ``(d, seed)``."""
    return [
        DatasetKey(**regime_slice(), d=d, n=n, seed=seed) for d in ladder for seed in range(seeds)
    ]


def _log(msg: str = "") -> None:
    """Print and flush batch output immediately."""
    print(msg, flush=True)


#: The metric columns :func:`rescore` recomputes.
SCORE_COLUMNS = (
    "shd",
    "skeleton_precision",
    "skeleton_recall",
    "skeleton_f1",
    "arrowhead_precision",
    "arrowhead_recall",
    "arrowhead_f1",
)


#: The columns ``--rescore`` recomputes: those above, then the MEC pair a DAG's estimate has.
DISCRETE_SCORE_COLUMNS = (*SCORE_COLUMNS, "mec_shd", "mec_arrowhead_f1")


def comparable(row: pd.Series, column: str) -> float:
    """A row's `column` in the form comparable across estimate kinds: the column itself for a CPDAG,
    and ``mec_<column>`` for a DAG, since ``shd`` and ``arrowhead_f1`` score the DAG itself."""
    return float(row[f"mec_{column}" if row["output_type"] == "dag" else column])


def rescore(
    out: Path,
    records_dir: Path,
    keys: Mapping[str, DatasetKey],
    columns: tuple[str, ...] = SCORE_COLUMNS,
) -> None:
    """Recompute `columns` of every part file from its stored estimates.

    `keys` maps each ``dataset_id`` of the run to its :class:`DatasetKey`. Each fit's estimate is
    read back and scored with :func:`scoring.score_result` against the true DAG; a fit with no
    estimate, or a column its estimate's kind has no score for, keeps ``NaN``. Each part is
    rewritten atomically.
    """
    store = datasets.store_dir(out)
    kinds = datasets.load_registry(store)["keys"]
    truths: dict[str, GraphStructure] = {}
    for part in sorted(records_dir.glob("*.parquet")):
        frame = pd.read_parquet(part)
        for column in columns:
            frame[column] = float("nan")
        for index, record in frame.iterrows():
            estimate = rundir.read_structure(out, record.run_id) if record.status == "ok" else None
            if estimate is None:
                continue
            if record.dataset_id not in truths:
                if record.dataset_id not in keys:
                    raise SystemExit(
                        f"{part.name}: dataset {record.dataset_id} is not in this run's ladder; "
                        "pass the first run's flags"
                    )
                key = keys[record.dataset_id]
                _, marks = datasets.load(store, key)
                truths[record.dataset_id] = GraphStructure.from_numpy(
                    marks, kind=kinds[key.label]["graph_kind"]
                )
            metrics = scoring.score_result(estimate, truths[record.dataset_id])
            for column in columns:
                frame.at[index, column] = float(metrics.get(column, float("nan")))
        scratch = part.with_name(f".{part.name}.tmp")
        frame.to_parquet(scratch, index=False)
        os.replace(scratch, part)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Categorical tests and scores: Andrey, causal-learn, bnlearn, pgmpy"
    )
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--ladder", type=int, nargs="+", default=[10, 20, 40], help="d sizes (default: 10 20 40)"
    )
    p.add_argument("--seeds", type=int, default=5, help="datasets per rung (default: 5)")
    p.add_argument("--n", type=int, default=2000, help="rows per dataset (default: 2000)")
    p.add_argument(
        "--solutions",
        nargs="+",
        default=None,
        help="run only these solution names (default: every solution); a run of other solutions "
        "than the first run's needs a fresh --out",
    )
    p.add_argument("--repeats", type=int, default=1, help="timed repeats per fit (default: 1)")
    p.add_argument(
        "--warmup",
        type=int,
        default=WARMUP,
        help="untimed fits per repeat; the wall cap covers them too (default: 1)",
    )
    p.add_argument("--cap-wall-s", type=float, default=1800.0, help="wall cap per repeat, seconds")
    p.add_argument("--cap-mem-mb", type=float, default=None, help="memory cap per repeat, MB")
    p.add_argument("--mem-route", choices=memcap.ROUTES, default=None, help="cap enforcement route")
    p.add_argument(
        "--cores",
        type=int,
        default=available_cores(),
        help="core budget per fit (default: every core this job is allowed)",
    )
    p.add_argument(
        "--machine-id",
        default=socket.gethostname(),
        help="machine id stamped on every record and in run_id (default: this host's name)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="run d=5, one seed, 500 rows, and a 300 s cap, replacing those flags",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue the run already in --out: skip units whose records are on disk. Requires "
        "the same run_meta.json, so pass the flags the first run was given",
    )
    p.add_argument(
        "--rescore",
        action="store_true",
        help="fit nothing: recompute the score columns of the run already in --out and rewrite "
        "runs.parquet. Takes the first run's flags, as --resume does",
    )
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE)
    solutions = SOLUTIONS
    if args.solutions is not None:
        unknown = sorted(set(args.solutions) - {a.name for a in SOLUTIONS})
        # A typo would otherwise run fewer solutions, and the missing rows would read as failures.
        if unknown:
            raise SystemExit(f"unknown solutions {unknown}; known: {[a.name for a in SOLUTIONS]}")
        solutions = [a for a in SOLUTIONS if a.name in args.solutions]

    out = Path(args.out)
    keys = {key.digest: key for key in ladder_keys(args.ladder, args.seeds, args.n)}

    # Written before the first fit, so an interrupted benchmark still says what it was asked for.
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in solutions],
        "slice": regime_slice(),
        "generator settings": datasets.DISCRETE_SETTINGS,
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
        # Fits nothing, so the run's STATUS.json and sittings stay as the fitting runs left them.
        records_dir = out / rundir.RECORDS_DIRNAME
        if not any(records_dir.glob("*.parquet")):
            raise SystemExit(f"--rescore: {records_dir} holds no records")
        rescore(out, records_dir, keys, DISCRETE_SCORE_COLUMNS)
        table = rundir.assemble(records_dir)
        table.to_parquet(out / "runs.parquet", index=False)
        _log(f"rescored    : {len(table)} rows  -> {out / 'runs.parquet'}")
        _summarize(table)
        return 0
    # Before the first fit, so a missing R library or pgmpy fails once; a run without those
    # solutions needs neither, and its run_meta.json is the one an earlier run of them wrote.
    if any(a.package == "bnlearn" for a in solutions):
        run_meta["r environment"] = rsession.check_environment("bnlearn")
    if any(a.package == "pgmpy" for a in solutions):
        run_meta.update(pgmpy_hc.check_environment())
    records_dir = rundir.start_run(out, run_meta, resume=args.resume)
    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())

    _log(f"=== Categorical tests and scores, Andrey, causal-learn, bnlearn, pgmpy -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:12s}: {value}")
    _log(f"{'git_sha':12s}: {prov['git_sha']}")
    _log("")

    records: list = []
    errors: list[str] = []
    skipped = 0
    start = time.perf_counter()

    for d in args.ladder:
        # One rung at a time, so a rung's data is materialized only when the ladder reaches it.
        result = run_batch(
            datasets.store_dir(out),
            solutions,
            ladder_keys([d], args.seeds, args.n),
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
        records.extend(result.records)
        errors.extend(result.errors)
        skipped += result.skipped

        # What each solution cost at this rung, or what stopped it.
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

    # From the parts, not from memory: a resume that measured nothing still writes the table for
    # everything measured before it.
    table = rundir.assemble(records_dir)
    table.to_parquet(out / "runs.parquet", index=False)
    rundir.finish_run(out)

    _log("")
    _log(f"elapsed     : {time.perf_counter() - start:.1f}s")
    _log(f"measured    : {len(records)} records, {skipped} units skipped")
    _log(f"table       : {len(table)} rows  -> {out / 'runs.parquet'}")
    _log(f"errors      : {len(errors)}")
    for err in errors:
        _log(f"  ! {err}")
    _summarize(table)
    return 1 if errors else 0


def _summarize(table: pd.DataFrame) -> None:
    """Print, per solution and rung, the finished fits with median wall time and error, then each
    pair's speed ratio.

    Wall time and error are medians over the finished fits; the F1 columns are means. Error and
    arrowhead F1 are a CPDAG row's own columns and a DAG row's ``mec_`` ones, which score its
    equivalence class. ``speed-up`` is the other package's median wall time over Andrey's, on the
    datasets where both finished.
    """
    if table.empty:
        return
    ok = table[table["status"] == "ok"].copy()
    ok["error"] = ok.apply(comparable, axis=1, column="shd") if not ok.empty else []
    ok["arrow"] = ok.apply(comparable, axis=1, column="arrowhead_f1") if not ok.empty else []
    _log("")
    _log(
        f"{'solution':28s} {'d':>3s} {'ok/all':>7s} {'wall_s':>8s} {'error':>6s} "
        f"{'f1 skel':>7s} {'f1 arrow':>8s}"
    )
    for (name, d), rows in table.groupby(["name", "d"], sort=True):
        done = ok[(ok["name"] == name) & (ok["d"] == d)]
        _log(
            f"{name:28s} {d:3d} {len(done):3d}/{len(rows):<3d} {done['wall_s'].median():8.3g} "
            f"{done['error'].median():6.3g} {done['skeleton_f1'].mean():7.3f} "
            f"{done['arrow'].mean():8.3f}"
        )
    _log("")
    _log(f"{'pair':18s} {'d':>3s} {'paired':>6s} {'speed-up':>8s}")
    for method, (mine, theirs) in {**PAIRS, **COMPARATOR_PAIRS}.items():
        for d in sorted(ok["d"].unique()):
            a = ok[(ok["name"] == mine.name) & (ok["d"] == d)].set_index("dataset_id")["wall_s"]
            b = ok[(ok["name"] == theirs.name) & (ok["d"] == d)].set_index("dataset_id")["wall_s"]
            both = a.index.intersection(b.index)
            if both.empty:
                continue
            ratio = b[both].median() / a[both].median()
            _log(f"{method:18s} {d:3d} {len(both):6d} {ratio:8.3g}")


if __name__ == "__main__":
    raise SystemExit(main())
