#!/usr/bin/env python
"""Benchmark hidden-confounder methods on CPU: Andrey against lingam.

Runs RCD, BottomUpParceLiNGAM, and CAM-UV from both packages on data with hidden common causes,
with the empty graph as a quality baseline. RCD and BottomUpParceLiNGAM run on the
``latent_lingam_er`` regime, linear with uniform noise. CAM-UV runs on ``latent_anm_er``,
nonlinear with Gaussian additive noise. Every solution is serial, and both packages run a method
with the same settings.

``d`` includes hidden nodes: one node in five is hidden, at least one, so each rung has
``d - latents`` data columns. Every estimate is scored against the PAG of the observed nodes. Each
fit also gets the pairs it attributes to a hidden common cause, scored against two sets of true
pairs. A timeout is a timeout: it has no scores.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/confounded_cpu.py \\
        --out outputs/confounded-cpu --cap-mem-mb 32000
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
from andrey_bench import datasets, memcap, rundir, scoring
from andrey_bench.adapters.andrey_bottom_up_parce import AndreyBottomUpParceLiNGAM
from andrey_bench.adapters.andrey_camuv import AndreyCAMUV
from andrey_bench.adapters.andrey_rcd import AndreyRCD
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.lingam_bottom_up_parce import LingamBottomUpParceLiNGAM
from andrey_bench.adapters.lingam_camuv import LingamCAMUV
from andrey_bench.adapters.lingam_rcd import LingamRCD
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores

#: Each regime with the solutions that run on it. The empty graph runs on both.
LANES: dict[str, list] = {
    "latent_lingam_er": [
        AndreyRCD(),
        LingamRCD(),
        AndreyBottomUpParceLiNGAM(),
        LingamBottomUpParceLiNGAM(),
        BaselineEmpty(),
    ],
    "latent_anm_er": [AndreyCAMUV(), LingamCAMUV(), BaselineEmpty()],
}

#: Which edge marks carry each solution's confounded pairs. The empty graph declares none and has
#: no such edges, so the bidirected edges stand in for it.
CONFOUNDED_MARK = {
    a.name: getattr(a, "confounded_mark", "bidirected") for lane in LANES.values() for a in lane
}

#: Smallest run that touches every solution: one tiny dataset per regime.
SMOKE = {"ladder": [5], "seeds": 1, "n": 500, "cap_wall_s": 300.0}

#: Untimed fits before each measurement. RCD and CAM-UV fits take minutes, so a warm-up fit would
#: double their cost against the wall cap.
WARMUP = 0


def regime_slice(regime: str) -> dict:
    """Return the :class:`DatasetKey` coordinates of `regime`."""
    topology, functional, noise, standardize = datasets.REGIMES[regime]
    return {
        "topology": topology,
        "functional": functional,
        "noise": noise,
        "standardize": standardize,
        "density": datasets.CANONICAL_DENSITY,
    }


def ladder_keys(regime: str, ladder: list[int], seeds: int, n: int) -> list[DatasetKey]:
    """Return one `regime` dataset of `n` rows per ``(d, seed)``, with the regime's hidden nodes."""
    return [
        DatasetKey(
            **regime_slice(regime), d=d, n=n, seed=seed, latents=datasets.latents_for(regime, d)
        )
        for d in ladder
        for seed in range(seeds)
    ]


def _log(msg: str = "") -> None:
    """Print and flush batch output immediately."""
    print(msg, flush=True)


def add_pair_scores(out: Path, records_dir: Path, keys: Mapping[str, DatasetKey]) -> None:
    """Add the confounded-pair columns to every part file that lacks them.

    `keys` maps each ``dataset_id`` of the run to its :class:`DatasetKey`. Each fit's stored
    estimate is read back and scored with :func:`scoring.confounded_pair_scores`, using the edge
    marks its solution declares. A fit with no estimate has ``NaN`` in every column. The rewrite is
    atomic, and a part that already has the columns is left alone, so a resumed run scores only the
    units it measured.
    """
    store = datasets.store_dir(out)
    kinds = datasets.load_registry(store)["keys"]
    # The last dataset's truth, kept so its PAG projection runs once for all of its solutions.
    truth: tuple[str, GraphStructure] | None = None
    for part in sorted(records_dir.glob("*.parquet")):
        frame = pd.read_parquet(part)
        if scoring.CONFOUNDED_COLUMNS[0] in frame.columns:
            continue
        rows = []
        for record in frame.itertuples():
            estimate = rundir.read_structure(out, record.run_id) if record.status == "ok" else None
            if estimate is None:
                rows.append({})
                continue
            if truth is None or truth[0] != record.dataset_id:
                key = keys[record.dataset_id]
                _, marks = datasets.load(store, key)
                truth = (
                    record.dataset_id,
                    GraphStructure.from_numpy(
                        marks,
                        kind=kinds[key.label]["graph_kind"],
                        node_types=datasets.latent_node_types(len(marks), key.latents),
                    ),
                )
            rows.append(
                scoring.confounded_pair_scores(estimate, truth[1], CONFOUNDED_MARK[record.name])
            )
        scores = pd.DataFrame(
            rows, index=frame.index, columns=list(scoring.CONFOUNDED_COLUMNS)
        ).astype(float)
        scratch = part.with_name(f".{part.name}.tmp")
        pd.concat([frame, scores], axis=1).to_parquet(scratch, index=False)
        os.replace(scratch, part)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Hidden-confounder methods, Andrey against lingam, CPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--ladder", type=int, nargs="+", default=[5, 10, 15], help="d sizes (default: 5 10 15)"
    )
    p.add_argument("--seeds", type=int, default=5, help="datasets per rung (default: 5)")
    p.add_argument("--n", type=int, default=2000, help="rows per dataset (default: 2000)")
    p.add_argument("--repeats", type=int, default=1, help="timed repeats per fit (default: 1)")
    p.add_argument(
        "--warmup",
        type=int,
        default=WARMUP,
        help="untimed fits per repeat; the wall cap covers them too (default: 0)",
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
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE)

    out = Path(args.out)
    keys = {
        key.digest: key
        for regime in LANES
        for key in ladder_keys(regime, args.ladder, args.seeds, args.n)
    }

    # Written before the first fit, so an interrupted benchmark still says what it was asked for.
    run_meta = {
        "driver": Path(__file__).name,
        # One entry per (regime, solution), so the empty graph appears once for each regime.
        "solution list": [a.name for lane in LANES.values() for a in lane],
        "lanes": {regime: [a.name for a in lane] for regime, lane in LANES.items()},
        "slices": {regime: regime_slice(regime) for regime in LANES},
        "ladder": args.ladder,
        "latents": {
            regime: [datasets.latents_for(regime, d) for d in args.ladder] for regime in LANES
        },
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
    # Reads run_meta.json before writing it, so --resume has something to check against.
    records_dir = rundir.start_run(out, run_meta, resume=args.resume)
    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())

    _log(f"=== Hidden-confounder methods, Andrey and lingam -> {out} ===")
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
        for regime, lane in LANES.items():
            result = run_batch(
                datasets.store_dir(out),
                lane,
                ladder_keys(regime, [d], args.seeds, args.n),
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
            for adapter in lane:
                mine = [r for r in result.records if r.name == adapter.name]
                walls = [r.wall_s for r in mine if r.status == "ok" and r.wall_s is not None]
                statuses = sorted({r.status for r in mine})
                cost = (
                    f"{statistics.median(walls):.3g}s"
                    if walls
                    else "/".join(statuses) or "no-records"
                )
                ran.append(f"{adapter.name}={cost}")
            _log(f"[d={d} {regime}] {'  '.join(ran)} | elapsed {time.perf_counter() - start:.0f}s")
        add_pair_scores(out, records_dir, keys)

    # From the parts, not from memory: a resume that measured nothing still writes the table for
    # everything measured before it.
    add_pair_scores(out, records_dir, keys)
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
    """Print, per regime, solution, and rung, the fits that finished, with their medians and means.

    Wall time and SHD are medians over the finished fits. The F1 columns are means over the fits
    where F1 is defined, so ``nan`` marks a rung where every fit had nothing to find and claimed
    nothing. The ``true`` columns are the mean number of true pairs per target.
    """
    if table.empty:
        return
    _log("")
    _log(
        f"{'regime':10s} {'solution':30s} {'d':>3s} {'ok/all':>7s} {'wall_s':>8s} {'shd':>6s} "
        f"{'f1 pag':>7s} {'f1 cause':>9s} {'true pag':>9s} {'true cause':>10s}"
    )
    for (regime, name, d), rows in table.groupby(["functional", "name", "d"], sort=True):
        ok = rows[rows["status"] == "ok"]
        _log(
            f"{regime:10s} {name:30s} {d:3d} {len(ok):3d}/{len(rows):<3d} "
            f"{ok['wall_s'].median():8.3g} {ok['shd'].median():6.3g} "
            f"{ok['confounded_pag_f1'].mean():7.3f} {ok['confounded_hidden_cause_f1'].mean():9.3f} "
            f"{ok['confounded_pag_n_true'].mean():9.2f} "
            f"{ok['confounded_hidden_cause_n_true'].mean():10.2f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
