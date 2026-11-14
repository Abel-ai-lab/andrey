#!/usr/bin/env python
"""Benchmark the target-centered methods MB-by-MB and CMB on categorical networks, with global PC
read at the target.

Runs the simulations of the MB-by-MB paper (Wang et al., 2014, Sect. 5) and the CMB paper (Gao and
Ji, 2015, Sect. 5): ``n`` rows of a categorical network, one answer per target, scored on the edges
at the target against the true graph in both papers' readings
(:func:`andrey_bench.direct_lane.dag_counts`, against the DAG, and
:func:`~andrey_bench.direct_lane.cpdag_counts`, against its CPDAG). A cell is ``net:<network>:<n>``
and its networks are:

- ``alarm`` and ``hailfinder``, with every node a target (the first from :mod:`andrey.data`, the
  second from the published network's own tables, its columns permuted by the seed);
- ``chess45``, the 45 x 45 chessboard of 2025 binary variables with the paper's 10 fixed targets
  (:mod:`andrey_bench.chessboard`).

Each solution answers at every target of a dataset:

- a local solution (``local = True`` on its adapter, :class:`~andrey_bench.local_lane.LocalAdapter`)
  is fitted once per target;
- a solution that learns a whole graph, here PC, is fitted once per dataset and read at every
  target, as the global arms of the papers are;
- the empty answer is the floor, its counts the target's degree in the truth.

The solutions are Andrey's MB-by-MB and CMB, Andrey's PC with ``gsq``, bnlearn's ``pc.stable`` with
``mi-adf``, and the empty answer, all at the same ``--alpha``. ``--max-depth`` caps the conditioning
sets of the two PC arms only: MB-by-MB and CMB keep their own caps (see
:mod:`andrey_bench.adapters.andrey_target`). ``--paper`` sets the flags a paper used:

- ``mb_by_mb``: alarm and hailfinder at 500, 1,000, 2,000, and 5,000 rows, ``--alpha 0.05`` (the
  paper also runs 0.01), and conditioning sets of at most 5 in every algorithm;
- ``cmb``: alarm at 1,000 rows, ``--alpha 0.02``, and no cap.

One JSON line per fit and target goes to ``<out>/records.jsonl`` with the marks at the target and
the scores, so ``--rescore`` recomputes the scores without fitting; ``<out>/provenance.json``
records the build and the machine. A record of a global solution has ``scope`` ``graph``: its
``wall_s``, ``cpu_s``, and ``n_tests`` are those of the one fit every target of the dataset reads.
A fit past the per-fit wall cap, or one that fails, is a record with its ``status`` and no marks and
no scores, never an empty answer. Load R with bnlearn as the README's R lane describes before
running the bnlearn arm.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/direct_cpu.py --out outputs/direct
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

from andrey_bench import chessboard, direct_lane, networks, rsession, rundir
from andrey_bench.adapters import bnlearn_blanket
from andrey_bench.adapters.andrey_target import AndreyPCCounted, AndreyTarget
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.bnlearn_pc_graph import BnlearnPCStableGraph
from andrey_bench.local_lane import (
    TARGET_PARAM,
    entries_to_marks,
    marks_to_entries,
    unpack_answer,
)
from andrey_bench.runner import available_cores, check_interpreters, run_task, split_cores

#: Adapters of Andrey's own target-centered methods. Each follows
#: :class:`~andrey_bench.local_lane.LocalAdapter`: ``fit`` reads the target's column from
#: ``params[TARGET_PARAM]`` and packs the marks at the target and its test count.
ANDREY_ARMS: list = [AndreyTarget("mb_by_mb"), AndreyTarget("cmb")]

#: The global solutions, fitted once per dataset and read at every target.
GLOBAL_ARMS: list = [AndreyPCCounted(), BnlearnPCStableGraph()]

#: What each paper ran: its networks, sample sizes, level, conditioning-set cap, and solutions.
PAPERS = {
    "mb_by_mb": {
        "networks": ["alarm", "hailfinder"],
        "rows": [500, 1000, 2000, 5000],
        "alpha": 0.05,
        "max_depth": 5,
        "solutions": ["andrey.mb_by_mb", "andrey.pc.gsq", "bnlearn.pc-stable.graph"],
    },
    "cmb": {
        "networks": ["alarm"],
        "rows": [1000],
        "alpha": 0.02,
        "max_depth": None,
        "solutions": ["andrey.cmb", "andrey.pc.gsq", "bnlearn.pc-stable.graph"],
    },
}

#: Smallest run that touches every solution: three targets of one dataset of alarm.
SMOKE = {"networks": ["alarm"], "rows": [500], "seeds": 1, "targets": 3, "cap_wall_s": 300.0}

RECORDS = "records.jsonl"


def solutions() -> list:
    """Every solution of the lane, in the order records are written."""
    return [*ANDREY_ARMS, *GLOBAL_ARMS, BaselineEmpty()]


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _read(out: Path) -> list[dict]:
    path = out / RECORDS
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def parse_max_depth(text: str) -> int | None:
    """``none`` or an integer of at least 1, the least bnlearn's ``max.sx`` takes."""
    if text.lower() == "none":
        return None
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("--max-depth is none or at least 1 (bnlearn's max.sx)")
    return value


def cell_label(network: str, n: int) -> str:
    """The cell's name, ``net:<network>:<n>``."""
    return f"net:{network}:{n}"


def _fit_params(adapter, target: int | None, alpha: float, max_depth: int | None) -> dict:
    params = dict(adapter.params())
    if "alpha" in params:
        params["alpha"] = alpha
    if "max_depth" in params:
        params["max_depth"] = max_depth
    if getattr(adapter, "local", False):
        params[TARGET_PARAM] = target
    return params


def score_record(
    record: dict, at_target: np.ndarray, dag_marks: np.ndarray, cpdag_marks: np.ndarray
) -> None:
    """Add ``scores`` to ``record``: both readings' counts of the marks at its target."""
    record["scores"] = direct_lane.scores(at_target, dag_marks, cpdag_marks, record["target"])


def rescore(out: Path, meta: dict) -> list[dict]:
    """Recompute every record's scores from its stored marks and rewrite the records."""
    records = _read(out)
    if any(r["network"] == direct_lane.CHESSBOARD for r in records) and (
        meta.get("chessboard") != chessboard.SAMPLER
    ):
        raise SystemExit(
            f"--rescore: the run's chessboard came from {meta.get('chessboard', 'an earlier one')}, "
            f"not {chessboard.SAMPLER}; rescore it with the build that recorded it"
        )
    truths: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for record in records:
        if record.get("marks") is None:
            continue
        key = (record["network"], record["seed"])
        if key not in truths:
            dag = direct_lane.truth(*key).dag
            truths[key] = (direct_lane.marks_of(dag), direct_lane.cpdag_of(dag))
        marks = entries_to_marks(record["marks"], record["d"])
        score_record(record, marks, *truths[key])
    tmp = out / f"{RECORDS}.tmp"
    tmp.write_text("".join(json.dumps(r) + "\n" for r in records))
    tmp.replace(out / RECORDS)
    return records


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="MB-by-MB and CMB at a target, with global PC read at the target, CPU"
    )
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--paper",
        choices=sorted(PAPERS),
        default="mb_by_mb",
        help="the paper whose flags are the defaults of the flags below (default: mb_by_mb)",
    )
    p.add_argument(
        "--networks",
        nargs="+",
        default=None,
        help="alarm, hailfinder, chess45, or another published network (default: the paper's)",
    )
    p.add_argument(
        "--rows", type=int, nargs="+", default=None, help="sample sizes (default: the paper's)"
    )
    p.add_argument(
        "--alpha", type=float, default=None, help="level of every test (default: the paper's)"
    )
    p.add_argument(
        "--max-depth",
        type=parse_max_depth,
        default=argparse.SUPPRESS,  # absent means the paper's, and "none" is no cap
        help="largest conditioning set of the two PC arms: none or at least 1 (default: the "
        "paper's)",
    )
    p.add_argument("--seeds", type=int, default=20, help="datasets per cell (default: 20)")
    p.add_argument("--first-seed", type=int, default=0, help="the first seed (default: 0)")
    p.add_argument(
        "--targets",
        type=int,
        default=None,
        help="answer for the first this many targets of each dataset (default: all)",
    )
    p.add_argument("--part", type=int, default=0, help="run every --parts-th dataset from this one")
    p.add_argument("--parts", type=int, default=1, help="parts the datasets are split into")
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
        "--solutions",
        default=None,
        help="run only these solution names, comma-separated (default: the paper's, and the "
        "empty answer)",
    )
    p.add_argument(
        "--machine-id",
        default=socket.gethostname(),
        help="machine id stamped on every record (default: this host's name)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="run alarm, 500 rows, one seed, three targets, and a 300 s cap, replacing those flags",
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
    preset = PAPERS[args.paper]
    for flag in ("networks", "rows", "alpha"):
        if getattr(args, flag) is None:
            setattr(args, flag, preset[flag])
    if not hasattr(args, "max_depth"):
        args.max_depth = preset["max_depth"]
    if args.smoke:
        vars(args).update(SMOKE)
    known_networks = {*networks.available(), direct_lane.CHESSBOARD}
    unknown = sorted(set(args.networks) - known_networks)
    if unknown:
        p.error(f"unknown networks {unknown}; the networks are {sorted(known_networks)}")

    field = solutions()
    if args.solutions is not None:
        chosen = set(args.solutions.split(","))
        known = {a.name for a in field}
        if chosen - known:
            p.error(
                f"unknown solutions {sorted(chosen - known)}; the solutions are {sorted(known)}"
            )
    else:
        chosen = {*preset["solutions"], BaselineEmpty.name}
    field = [a for a in field if a.name in chosen]

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
    check_interpreters(field)

    uses_r = any(a.package == "bnlearn" for a in field)
    run_meta = {
        "driver": Path(__file__).name,
        "paper": args.paper,
        "solution list": [a.name for a in field],
        "params": {a.name: _fit_params(a, None, args.alpha, args.max_depth) for a in field},
        "r environment": rsession.check_environment("bnlearn") if uses_r else {},
        "chessboard": chessboard.SAMPLER,
        "networks": args.networks,
        "rows": args.rows,
        "seeds": args.seeds,
        "first_seed": args.first_seed,
        "targets": args.targets,
        "alpha": args.alpha,
        "max_depth": args.max_depth,
        "part": args.part,
        "parts": args.parts,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "cap_wall_s": args.cap_wall_s,
        "cap_mem_mb": args.cap_mem_mb,
        "cores": args.cores,
        "machine_id": args.machine_id,
    }
    if uses_r:
        bnlearn_blanket.check_version()
    if args.resume and meta_path.exists() and json.loads(meta_path.read_text()) != run_meta:
        raise SystemExit(f"--resume: {meta_path} differs from these flags; pass the first run's")
    meta_path.write_text(json.dumps(run_meta, indent=2))
    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())
    _log(f"=== MB-by-MB and CMB, with PC read at the target -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:12s}: {value}")
    _log(f"{'git_sha':12s}: {prov['git_sha']}")
    _log("")

    done = {(r["cell"], r["seed"], r["solution"], r["target"], r["repeat"]) for r in _read(out)}
    units = [
        (name, n, seed)
        for name in args.networks
        for n in args.rows
        for seed in range(args.first_seed, args.first_seed + args.seeds)
    ][args.part :: args.parts]
    start = time.perf_counter()
    with (out / RECORDS).open("a") as sink:
        for name, n, seed in units:
            cell = cell_label(name, n)
            draw = direct_lane.draw(name, n, seed)
            d = draw.data.shape[1]
            dag_marks = direct_lane.marks_of(draw.dag)
            cpdag_marks = direct_lane.cpdag_of(draw.dag)
            targets = draw.targets if args.targets is None else draw.targets[: args.targets]
            walls: dict[str, list[float]] = {a.name: [] for a in field}
            for adapter in field:
                local = getattr(adapter, "local", False)
                # A local solution answers for one target per fit, a global one for all of them.
                for answered in [[t] for t in targets] if local else [targets]:
                    keys = [
                        (cell, seed, adapter.name, t, r)
                        for t in answered
                        for r in range(args.repeats)
                    ]
                    if all(key in done for key in keys):
                        continue
                    params = _fit_params(
                        adapter, answered[0] if local else None, args.alpha, args.max_depth
                    )
                    threads, n_jobs = split_cores(args.cores, adapter.mode)
                    measures = run_task(
                        adapter,
                        draw.data,
                        params,
                        cap_wall_s=args.cap_wall_s,
                        cap_mem_mb=args.cap_mem_mb,
                        repeats=args.repeats,
                        warmup=args.warmup,
                        threads=threads,
                        n_jobs=n_jobs,
                    )
                    for repeat, m in enumerate(measures):
                        whole, tests = None, None
                        if m.get("status") == "ok" and m.get("adj") is not None:
                            whole, tests = unpack_answer(np.asarray(m["adj"]))
                            walls[adapter.name].append(m["wall_s"])
                        for t in answered:
                            if (cell, seed, adapter.name, t, repeat) in done:
                                continue
                            record = {
                                "cell": cell,
                                "network": name,
                                "n": n,
                                "seed": seed,
                                "d": d,
                                "solution": adapter.name,
                                "package": adapter.package,
                                "package_version": adapter.package_version,
                                "scope": "target" if local else "graph",
                                "target": t,
                                "target_name": draw.labels[t],
                                "repeat": repeat,
                                "status": m.get("status"),
                                "wall_s": m.get("wall_s"),
                                "cpu_s": m.get("cpu_s"),
                                "peak_rss_mb": m.get("peak_rss_mb"),
                                "error_text": m.get("error_text") or m.get("stderr"),
                                "params": params,
                                "machine_id": args.machine_id,
                                "n_tests": tests,
                                "marks": None,
                                "scores": None,
                            }
                            if whole is not None:
                                at_target = direct_lane.star_at(whole, t)
                                record["marks"] = marks_to_entries(at_target)
                                score_record(record, at_target, dag_marks, cpdag_marks)
                            sink.write(json.dumps(record, default=str) + "\n")
                            sink.flush()
            cost = "  ".join(
                f"{a}={statistics.median(w):.3g}s" if w else f"{a}=no new fit finished"
                for a, w in walls.items()
            )
            _log(f"[{cell} seed={seed}] {cost} | elapsed {time.perf_counter() - start:.0f}s")
    _summarize(_read(out))
    return 0


def _summarize(records: list[dict]) -> None:
    """Print, per cell and solution, the finished records, the median fit time, and the counts.

    ``dag`` is the MB-by-MB paper's reading, the mean per target of ``undirected``, ``miss``,
    ``extra``, ``reverse``, and ``shd``; ``cpdag`` the CMB paper's, the mean of ``extra``,
    ``missing``, ``reversed``, and ``total``. ``tests`` is the mean test count of the records that
    report one: a global solution's is its one fit's.
    """
    if not records:
        return
    _log("")
    _log(
        f"{'solution':24s} {'cell':20s} {'ok/all':>9s} {'wall_s':>8s} {'tests':>9s} | "
        f"{'und':>5s} {'miss':>5s} {'extra':>5s} {'rev':>5s} {'shd':>5s} | "
        f"{'extra':>5s} {'miss':>5s} {'revd':>5s} {'total':>5s}"
    )
    for cell in sorted({r["cell"] for r in records}):
        for name in sorted({r["solution"] for r in records}):
            mine = [r for r in records if (r["cell"], r["solution"]) == (cell, name)]
            fit = [r for r in mine if r["status"] == "ok"]
            if not fit:
                continue
            counted = [r["n_tests"] for r in fit if r["n_tests"] is not None]
            dag = [r["scores"]["dag"] for r in fit]
            cpdag = [r["scores"]["cpdag"] for r in fit]

            def mean(rows: list[dict], key: str) -> float:
                return statistics.mean(row[key] for row in rows)

            _log(
                f"{name:24s} {cell:20s} {len(fit):4d}/{len(mine):<4d} "
                f"{statistics.median(r['wall_s'] for r in fit):8.3g} "
                f"{statistics.mean(counted) if counted else float('nan'):9.5g} | "
                + " ".join(f"{mean(dag, k):5.2f}" for k in ("undirected", "miss", "extra"))
                + f" {mean(dag, 'reverse'):5.2f} {mean(dag, 'shd'):5.2f} | "
                + " ".join(f"{mean(cpdag, k):5.2f}" for k in ("extra", "missing", "reversed"))
                + f" {mean(cpdag, 'total'):5.2f}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
