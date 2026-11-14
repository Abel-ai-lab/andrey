#!/usr/bin/env python
"""Benchmark the local PAG at a target under hidden variables: MMB-by-MMB's authors' code against
FCI and RFCI.

Runs the simulation of the MMB-by-MMB paper (Xie et al., 2024, Sect. 5.1) on published networks
(:mod:`andrey_bench.latent_networks`): each network's DAG as a linear-Gaussian model with random
nodes hidden, ``n`` rows, and the paper's targets (the two nodes with the most neighbors in a
network outside its four, or ``--targets k`` of them in any network). Every solution
answers at each target; a graph is scored on the edges at the target against the PAG of the
observed nodes (the CPDAG with ``--hidden 0``), an edge counting only with both marks right
(:func:`andrey_bench.local_lane.target_edge_scores`):

- a local solution (``local = True`` on its adapter, :class:`~andrey_bench.local_lane.LocalAdapter`)
  is fitted once per target;
- a solution that learns a whole graph, such as FCI, is fitted once per dataset and read at every
  target;
- the empty answer's SHD at a target is the target's degree in the truth.

The solutions are the authors' MMB-by-MMB, pcalg's RFCI with its tests counted, Andrey's FCI, and
the empty answer, all at the same ``alpha``. :data:`ANDREY_ARMS` is where the adapters of Andrey's
own local methods go; each needs the contract of :class:`~andrey_bench.local_lane.LocalAdapter`
and nothing else in the driver.

:data:`NETWORKS` and :data:`ROWS` are the paper's four networks and two sizes, with its hidden
counts (:data:`~andrey_bench.latent_networks.PAPER_HIDDEN`). One JSON line per fit goes to
``<out>/records.jsonl`` with the graph's marks and the scores, so ``--rescore`` recomputes the
scores without fitting; ``<out>/provenance.json`` records the build and the machine. A fit past the
per-fit wall cap is a ``timeout`` with no scores. Load R with pcalg as the README's R lane
describes, and set ``$ANDREY_BENCH_MMB`` and ``$ANDREY_BENCH_MMB_PYTHON`` as ``start_mmb.sh``
prints, before running the references.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/mmb_by_mmb_cpu.py --out outputs/mmb
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from andrey_bench import latent_networks, networks, rsession, rundir
from andrey_bench.adapters.andrey_fci import AndreyFCI
from andrey_bench.adapters.andrey_mmb_by_mmb import AndreyMMBbyMMB
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.pcalg_rfci_local import PcalgRFCICounted
from andrey_bench.adapters.xie_mmb_by_mmb import XieMMBbyMMB
from andrey_bench.local_lane import (
    TARGET_PARAM,
    entries_to_marks,
    marks_to_entries,
    target_edge_scores,
    unpack_answer,
)
from andrey_bench.runner import available_cores, check_interpreters, run_task, split_cores

#: Adapters of Andrey's own local methods (``mb_by_mb``, ``cmb``, ``mmb_by_mmb``). Each follows
#: :class:`~andrey_bench.local_lane.LocalAdapter`: ``fit`` reads the target's column from
#: ``params[TARGET_PARAM]`` and packs the marks at the target and its test count. ``mmb_by_mmb``
#: runs with the paper's blanket learner, total conditioning, and with IAMB.
ANDREY_ARMS: list = [AndreyMMBbyMMB(), AndreyMMBbyMMB("iamb")]

#: The references and the context solutions, then the empty answer that floors the field.
REFERENCES = [XieMMBbyMMB(), PcalgRFCICounted(), AndreyFCI()]

#: The level passed to every solution; the paper states none (the authors' example, and so their
#: code's default setting here, is 0.01). Andrey's MMB-by-MMB with total conditioning and the
#: authors' code test each blanket at ALPHA / (10 * floor(d / 10)) over d >= 10 columns.
ALPHA = 0.01

#: The paper's networks and sample sizes.
NETWORKS = ("mildew", "alarm", "win95pts", "andes")
ROWS = (1000, 5000)

#: Smallest run that touches every solution: two datasets of the smallest network.
SMOKE = {"networks": ["mildew"], "rows": [1000], "seeds": 2, "cap_wall_s": 300.0}

RECORDS = "records.jsonl"


def solutions() -> list:
    """Every solution of the lane, in the order records are written."""
    return [*ANDREY_ARMS, *REFERENCES, BaselineEmpty()]


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _read(out: Path) -> list[dict]:
    path = out / RECORDS
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def targets_for(network: networks.Network, count: int | None) -> tuple[str, ...]:
    """The targets of ``network``: its ``count`` nodes with the most neighbors, else the paper's
    targets in its four networks and the two with the most neighbors in any other."""
    if count is None and network.name in latent_networks.PAPER_TARGETS:
        return latent_networks.PAPER_TARGETS[network.name]
    return latent_networks.default_targets(network, 2 if count is None else count)


def hidden_for(network: str, override: int | None) -> int:
    """The hidden nodes of ``network``: ``override``, else the paper's count."""
    if override is not None:
        return override
    if network not in latent_networks.PAPER_HIDDEN:
        raise SystemExit(f"{network} has no hidden count in the paper's four; pass one")
    return latent_networks.PAPER_HIDDEN[network]


def _fit_params(adapter, target: int | None, alpha: float) -> dict:
    params = dict(adapter.params())
    if "alpha" in params:
        params["alpha"] = alpha
    if getattr(adapter, "local", False):
        params[TARGET_PARAM] = target
    return params


def score_record(
    record: dict, marks: np.ndarray, truth: np.ndarray, targets: dict[int, str]
) -> None:
    """Add ``scores`` to ``record``: the answer's scores at each target it answers for."""
    record["scores"] = {
        str(t): target_edge_scores(marks, truth, t)
        for t in ([record["target"]] if record["target"] is not None else targets)
    }


def rescore(out: Path, meta: dict, hidden: dict[str, int]) -> list[dict]:
    """Recompute every record's scores from its stored marks and rewrite the records."""
    if meta.get("sampler") != latent_networks.SAMPLER:
        raise SystemExit(
            f"--rescore: the run's data came from {meta.get('sampler', 'an earlier sampler')}, "
            f"not {latent_networks.SAMPLER}; rescore it with the build that recorded it"
        )
    records, truths = _read(out), {}
    for record in records:
        if record.get("marks") is None:
            continue
        key = (record["network"], record["seed"])
        if key not in truths:
            net = networks.load_network(record["network"])
            draw = latent_networks.sample(
                net,
                n=2,
                seed=record["seed"],
                hidden=record["hidden"],
                targets=meta["targets"][record["network"]],
            )
            truths[key] = (latent_networks.truth(draw), {t: draw.columns[t] for t in draw.targets})
        truth, targets = truths[key]
        marks = entries_to_marks(record["marks"], record["d_observed"])
        score_record(record, marks, truth, targets)
    (out / RECORDS).write_text("".join(json.dumps(r) + "\n" for r in records))
    return records


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MMB-by-MMB's authors' code against FCI and RFCI, CPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--networks",
        nargs="+",
        default=list(NETWORKS),
        help=f"published networks (default: the paper's {' '.join(NETWORKS)})",
    )
    p.add_argument(
        "--hidden",
        type=int,
        default=None,
        help="hidden nodes per network (default: the paper's; 0 gives the CPDAG as truth)",
    )
    p.add_argument(
        "--rows", type=int, nargs="+", default=list(ROWS), help="sample sizes (default: 1000 5000)"
    )
    p.add_argument("--seeds", type=int, default=20, help="datasets per network (default: 20)")
    p.add_argument("--first-seed", type=int, default=0, help="the first seed (default: 0)")
    p.add_argument(
        "--targets",
        type=int,
        default=None,
        help="the nodes with the most neighbors taken as targets (default: the paper's targets)",
    )
    p.add_argument("--alpha", type=float, default=ALPHA, help="level of every test (default: 0.01)")
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
        help="run mildew, 1,000 rows, two seeds, and a 300 s cap, replacing those flags",
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
    unknown = sorted(set(args.networks) - set(networks.available()))
    if unknown:
        p.error(f"unknown networks {unknown}; the networks are {list(networks.available())}")
    hidden = {name: hidden_for(name, args.hidden) for name in args.networks}

    field = solutions()
    if args.solutions is not None:
        chosen = set(args.solutions.split(","))
        known = {a.name for a in field}
        if chosen - known:
            p.error(
                f"unknown solutions {sorted(chosen - known)}; the solutions are {sorted(known)}"
            )
        field = [a for a in field if a.name in chosen]

    out = Path(args.out)
    meta_path = out / "run_meta.json"
    if args.rescore:
        if not (out / RECORDS).exists():
            raise SystemExit(f"--rescore: {out / RECORDS} holds no records")
        records = rescore(out, json.loads(meta_path.read_text()), hidden)
        _log(f"rescored    : {len(records)} records -> {out / RECORDS}")
        _summarize(records)
        return 0
    if (out / RECORDS).exists() and not args.resume:
        raise SystemExit(f"{out} already holds a run; pass --resume to continue it")
    out.mkdir(parents=True, exist_ok=True)
    check_interpreters(field)

    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in field],
        "params": {a.name: _fit_params(a, None, args.alpha) for a in field},
        "r environment": (
            rsession.check_environment("pcalg") if any(a.package == "pcalg" for a in field) else {}
        ),
        "sampler": latent_networks.SAMPLER,
        "weight_range": list(latent_networks.WEIGHT_RANGE),
        "networks": args.networks,
        "hidden": hidden,
        "targets": {
            name: list(targets_for(networks.load_network(name), args.targets))
            for name in args.networks
        },
        "rows": args.rows,
        "seeds": args.seeds,
        "first_seed": args.first_seed,
        "alpha": args.alpha,
        "part": args.part,
        "parts": args.parts,
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
    mmb = os.environ.get("ANDREY_BENCH_MMB_PYTHON")
    venvs = (
        {"mmb": Path(mmb).absolute().parent.parent}
        if mmb and XieMMBbyMMB.name in {a.name for a in field}
        else None
    )
    prov = rundir.provenance_once(
        out, timestamp=datetime.now(timezone.utc).isoformat(), extra_venvs=venvs
    )
    _log(f"=== MMB-by-MMB, FCI and RFCI at a target -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:12s}: {value}")
    _log(f"{'git_sha':12s}: {prov['git_sha']}")
    _log("")

    done = {
        (r["network"], r["n"], r["seed"], r["solution"], r["target"], r["repeat"])
        for r in _read(out)
    }
    units = [
        (name, seed)
        for name in args.networks
        for seed in range(args.first_seed, args.first_seed + args.seeds)
    ][args.part :: args.parts]
    start = time.perf_counter()
    with (out / RECORDS).open("a") as sink:
        for name, seed in units:
            net = networks.load_network(name)
            truth, labels = None, {}
            walls: dict[str, list[float]] = {a.name: [] for a in field}
            for n in args.rows:
                draw = latent_networks.sample(
                    net, n=n, seed=seed, hidden=hidden[name], targets=run_meta["targets"][name]
                )
                if truth is None:  # the model, so the truth, is the same for every n
                    truth = latent_networks.truth(draw)
                    labels = {t: draw.columns[t] for t in draw.targets}
                for adapter in field:
                    local = getattr(adapter, "local", False)
                    for target in draw.targets if local else (None,):
                        key = (name, n, seed, adapter.name, target)
                        if all((*key, r) in done for r in range(args.repeats)):
                            continue
                        params = _fit_params(adapter, target, args.alpha)
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
                            if (*key, repeat) in done:
                                continue
                            record = {
                                "network": name,
                                "n": n,
                                "seed": seed,
                                "hidden": hidden[name],
                                "d_observed": draw.data.shape[1],
                                "solution": adapter.name,
                                "package": adapter.package,
                                "package_version": adapter.package_version,
                                "target": target,
                                "target_names": labels,
                                "repeat": repeat,
                                "status": m.get("status"),
                                "wall_s": m.get("wall_s"),
                                "cpu_s": m.get("cpu_s"),
                                "peak_rss_mb": m.get("peak_rss_mb"),
                                "error_text": m.get("error_text") or m.get("stderr"),
                                "params": params,
                                "machine_id": args.machine_id,
                                "n_tests": None,
                                "marks": None,
                            }
                            if m.get("status") == "ok" and m.get("adj") is not None:
                                marks, tests = unpack_answer(np.asarray(m["adj"]))
                                record["n_tests"] = tests
                                record["marks"] = marks_to_entries(marks)
                                score_record(record, marks, truth, labels)
                                walls[adapter.name].append(m["wall_s"])
                            sink.write(json.dumps(record, default=str) + "\n")
                            sink.flush()
            cost = "  ".join(
                f"{a}={statistics.median(w):.3g}s" if w else f"{a}=no new fit finished"
                for a, w in walls.items()
            )
            _log(f"[{name} seed={seed}] {cost} | elapsed {time.perf_counter() - start:.0f}s")
    _summarize(_read(out))
    return 0


def _summarize(records: list[dict]) -> None:
    """Print, per network, size, and solution, the finished fits, the median time, and the scores.

    Precision and recall pool the counts of every (dataset, target) the solution answered;
    ``shd`` and ``empty`` are their means, and ``tests`` the mean test count of the fits that
    report one.
    """
    if not records:
        return
    _log("")
    _log(
        f"{'solution':22s} {'network':10s} {'n':>5s} {'ok/all':>7s} {'wall_s':>8s} {'prec':>6s} "
        f"{'recall':>6s} {'shd':>6s} {'empty':>6s} {'tests':>9s}"
    )
    for network, n in sorted({(r["network"], r["n"]) for r in records}):
        for name in sorted({r["solution"] for r in records}):
            mine = [
                r for r in records if (r["network"], r["n"], r["solution"]) == (network, n, name)
            ]
            fit = [r for r in mine if r["status"] == "ok"]
            if not fit:
                continue
            rows = [s for r in fit for s in r["scores"].values()]
            correct, est, true = (
                sum(s[k] for s in rows) for k in ("correct_edges", "est_edges", "true_edges")
            )
            counted = [r["n_tests"] for r in fit if r["n_tests"] is not None]
            _log(
                f"{name:22s} {network:10s} {n:5d} {len(fit):3d}/{len(mine):<3d} "
                f"{statistics.median(r['wall_s'] for r in fit):8.3g} "
                f"{correct / est if est else float('nan'):6.3f} "
                f"{correct / true if true else float('nan'):6.3f} "
                f"{statistics.mean(s['shd'] for s in rows):6.3g} "
                f"{statistics.mean(s['empty_shd'] for s in rows):6.3g} "
                f"{statistics.mean(counted) if counted else float('nan'):9.5g}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
