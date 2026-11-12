#!/usr/bin/env python
"""Benchmark ancestral claims under selection bias: Andrey's and the authors' Y-structure code.

Runs the simulation of Versteeg, Zhang, and Mooij (2022, Sect. 4.2) (:mod:`andrey_bench.selection_models`):
a linear-Gaussian model on the fixed 7-node graph of its Figure 5(a), or on a random graph of
``p = 8`` or ``p = 16`` system variables, with a context variable. Each model gives two datasets,
one under selection bias (a sum of a few variables kept in ``[2, 2.5]``) and one without. A
solution claims ancestral pairs among the system variables, with a score for each claim, and the
claims are scored against the model's ancestral pairs (:func:`andrey_bench.local_lane.claim_scores`):
claims, true and false positives, precision, recall, the area under the precision-recall curve,
and the base rate, the precision of claiming every pair, which a method that claims anything has
to beat.

The solutions are Andrey's ``y_structures`` and the authors' code, each as the Y-structure and the
extended Y-structure, on the same data at the same ``alpha``. :data:`ANDREY_ARMS` holds the adapters
of Andrey's method; each needs the contract of :class:`~andrey_bench.local_lane.ClaimAdapter` and
nothing else in the driver. The empty answer claims nothing, so its row is the base rate.

:data:`ROWS` is the paper's ``n = 10000``. One JSON line per fit goes to ``<out>/records.jsonl``
with the claims and their scores, so ``--rescore`` recomputes the scores without fitting;
``<out>/provenance.json`` records the build and the machine. A fit past the per-fit wall cap is a
``timeout`` with no scores. Load R with pcalg as the README's R lane describes, and set
``$ANDREY_BENCH_SBCD_R`` as ``start_sbcd_r.sh`` prints, before running the references.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/y_structures_cpu.py --out outputs/yst
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

from andrey_bench import rsession, rundir, selection_models
from andrey_bench.adapters.andrey_y_structures import AndreyYStructures
from andrey_bench.adapters.versteeg_ystructures import VersteegYStructures
from andrey_bench.local_lane import claim_scores, unpack_answer
from andrey_bench.runner import available_cores, check_interpreters, run_task, split_cores

#: Adapters of Andrey's ``y_structures``, the Y-structure and the extended one. Each follows
#: :class:`~andrey_bench.local_lane.ClaimAdapter`: ``fit`` reads the context in the last column and
#: packs the claims' scores.
ANDREY_ARMS: list = [AndreyYStructures(extended=False), AndreyYStructures(extended=True)]

#: The references: the Y-structure and the extended Y-structure.
REFERENCES = [VersteegYStructures(extended=False), VersteegYStructures(extended=True)]

#: The level of every test, for every solution: the authors' ``simulate.R``.
ALPHA = 0.01

#: The paper's sample size.
ROWS = (10000,)

#: Smallest run that touches every solution: two models of the fixed graph.
SMOKE = {"settings": ["fixed"], "rows": [10000], "seeds": 2, "cap_wall_s": 300.0}

RECORDS = "records.jsonl"

#: The two datasets of a model.
SELECTIONS = ("biased", "unbiased")


def solutions() -> list:
    """Every solution of the lane, in the order records are written."""
    return [*ANDREY_ARMS, *REFERENCES]


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _read(out: Path) -> list[dict]:
    path = out / RECORDS
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _fit_params(adapter, alpha: float) -> dict:
    params = dict(adapter.params())
    if "alpha" in params:
        params["alpha"] = alpha
    return params


def claims_matrix(entries: list[list[float]], d: int) -> np.ndarray:
    """The ``(d, d)`` scores of a record's ``[i, j, score]`` claims."""
    scores = np.zeros((d, d))
    for i, j, score in entries:
        scores[int(i), int(j)] = score
    return scores


def score_record(record: dict, ancestral: np.ndarray, d: int) -> None:
    """Add ``scores`` to ``record``: its claims against the model's ancestral pairs."""
    record["scores"] = claim_scores(claims_matrix(record["claims"], d), ancestral, np.arange(d - 1))


def rescore(out: Path, meta: dict) -> list[dict]:
    """Recompute every record's scores from its stored claims and rewrite the records."""
    if meta.get("sampler") != selection_models.SAMPLER:
        raise SystemExit(
            f"--rescore: the run's data came from {meta.get('sampler', 'an earlier sampler')}, "
            f"not {selection_models.SAMPLER}; rescore it with the build that recorded it"
        )
    records, truths = _read(out), {}
    for record in records:
        if record.get("claims") is None:
            continue
        key = (record["setting"], record["seed"])
        if key not in truths:  # the model, so the truth, does not depend on n
            truths[key] = selection_models.sample(key[0], n=2, seed=key[1]).ancestral
        score_record(record, truths[key], record["d"])
    (out / RECORDS).write_text("".join(json.dumps(r) + "\n" for r in records))
    return records


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Y-structures under selection bias, the authors' code")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--settings",
        nargs="+",
        default=list(selection_models.SETTINGS),
        help="the paper's settings (default: fixed small large)",
    )
    p.add_argument("--rows", type=int, nargs="+", default=list(ROWS), help="sample sizes")
    p.add_argument("--seeds", type=int, default=20, help="models per setting (default: 20)")
    p.add_argument("--first-seed", type=int, default=0, help="the first seed (default: 0)")
    p.add_argument("--alpha", type=float, default=ALPHA, help="level of every test (default: 0.01)")
    p.add_argument("--part", type=int, default=0, help="run every --parts-th model from this one")
    p.add_argument("--parts", type=int, default=1, help="parts the models are split into")
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
        help="run the fixed graph, 10,000 rows, two seeds, and a 300 s cap, replacing those flags",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue the run already in --out: skip the fits whose records are on disk",
    )
    p.add_argument(
        "--rescore",
        action="store_true",
        help="fit nothing: recompute the scores of the run already in --out from its stored claims",
    )
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE)
    unknown = sorted(set(args.settings) - set(selection_models.SETTINGS))
    if unknown:
        p.error(f"unknown settings {unknown}; the settings are {list(selection_models.SETTINGS)}")

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
        records = rescore(out, json.loads(meta_path.read_text()))
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
        "params": {a.name: _fit_params(a, args.alpha) for a in field},
        "r environment": (
            rsession.check_environment("pcalg") if any(a.package == "sbcd" for a in field) else {}
        ),
        "sampler": selection_models.SAMPLER,
        "weight_range": list(selection_models.WEIGHT_RANGE),
        "interval": list(selection_models.INTERVAL),
        "settings": args.settings,
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
    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())
    _log(f"=== Y-structures under selection bias -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:12s}: {value}")
    _log(f"{'git_sha':12s}: {prov['git_sha']}")
    _log("")

    done = {
        (r["setting"], r["n"], r["seed"], r["selection"], r["solution"], r["repeat"])
        for r in _read(out)
    }
    units = [
        (setting, seed)
        for setting in args.settings
        for seed in range(args.first_seed, args.first_seed + args.seeds)
    ][args.part :: args.parts]
    start = time.perf_counter()
    with (out / RECORDS).open("a") as sink:
        for setting, seed in units:
            walls: dict[str, list[float]] = {a.name: [] for a in field}
            for n in args.rows:
                draw = selection_models.sample(setting, n=n, seed=seed)
                d = draw.biased.shape[1]
                for selection in SELECTIONS:
                    data = getattr(draw, selection)
                    for adapter in field:
                        key = (setting, n, seed, selection, adapter.name)
                        if all((*key, r) in done for r in range(args.repeats)):
                            continue
                        params = _fit_params(adapter, args.alpha)
                        threads, n_jobs = split_cores(args.cores, adapter.mode)
                        measures = run_task(
                            adapter,
                            data,
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
                                "setting": setting,
                                "n": n,
                                "seed": seed,
                                "selection": selection,
                                "d": d,
                                "acceptance": draw.acceptance,
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
                                "n_tests": None,
                                "claims": None,
                            }
                            if m.get("status") == "ok" and m.get("adj") is not None:
                                scores, tests = unpack_answer(np.asarray(m["adj"]))
                                record["n_tests"] = tests
                                record["claims"] = [
                                    [int(i), int(j), float(scores[i, j])]
                                    for i, j in zip(*np.nonzero(scores > 0), strict=True)
                                ]
                                score_record(record, draw.ancestral, d)
                                walls[adapter.name].append(m["wall_s"])
                            sink.write(json.dumps(record, default=str) + "\n")
                            sink.flush()
            cost = "  ".join(
                f"{a}={statistics.median(w):.3g}s" if w else f"{a}=no new fit finished"
                for a, w in walls.items()
            )
            _log(f"[{setting} seed={seed}] {cost} | elapsed {time.perf_counter() - start:.0f}s")
    _summarize(_read(out))
    return 0


def _summarize(records: list[dict]) -> None:
    """Print, per setting, size, selection, and solution, the finished fits, the median time, and
    the claims.

    Precision pools the claims of the setting's models; ``base`` and ``ap`` are the means of the
    models' base rate and average precision, and ``claims`` the mean number of claims per model.
    """
    if not records:
        return
    _log("")
    _log(
        f"{'solution':14s} {'setting':8s} {'n':>6s} {'selection':9s} {'ok/all':>7s} {'wall_s':>8s} "
        f"{'claims':>7s} {'prec':>6s} {'base':>6s} {'ap':>6s}"
    )
    keys = sorted({(r["setting"], r["n"], r["selection"]) for r in records})
    for setting, n, selection in keys:
        for name in sorted({r["solution"] for r in records}):
            mine = [
                r
                for r in records
                if (r["setting"], r["n"], r["selection"], r["solution"])
                == (setting, n, selection, name)
            ]
            fit = [r for r in mine if r["status"] == "ok"]
            if not fit:
                continue
            tp, claims = (
                sum(r["scores"]["tp"] for r in fit),
                sum(r["scores"]["claims"] for r in fit),
            )
            aps = [r["scores"]["average_precision"] for r in fit]
            _log(
                f"{name:14s} {setting:8s} {n:6d} {selection:9s} {len(fit):3d}/{len(mine):<3d} "
                f"{statistics.median(r['wall_s'] for r in fit):8.3g} "
                f"{claims / len(fit):7.3g} {tp / claims if claims else float('nan'):6.3f} "
                f"{statistics.mean(r['scores']['base_rate'] for r in fit):6.3f} "
                f"{statistics.mean(aps):6.3f}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
