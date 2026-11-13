#!/usr/bin/env python
"""Benchmark the treatment-outcome methods: effect answers, adjustment sets, and effect errors.

Each cell of the lane is a planted model (:mod:`andrey_bench.adjustment_models`, one generator per
paper of the group) at a sample size. For every seed the driver draws the model, the rows the
methods see, and a fresh sample of the same model, and fits each solution once per treatment-outcome
pair of the model. A solution answers with ``effects`` records (:mod:`andrey_bench.adjustment_lane`):
whether the effect is nonzero, whether it is identifiable, and the adjustment sets. The answer is
scored against the oracle the method's own definition gives on the true graph
(:func:`andrey.metrics.effect_oracle`, :func:`andrey.metrics.effect_scores`), each set against the
criterion and the optimal set (:func:`andrey.metrics.adjustment_scores`), and the effect estimate of
each set, least squares on the fresh sample, against the model's true effect. A solution that gives
no set and does not say the effect is zero abstains and has no effect error: abstentions are never
scored as the empty set, and the summary reports the answer rate and compares the effect error with
the empty arm on the datasets the solution answered.

The solutions are Andrey's methods (:data:`ANDREY_ARMS`, each the row of a table that names it), the
PC arm every paper reports, the two baselines (adjust for nothing, adjust for everything), and the
authors' code of the reference solutions (:data:`REFERENCES`), each at the paper's or the code's own
settings. A solution runs on the cells that ask for its estimand and, when it identifies the effect
from the Markov equivalence class, on those without hidden variables.

One JSON line per fit goes to ``<out>/records.jsonl`` with the answer and its scores, so
``--rescore`` recomputes the scores without fitting; ``<out>/provenance.json`` records the build and
the machine. A fit past the per-fit wall cap is a ``timeout`` with no answer, never an empty one.
The references need the environments the ``start_*.sh`` scripts build; export what they print.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/adjustment_cpu.py --out outputs/adj
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from andrey_bench import adjustment_models as M
from andrey_bench import rundir
from andrey_bench.adapters import andrey_adjustment
from andrey_bench.adapters.andrey_adjustment import AndreyAdjustment, AndreyPCAdjustment
from andrey_bench.adapters.baseline_adjustment import AdjustForEverything, AdjustForNothing
from andrey_bench.adapters.gupta_ldecc import GuptaLDECC
from andrey_bench.adapters.maasch_ldp import MaaschLD3, MaaschLDP
from andrey_bench.adapters.schubert_load import SchubertLOAD
from andrey_bench.adapters.schubert_snap import SchubertSNAP
from andrey_bench.adjustment_lane import (
    ESTIMAND_PARAM,
    OUTCOME_PARAM,
    TREATMENT_PARAM,
    VARIABLES_PARAM,
    answer_records,
    error_key,
    oracle_records,
    score_answer,
)
from andrey_bench.runner import available_cores, check_interpreters, run_task, split_cores

#: The baselines and the PC arm: no method of Andrey needs to exist for them to run.
FIXED_ARMS: list = [AdjustForNothing(), AdjustForEverything(), AndreyPCAdjustment()]

#: The references, the authors' code each in its own environment (the ``start_*.sh`` scripts).
REFERENCES: list = [GuptaLDECC(), MaaschLDP(), MaaschLD3(), SchubertSNAP(), SchubertLOAD()]

#: The claimed settings of the papers at sizes they report, one row of cells per paper; a campaign
#: chooses its own grid.
DEFAULT_CELLS: tuple[str, ...] = (
    *(
        f"entner-{task}@{n}"
        for task in ("effect-admissible", "effect-none", "zero-admissible", "zero-none")
        for n in (100, 1000, 10000)
    ),
    *(f"locpc-{kind}@5000,v={v}" for kind in ("identifiable", "nonidentifiable") for v in (10, 50)),
    "ldp@1000",
    "ldp@5000",
    "ld3-dag@1000",
    "ld3-dag@10000",
    "ldecc@8000",
    "ldecc@32000",
    "random-graph@1000,degree=3,nodes=100",
    "random-graph@10000,degree=2,nodes=100",
    "magic-niab@8000",
    *(f"sangiovese@{n}" for n in (250, 500, 1000)),
    "mildew@25000",
)

#: Smallest run that touches every kind of cell and every reference: two models of each.
SMOKE = {
    "cells": [
        "entner-effect-admissible@1000",
        "ldecc@2000",
        "ldp@1000",
        "ld3-dag@1000",
        "locpc-identifiable@2000,v=10",
    ],
    "seeds": 2,
    "cap_wall_s": 300.0,
}

RECORDS = "records.jsonl"


def andrey_arms() -> list:
    """The adapters of Andrey's methods the installed Andrey has (:func:`andrey_adjustment.available`)."""
    return [AndreyAdjustment(name) for name in andrey_adjustment.available()]


def solutions() -> list:
    """Every solution of the lane, in the order records are written: Andrey's methods, the PC arm
    and the baselines, then the references."""
    return [*andrey_arms(), *FIXED_ARMS, *REFERENCES]


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _read(out: Path) -> list[dict]:
    path = out / RECORDS
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def kind_of(adapter: Any, estimand: str) -> dict[str, Any]:
    """What the oracle of ``adapter`` on a cell of ``estimand`` is built from.

    The arms that answer either estimand (``"any"``) keep their own kind of set for a total effect
    and are judged against the outcome's parents for a direct one.
    """
    set_kind = adapter.set_kind
    if adapter.estimand == "any" and estimand == "direct":
        set_kind = "parents_of_outcome"
    return {
        "estimand": estimand,
        "set_kind": set_kind,
        "identified_from": adapter.identified_from,
        "unordered": bool(adapter.unordered),
    }


def runs_on(adapter: Any, model: M.LinearModel) -> bool:
    """Whether ``adapter`` answers the question ``model`` asks: its estimand, the test the cell
    gives (``adapter.indep_tests``, every test when absent), and no hidden variable for a method
    that identifies the effect from the Markov equivalence class."""
    estimands = {pair.estimand for pair in model.pairs}
    if adapter.estimand != "any" and adapter.estimand not in estimands:
        return False
    if model.indep_test not in getattr(adapter, "indep_tests", (model.indep_test,)):
        return False
    return not (adapter.identified_from == "cpdag" and model.hidden)


def pair_params(adapter: Any, pair: M.Pair) -> dict[str, Any]:
    """The pair's columns as the adapter reads them, and the estimand the cell asks for."""
    cols = (
        {VARIABLES_PARAM: sorted([pair.treatment, pair.outcome])}
        if adapter.unordered
        else {TREATMENT_PARAM: pair.treatment, OUTCOME_PARAM: pair.outcome}
    )
    return {**cols, ESTIMAND_PARAM: pair.estimand}


def keep_asked(records: list[dict], pair: M.Pair, unordered: bool) -> tuple[list[dict], int]:
    """The records that answer the pair asked, and how many others the solution gave."""
    asked = {(pair.treatment, pair.outcome)} | (
        {(pair.outcome, pair.treatment)} if unordered else set()
    )
    kept = [r for r in records if (r["treatment"], r["outcome"]) in asked]
    return kept, len(records) - len(kept)


def score_record(
    record: dict, model: M.LinearModel, pair: M.Pair, fresh: np.ndarray | None
) -> None:
    """Add ``scores`` to ``record``: its answer scored on the model's truth for its pair."""
    kind = record["answer_kind"]
    oracle = oracle_records(
        model.dag,
        model.hidden,
        (pair.treatment, pair.outcome),
        estimand=kind["estimand"],
        set_kind=kind["set_kind"],
        identified_from=kind["identified_from"],
        unordered=kind["unordered"],
    )
    record["scores"] = score_answer(
        record["answer"], oracle, model.dag, model.hidden, fresh, model.effects()
    )


def rescore(out: Path, meta: dict) -> list[dict]:
    """Recompute every record's scores from its stored answer and rewrite the records."""
    if meta.get("sampler") != M.SAMPLER:
        raise SystemExit(
            f"--rescore: the run's data came from {meta.get('sampler', 'an earlier sampler')}, "
            f"not {M.SAMPLER}; rescore it with the build that recorded it"
        )
    records, models = _read(out), {}
    for record in records:
        if record.get("answer") is None:
            continue
        cell = M.parse_cell(record["cell"])
        key = (record["cell"], record["seed"])
        if key not in models:
            model = cell.model(record["seed"])
            fresh = None if model.discrete else model.rows(cell.n, M.FRESH)
            models[key] = (model, fresh)
        model, fresh = models[key]
        score_record(record, model, model.pairs[record["pair"]], fresh)
    (out / RECORDS).write_text("".join(json.dumps(r) + "\n" for r in records))
    return records


def _fit_params(adapter: Any, alpha: float | None, max_depth: int | None) -> dict:
    params = dict(adapter.params())
    if alpha is not None and "alpha" in params:
        params["alpha"] = alpha
    if max_depth is not None and getattr(adapter, "takes_max_depth", False):
        params["max_depth"] = max_depth
    return params


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Treatment-outcome methods: effects and adjustment sets"
    )
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--cells",
        nargs="+",
        default=list(DEFAULT_CELLS),
        help="cells as setting@n[,key=value...]; settings: " + " ".join(M.SETTINGS),
    )
    p.add_argument("--seeds", type=int, default=20, help="models per cell (default: 20)")
    p.add_argument("--first-seed", type=int, default=0, help="the first seed (default: 0)")
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="level of every test, for every solution that has one (default: each solution's own)",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="largest conditioning set of any test, for the solutions that take it (default: none)",
    )
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
        help="run one cell of each kind, two seeds, and a 300 s cap, replacing those flags",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue the run already in --out: skip the fits whose records are on disk",
    )
    p.add_argument(
        "--rescore",
        action="store_true",
        help="fit nothing: recompute the scores of the run already in --out from its stored answers",
    )
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE)
    try:
        cells = [M.parse_cell(text) for text in args.cells]
    except ValueError as error:
        p.error(str(error))

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
        "params": {a.name: _fit_params(a, args.alpha, args.max_depth) for a in field},
        "sampler": M.SAMPLER,
        "cells": [c.key for c in cells],
        "seeds": args.seeds,
        "first_seed": args.first_seed,
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
    if args.resume and meta_path.exists() and json.loads(meta_path.read_text()) != run_meta:
        raise SystemExit(f"--resume: {meta_path} differs from these flags; pass the first run's")
    meta_path.write_text(json.dumps(run_meta, indent=2))
    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())
    _log(f"=== treatment-outcome methods -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:12s}: {value}")
    _log(f"{'git_sha':12s}: {prov['git_sha']}")
    _log("")

    done = {(r["cell"], r["seed"], r["pair"], r["solution"], r["repeat"]) for r in _read(out)}
    units = [
        (cell, seed)
        for cell in cells
        for seed in range(args.first_seed, args.first_seed + args.seeds)
    ][args.part :: args.parts]
    start = time.perf_counter()
    with (out / RECORDS).open("a") as sink:
        for cell, seed in units:
            model = cell.model(seed)
            data = model.rows(cell.n, M.DATA)
            fresh = None if model.discrete else model.rows(cell.n, M.FRESH)
            walls: dict[str, list[float]] = {}
            for index, pair in enumerate(model.pairs):
                for adapter in field:
                    if not runs_on(adapter, model):
                        continue
                    key = (cell.key, seed, index, adapter.name)
                    if all((*key, r) in done for r in range(args.repeats)):
                        continue
                    params = {
                        **_fit_params(adapter, args.alpha, args.max_depth),
                        **pair_params(adapter, pair),
                    }
                    if "indep_test" in params:
                        params["indep_test"] = model.indep_test
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
                            "cell": cell.key,
                            "setting": cell.setting,
                            "n": cell.n,
                            "seed": seed,
                            "pair": index,
                            "treatment": pair.treatment,
                            "outcome": pair.outcome,
                            "estimand": pair.estimand,
                            "d": model.p,
                            "hidden": len(model.hidden),
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
                            "answer_kind": kind_of(adapter, pair.estimand),
                            "n_tests": None,
                            "answer": None,
                        }
                        if m.get("status") == "ok" and m.get("adj") is not None:
                            answered, tests = answer_records(adapter, np.asarray(m["adj"]), params)
                            record["answer"], record["unasked"] = keep_asked(
                                answered, pair, adapter.unordered
                            )
                            record["n_tests"] = tests
                            score_record(record, model, pair, fresh)
                            walls.setdefault(adapter.name, []).append(m["wall_s"])
                        sink.write(json.dumps(record, default=str) + "\n")
                        sink.flush()
            cost = "  ".join(f"{a}={statistics.median(w):.3g}s" for a, w in walls.items())
            _log(
                f"[{cell.key} seed={seed}] {cost or 'no new fit finished'} | elapsed {time.perf_counter() - start:.0f}s"
            )
    _summarize(_read(out))
    return 0


def _mean(values: list[float]) -> float:
    values = [v for v in values if v == v]
    return statistics.mean(values) if values else float("nan")


def asked_record(fit: dict) -> dict | None:
    """The answer record of a fit for the treatment and the outcome it asked, ``None`` when the
    answer has none. A method that does not know the order answers both directions; only this
    record is the answer to the question."""
    asked = (fit["treatment"], fit["outcome"])
    return next((a for a in fit["answer"] if (a["treatment"], a["outcome"]) == asked), None)


def asked_error(fit: dict) -> dict | None:
    """The effect estimate (:func:`andrey_bench.adjustment_lane.effect_estimates`) of the record
    for the treatment and the outcome a fit asked, ``None`` when that record abstains."""
    return (fit["scores"]["error"] or {}).get(error_key(fit["treatment"], fit["outcome"]))


def gave_an_answer(fit: dict) -> bool:
    """Whether a fit's record for the pair asked gives sets or says the effect is zero, rather
    than abstain."""
    record = asked_record(fit)
    return record is not None and bool(record["adjustment_sets"] or record["nonzero"] is False)


def summarize(records: list[dict]) -> list[dict[str, Any]]:
    """One row per cell and solution: fits, answer rate, how the answers fare, and the effect error.

    ``answered`` is the share of fits whose record for the pair asked gave sets or said the effect
    is zero (:func:`gave_an_answer`); ``adj`` counts the adjustment answers (``correct``,
    ``wrong``, ``incomplete``) of the pairs whose effect is not zero; ``valid`` is the share of
    returned sets that satisfy the adjustment criterion and ``f1`` the mean F1 of a returned set
    against the optimal set (total effects, no hidden variables), over every record the fit gave.
    ``error`` is the mean effect error of the record for the pair asked (:func:`asked_error`) over
    the fits that have one, and ``empty`` the unadjusted regression's on the same fits; ``paired``
    is their paired comparison (:func:`andrey_bench.pairing.compare`, ``better`` when the
    solution's error is lower).
    """
    from andrey_bench import pairing

    rows = []
    by_empty: dict[tuple, float | None] = {}
    for r in records:
        if r["solution"] == AdjustForNothing.name and r["status"] == "ok":
            estimate = asked_error(r)
            by_empty[(r["cell"], r["seed"], r["pair"])] = estimate["error"] if estimate else None
    for cell in dict.fromkeys(r["cell"] for r in records):
        for name in dict.fromkeys(r["solution"] for r in records if r["cell"] == cell):
            mine = [r for r in records if r["cell"] == cell and r["solution"] == name]
            ok = [r for r in mine if r["status"] == "ok"]
            sets = [s for r in ok for s in r["scores"]["sets"]]
            errors = {}
            for r in ok:
                if estimate := asked_error(r):
                    errors[(r["seed"], r["pair"])] = estimate["error"]
            paired = [
                (errors[(r["seed"], r["pair"])], by_empty[(cell, r["seed"], r["pair"])])
                for r in ok
                if (r["seed"], r["pair"]) in errors
                and by_empty.get((cell, r["seed"], r["pair"])) is not None
            ]
            adjustment = [r["scores"]["answer"]["adjustment"] for r in ok]
            rows.append(
                {
                    "cell": cell,
                    "solution": name,
                    "fits": len(mine),
                    "ok": len(ok),
                    "answered": (sum(map(gave_an_answer, ok)) / len(ok)) if ok else float("nan"),
                    "adj": {
                        k: sum(a[k] for a in adjustment) for k in ("correct", "wrong", "incomplete")
                    },
                    "valid": _mean([float(s["valid"]) for s in sets]),
                    "f1": _mean([s.get("f1", float("nan")) for s in sets]),
                    "error": _mean(list(errors.values())),
                    "empty": _mean([e for _, e in paired]),
                    "paired": (
                        pairing.compare([a for a, _ in paired], [b for _, b in paired])
                        if len(paired) >= pairing.MIN_PAIRS
                        else None
                    ),
                    "wall_s": _mean([r["wall_s"] for r in ok if r["wall_s"] is not None]),
                }
            )
    return rows


def _summarize(records: list[dict]) -> None:
    """Print :func:`summarize` as a table."""
    if not records:
        return
    _log("")
    _log(
        f"{'cell':34s} {'solution':26s} {'ok/all':>7s} {'answ':>5s} {'right':>5s} {'wrong':>5s} "
        f"{'valid':>5s} {'f1':>5s} {'error':>6s} {'empty':>6s} {'vs empty':>9s} {'wall_s':>7s}"
    )
    for row in summarize(records):
        verdict = row["paired"]["verdict"] if row["paired"] else "-"
        _log(
            f"{row['cell']:34s} {row['solution']:26s} {row['ok']:3d}/{row['fits']:<3d} "
            f"{row['answered']:5.2f} {row['adj']['correct']:5d} {row['adj']['wrong']:5d} "
            f"{row['valid']:5.2f} {row['f1']:5.2f} {row['error']:6.3f} {row['empty']:6.3f} "
            f"{verdict:>9s} {row['wall_s']:7.3g}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
