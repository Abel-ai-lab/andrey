#!/usr/bin/env python
"""Judge the set learners of the blanket lane per cell, from the records of ``blanket_cpu.py``.

Each run folder is named by its arm, ``ARM=FOLDER``: one arm is one set of lane settings (the
bnlearn test, ``max_depth``), and a record is read as ``(arm, solution)``. :data:`COMPARISONS`
lists what is compared with what; a comparison marked judged is the method against its reference
on the same data with the matching test (bnlearn, or the authors' Causal Explorer), the others are
context.

Per dataset, over the targets whose true set is not empty (the same targets on both sides):

- ``distance``: the mean of the distance of each target's precision and recall from (1, 1); an
  empty answer has no wrong member, so it counts precision 1 and distance 1;
- ``f1``: the mean of each target's F1, 2 tp / (found + true), 0 for an empty answer;
- ``errors``: the mean of each target's missing plus extra members, the wrong answers of
  :func:`andrey.metrics.target_answer_scores`; the empty set's is the true set's size;
- ``precision`` and ``recall``: the means of each target's, reported beside the verdicts; an
  empty answer counts precision 1.

A target one side could not answer (``unanswered`` in its record, such as a target at which Causal
Explorer stops with an error) is left out of both sides of that dataset.

:func:`~andrey_bench.pairing.compare` gives the mean of the per-dataset difference, Andrey minus
the other, with its percentile-bootstrap 95% interval; a verdict needs
:data:`~andrey_bench.pairing.MIN_PAIRS` paired datasets. A cell is worse on a measure when the
interval lies wholly on its worse side of 0 on the first seeds and again on the fresh seeds
(``--fresh``); a cell worse on the first seeds only is a tie, and a cell better on the first
seeds is better. The floor is ``errors`` minus the empty set's, which a method passes when the
interval lies wholly below 0.

    $ANDREY_BENCH_PYTHON benchmarks/blanket_summary.py mb-adf=RUN pc-adf=RUN ... --json cells.json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from andrey_bench.pairing import MIN_PAIRS, compare

#: The measures per dataset and whether a higher value is better.
MEASURES: dict[str, bool] = {"distance": False, "f1": True}


@dataclass(frozen=True)
class Comparison:
    """One method's arm against another arm on the same datasets."""

    method: str
    against: str
    andrey: tuple[str, str]
    other: tuple[str, str]
    judged: bool = False


COMPARISONS: tuple[Comparison, ...] = (
    Comparison("gs", "bnlearn gs", ("mb-adf", "andrey.gs"), ("mb-adf", "bnlearn.gs"), True),
    Comparison("gs", "bnlearn gs, plain test", ("mb-adf", "andrey.gs"), ("mb-plain", "bnlearn.gs")),
    Comparison("gs", "iamb", ("mb-adf", "andrey.gs"), ("mb-adf", "andrey.iamb")),
    Comparison(
        "fast_iamb",
        "bnlearn fast.iamb",
        ("mb-adf", "andrey.fast_iamb"),
        ("mb-adf", "bnlearn.fast-iamb"),
        True,
    ),
    Comparison(
        "fast_iamb",
        "bnlearn fast.iamb, plain test",
        ("mb-adf", "andrey.fast_iamb"),
        ("mb-plain", "bnlearn.fast-iamb"),
    ),
    Comparison("fast_iamb", "iamb", ("mb-adf", "andrey.fast_iamb"), ("mb-adf", "andrey.iamb")),
    Comparison(
        "fast_iamb", "inter_iamb", ("mb-adf", "andrey.fast_iamb"), ("mb-adf", "andrey.inter_iamb")
    ),
    Comparison(
        "hiton_pc",
        "bnlearn si.hiton.pc",
        ("pc-adf", "andrey.hiton_pc"),
        ("pc-adf", "bnlearn.si-hiton-pc"),
        True,
    ),
    Comparison(
        "hiton_pc",
        "bnlearn si.hiton.pc, plain test",
        ("pc-adf", "andrey.hiton_pc"),
        ("pc-plain", "bnlearn.si-hiton-pc"),
    ),
    Comparison("hiton_pc", "mmpc", ("pc-adf", "andrey.hiton_pc"), ("pc-adf", "andrey.mmpc")),
    Comparison(
        "hiton_pc, max_depth 3", "mmpc", ("hiton3", "andrey.hiton_pc"), ("pc-adf", "andrey.mmpc")
    ),
    Comparison(
        "hiton_pc, max_depth 3",
        "hiton_pc",
        ("hiton3", "andrey.hiton_pc"),
        ("pc-adf", "andrey.hiton_pc"),
    ),
    Comparison(
        "hiton_pc without the rows-per-cell rule",
        "hiton_pc",
        ("norule", "andrey.hiton_pc"),
        ("pc-adf", "andrey.hiton_pc"),
    ),
    Comparison(
        "hiton_mb",
        "Causal Explorer HITON_MB",
        ("hmb", "andrey.hiton_mb"),
        ("hmb", "causal-explorer.hiton-mb"),
        True,
    ),
    Comparison("hiton_mb", "mmmb, max_depth 3", ("hmb", "andrey.hiton_mb"), ("hmb", "andrey.mmmb")),
    *(
        Comparison("hiton_mb", other, ("hmb", "andrey.hiton_mb"), ("ctx", f"andrey.{other}"))
        for other in ("iamb", "inter_iamb", "gs", "pcmb")
    ),
    Comparison(
        "mmmb", "Causal Explorer MMMB", ("mm", "andrey.mmmb"), ("mm", "causal-explorer.mmmb"), True
    ),
    Comparison(
        "mmmb", "hiton_mb, max_depth None", ("mm", "andrey.mmmb"), ("mm", "andrey.hiton_mb")
    ),
    *(
        Comparison("mmmb", other, ("mm", "andrey.mmmb"), ("ctx", f"andrey.{other}"))
        for other in ("iamb", "inter_iamb", "gs", "pcmb")
    ),
    Comparison(
        "hiton_mb, degrees_of_freedom nonempty",
        "Causal Explorer HITON_MB",
        ("hmbne", "andrey.hiton_mb"),
        ("hmb", "causal-explorer.hiton-mb"),
    ),
    Comparison(
        "hiton_mb, degrees_of_freedom nonempty",
        "hiton_mb",
        ("hmbne", "andrey.hiton_mb"),
        ("hmb", "andrey.hiton_mb"),
    ),
    Comparison(
        "mmmb, degrees_of_freedom mean",
        "Causal Explorer MMMB",
        ("mmmean", "andrey.mmmb"),
        ("mm", "causal-explorer.mmmb"),
    ),
    Comparison(
        "mmmb, degrees_of_freedom mean", "mmmb", ("mmmean", "andrey.mmmb"), ("mm", "andrey.mmmb")
    ),
    Comparison("hiton_mb", "mmmb, max_depth 3", ("gh", "andrey.hiton_mb"), ("gh", "andrey.mmmb")),
    *(
        Comparison("hiton_mb", other, ("gh", "andrey.hiton_mb"), ("gctx", f"andrey.{other}"))
        for other in ("iamb", "inter_iamb", "gs", "pcmb")
    ),
)


def load(runs: list[str]) -> dict[tuple[str, str], dict[tuple[str, int], dict]]:
    """Target-mode records by ``(arm, solution)``, then by ``(cell, seed)``; each ``ARM=GLOB``."""
    out: dict[tuple[str, str], dict[tuple[str, int], dict]] = defaultdict(dict)
    for run in runs:
        arm, pattern = run.split("=", 1)
        for folder in sorted(glob.glob(pattern)):
            path = Path(folder) / "records.jsonl"
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                r = json.loads(line)
                if r["mode"] == "target" and r["package"] != "baseline":
                    out[arm, r["solution"]][r["cell"], r["seed"]] = r
    return out


def measures(record: dict) -> dict[str, float]:
    """``distance``, ``f1``, ``errors`` and ``empty_errors`` of one fit (see the module notes)."""
    sets = [s for s in record["per_target"].values() if s["true"]]
    dist = [s["distance"] if s["est"] else 1.0 for s in sets]
    return {
        "distance": float(np.mean(dist)),
        "f1": float(np.mean([2 * s["tp"] / (s["est"] + s["true"]) for s in sets])),
        "precision": float(np.mean([s["tp"] / s["est"] if s["est"] else 1.0 for s in sets])),
        "recall": float(np.mean([s["tp"] / s["true"] for s in sets])),
        "errors": float(np.mean([s["fp"] + s["fn"] for s in sets])),
        "empty_errors": float(np.mean([s["true"] for s in sets])),
        "tests": float(
            np.mean([v for v in record["n_tests"].values() if v is not None] or [np.nan])
        ),
    }


def _pairs(a: dict, b: dict, seeds: range) -> list[tuple[int, dict, dict]]:
    """The datasets both sides fitted, each record kept to the targets both answered."""
    out = []
    for (cell, seed), ra in sorted(a.items()):
        rb = b.get((cell, seed))
        if seed in seeds and rb and ra["status"] == "ok" and rb["status"] == "ok":
            assert ra["dataset"] == rb["dataset"] and ra["d"] == rb["d"], (cell, seed)
            both = set(ra["per_target"]) & set(rb["per_target"])
            out.append((seed, _only(ra, both), _only(rb, both)))
    return out


def _only(record: dict, targets: set[str]) -> dict:
    if set(record["per_target"]) == targets:
        return record
    return {**record, "per_target": {t: s for t, s in record["per_target"].items() if t in targets}}


def judge(pairs: list[tuple[int, dict, dict]]) -> dict:
    """The paired verdicts of ``a`` against ``b`` on each measure, with the per-dataset D."""
    out: dict = {"paired": len(pairs)}
    ma = [measures(ra) for _, ra, _ in pairs]
    mb = [measures(rb) for _, _, rb in pairs]
    for name, higher in MEASURES.items():
        va = [m[name] for m in ma]
        vb = [m[name] for m in mb]
        c = compare(va, vb, higher_is_better=higher) if pairs else {"verdict": "tie"}
        if len(pairs) < MIN_PAIRS:
            c["verdict"] = "too few pairs"
        c["andrey_mean"] = round(float(np.mean(va)), 4) if pairs else None
        c["other_mean"] = round(float(np.mean(vb)), 4) if pairs else None
        c["D_by_seed"] = {str(s): round(x - y, 4) for (s, _, _), x, y in zip(pairs, va, vb)}
        out[name] = c
    out["same_sets"] = round(
        float(np.mean([ra["sets"] == rb["sets"] for _, ra, rb in pairs])) if pairs else 0.0, 3
    )
    return out


def floor(fits: list[dict]) -> dict:
    """``errors`` of each fit minus the empty set's, with its bootstrap interval."""
    m = [measures(r) for r in fits]
    c = compare([x["errors"] for x in m], [x["empty_errors"] for x in m])
    return {
        "paired": c["paired"],
        "mean": c["mean_D"],
        "bootstrap_95": c["bootstrap_95"],
        "pass": c["paired"] >= MIN_PAIRS and c["bootstrap_95"][1] < 0,
    }


def _published(first: dict, fresh: dict | None) -> str:
    """The published verdict of one measure: worse only when the fresh seeds agree."""
    if first["verdict"] != "worse":
        return first["verdict"]
    if fresh is None:
        return "worse, not rerun"
    return "worse" if fresh["verdict"] == "worse" else "tie"


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 4) if values else None


def cell(comp: Comparison, cell: str, a: dict, b: dict, first: range, fresh: range) -> dict:
    """One cell of one comparison: its verdicts on the first and the fresh seeds, the floor of each
    side, pooled scores, test counts, and times."""
    a = {k: r for k, r in a.items() if k[0] == cell}
    b = {k: r for k, r in b.items() if k[0] == cell}
    pairs = _pairs(a, b, first)
    row: dict = {"method": comp.method, "against": comp.against, "judged": comp.judged}
    some = next(iter(a.values()))
    row.update(cell=cell, n=some["n"], d=some["d"])
    datasets = {s for (_, s), r in a.items() if s in first}
    ok_a = {s for (_, s), r in a.items() if s in first and r["status"] == "ok"}
    ok_b = {s for (_, s), r in b.items() if s in first and r["status"] == "ok"}
    row["coverage"] = {
        "datasets": len(datasets),
        "andrey_short": len(ok_b - ok_a),
        "reference_short": len(ok_a - ok_b),
        "andrey_status": dict(
            sorted(_count(r["status"] for (_, s), r in a.items() if s in first).items())
        ),
        "unanswered_targets": sum(
            len(r.get("unanswered", []))
            for side in (a, b)
            for (_, s), r in side.items()
            if s in first
        ),
        "other_status": dict(
            sorted(_count(r["status"] for (_, s), r in b.items() if s in first).items())
        ),
    }
    row["first"] = judge(pairs)
    fresh_pairs = _pairs(a, b, fresh)
    row["fresh"] = judge(fresh_pairs) if len(fresh_pairs) else None
    row["floor"] = {
        "andrey_minus_empty": floor([ra for _, ra, _ in pairs]),
        "other_minus_empty": floor([rb for _, _, rb in pairs]),
    }
    for name in MEASURES:
        fresh_m = row["fresh"][name] if row["fresh"] else None
        row[f"{name}_result"] = _published(row["first"][name], fresh_m)
    results = [row[f"{name}_result"] for name in MEASURES]
    row["result"] = next(
        (r for r in ("worse", "worse, not rerun", "better", "too few pairs") if r in results), "tie"
    )
    for side, i in (("andrey", 1), ("other", 2)):
        fits = [p[i] for p in pairs]
        row[f"{side}_tests_per_target"] = (
            round(float(np.nanmean([measures(r)["tests"] for r in fits])), 1) if fits else None
        )
        row[f"{side}_s"] = _median([r["wall_s"] for r in fits if r.get("wall_s") is not None])
        for name in ("errors", "precision", "recall"):
            row[f"{side}_{name}_mean"] = (
                round(float(np.mean([measures(r)[name] for r in fits])), 4) if fits else None
            )
    row["empty_errors_mean"] = (
        round(float(np.mean([measures(ra)["empty_errors"] for _, ra, _ in pairs])), 4)
        if pairs
        else None
    )
    return row


def _count(items) -> dict:
    out: dict = defaultdict(int)
    for x in items:
        out[x] += 1
    return out


def summarize(records: dict, first: range, fresh: range) -> list[dict]:
    """Every cell of every comparison whose two arms both have records there."""
    rows = []
    for comp in COMPARISONS:
        a, b = records.get(comp.andrey, {}), records.get(comp.other, {})
        for name in sorted({k[0] for k in a} & {k[0] for k in b}, key=_cell_order):
            rows.append(cell(comp, name, a, b, first, fresh))
    return rows


#: The release summary's metrics, each an error (fewer is better), by measure.
SUMMARY_METRICS: dict[str, str] = {"distance": "set_distance", "f1": "set_f1_loss"}


def _summary_side(judged: dict, measure: str) -> dict:
    """One measure of :func:`judge` as the release summary's error fields, ``andrey_<metric>_mean``
    and ``comparator_<metric>_mean``, with F1 turned into its loss, 1 - F1."""
    metric = SUMMARY_METRICS[measure]
    m = judged[measure]
    if MEASURES[measure]:
        a, b = 1 - m["andrey_mean"], 1 - m["other_mean"]
        mean_d, (lo, hi) = -m["mean_D"], m["bootstrap_95"]
        interval = [round(-hi, 4), round(-lo, 4)]
    else:
        a, b, mean_d, interval = m["andrey_mean"], m["other_mean"], m["mean_D"], m["bootstrap_95"]
    return {
        "paired": judged["paired"],
        f"andrey_{metric}_mean": round(a, 4),
        f"comparator_{metric}_mean": round(b, 4),
        "mean_D": round(mean_d, 4),
        "bootstrap_95": interval,
        "above_zero": interval[0] > 0,
    }


def summary_rows(rows: list[dict], measure: str, fresh: str) -> list[dict]:
    """The judged cells of one comparison as rows of the release summary for one measure."""
    out = []
    for r in rows:
        side = _summary_side(r["first"], measure)
        key = {"d": r["d"], "n": r["n"]}
        if r["cell"].startswith("gauss:"):
            parts = r["cell"].split(":")
            key = {"mean_degree": float(parts[3]) if len(parts) > 3 else 2.0, **key}
        else:
            key = {"network": r["cell"].split(":")[1], **key}
        fl = r["floor"]
        applies = fl["other_minus_empty"]["bootstrap_95"][1] < 0
        row = {
            **key,
            "seeds": r["coverage"]["datasets"],
            **side,
            "verdict": r["first"][measure]["verdict"],
            "coverage": {
                k: r["coverage"][k] for k in ("datasets", "andrey_short", "reference_short")
            },
            "floor": {
                "applies": applies,
                "pass": fl["andrey_minus_empty"]["bootstrap_95"][1] < 0 if applies else None,
                "andrey_minus_empty": {
                    k: fl["andrey_minus_empty"][k] for k in ("paired", "mean", "bootstrap_95")
                },
                "reference_minus_empty": {
                    k: fl["other_minus_empty"][k] for k in ("paired", "mean", "bootstrap_95")
                },
            },
            "andrey_s": r["andrey_s"],
            "comparator_s": r["other_s"],
            "result": r[f"{measure}_result"],
        }
        if r["fresh"] and r["first"][measure]["verdict"] == "worse":
            row["replication"] = {"seeds": fresh, **_summary_side(r["fresh"], measure)}
        out.append(row)
    return out


def _cell_order(name: str) -> tuple:
    parts = name.split(":")
    return (parts[0], parts[1], *(int(p) for p in parts[2:]))


def _seeds(text: str) -> range:
    first, last = (int(x) for x in text.split("-"))
    return range(first, last + 1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Blanket lane: the set learners judged per cell")
    p.add_argument("runs", nargs="+", help="ARM=GLOB of run folders of blanket_cpu.py")
    p.add_argument("--first", default="0-19", help="seeds judged first (default: 0-19)")
    p.add_argument("--fresh", default="20-39", help="seeds of the reruns (default: 20-39)")
    p.add_argument("--json", default=None, help="also write the cells to this file")
    p.add_argument(
        "--summary-rows",
        default=None,
        help="also write the judged cells as release-summary rows, by method, regime, and metric",
    )
    args = p.parse_args(argv)
    rows = summarize(load(args.runs), _seeds(args.first), _seeds(args.fresh))
    if args.summary_rows:
        tables: dict = defaultdict(dict)
        for comp in (c for c in COMPARISONS if c.judged):
            mine = [r for r in rows if r["method"] == comp.method and r["against"] == comp.against]
            for regime, picked in (
                ("discrete", [r for r in mine if r["cell"].startswith("net:")]),
                ("gaussian", [r for r in mine if r["cell"].startswith("gauss:")]),
            ):
                for measure, metric in SUMMARY_METRICS.items():
                    tables[comp.method][f"{regime}|{metric}"] = summary_rows(
                        picked, measure, args.fresh
                    )
        Path(args.summary_rows).write_text(json.dumps(tables, indent=1))
    for r in rows:
        f = r["first"]
        print(
            f"{r['method']:<40} {r['against']:<32} {r['cell']:<22} n={f['paired']:>2} "
            f"dist {f['distance']['andrey_mean']}/{f['distance']['other_mean']} "
            f"{f['distance']['verdict']:<5} f1 {f['f1']['andrey_mean']}/{f['f1']['other_mean']} "
            f"{f['f1']['verdict']:<5} -> {r['result']}"
            f"{'' if r['floor']['andrey_minus_empty']['pass'] else ' FLOOR'}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
