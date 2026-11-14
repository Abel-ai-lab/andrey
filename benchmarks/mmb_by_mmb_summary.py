#!/usr/bin/env python
"""Judge ``mmb_by_mmb`` against the authors' code per cell, from the records of ``mmb_by_mmb_cpu.py``.

A cell is one network at one sample size. Per dataset and solution, over the targets whose true
edge set is not empty, each measure is the mean over the targets of:

- ``distance``: the distance of the precision and recall of the edges at the target from (1, 1),
  an edge right only with both marks right; an answer with no edge counts precision 1, so its
  distance is 1;
- ``f1``: 2 correct / (answer's edges + true edges), 0 for an answer with no edge;
- ``errors``: the local endpoint SHD (extra, missing, and mismarked edges at the target); the
  empty answer's is the target's true degree.

A dataset counts for a solution only when every fit of it finished. :func:`judge` gives the mean
of the per-dataset difference, ``mmb_by_mmb`` minus the other solution, with its percentile
bootstrap 95% interval (:func:`~andrey_bench.pairing.compare`). A cell is worse on a measure when
the interval lies wholly on its worse side of 0 on the first seeds and again on the fresh seeds; a
cell worse on the first seeds only is a tie. The floor is ``errors`` minus the empty answer's, which
a solution passes when the interval lies wholly below 0; ``answers`` adds the floor of
:func:`andrey.metrics.target_answer_scores` (adjacency and marks: the net count against the better
of abstaining and the base-rate answer), pooled over the cell.

    $ANDREY_BENCH_PYTHON benchmarks/mmb_by_mmb_summary.py RUN [RUN ...] --json cells.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from andrey_bench import latent_networks, networks
from andrey_bench.local_lane import entries_to_marks
from andrey_bench.pairing import MIN_PAIRS, compare

#: The judged solution and its reference; the others are context.
ANDREY, REFERENCE = "andrey.mmb_by_mmb", "mmb-by-mmb.authors"
EMPTY = "baseline.empty"

#: The measures per dataset and whether a higher value is better.
MEASURES: dict[str, bool] = {"distance": False, "f1": True}

#: The release summary's metrics, each an error (fewer is better), by measure.
SUMMARY_METRICS: dict[str, str] = {"distance": "edge_distance", "f1": "edge_f1_loss"}


def load(runs: list[str]) -> dict[str, dict[tuple[str, int, int], list[dict]]]:
    """``{solution: {(network, n, seed): [records]}}`` over the run folders, each repeat 0."""
    out: dict = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for line in (Path(run) / "records.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("repeat", 0) == 0:
                out[r["solution"]][(r["network"], r["n"], r["seed"])].append(r)
    return out


def target_measures(s: dict) -> dict[str, float]:
    """The measures of one target's scores (:func:`~andrey_bench.local_lane.target_edge_scores`)."""
    est, true, correct = s["est_edges"], s["true_edges"], s["correct_edges"]
    precision = correct / est if est else 1.0
    recall = correct / true
    return {
        "distance": float(np.hypot(1 - precision, 1 - recall)),
        "f1": 2 * correct / (est + true),
        "errors": float(s["shd"]),
        "empty_errors": float(s["empty_shd"]),
    }


def dataset(records: list[dict]) -> dict | None:
    """One solution's measures on one dataset, means over the targets with a true edge; ``None``
    unless every fit finished."""
    if not records or any(r["status"] != "ok" for r in records):
        return None
    scores = {t: s for r in records for t, s in r["scores"].items() if s["true_edges"] > 0}
    per = [target_measures(s) for s in scores.values()]
    if not per:
        return None
    out = {k: float(np.mean([m[k] for m in per])) for k in per[0]}
    tests = [r["n_tests"] for r in records if r.get("n_tests") is not None]
    out["tests"] = float(np.mean(tests)) if tests else None
    out["wall_s"] = float(np.mean([r["wall_s"] for r in records]))
    out["targets"] = sorted(scores)
    return out


def judge(a: dict[int, dict], b: dict[int, dict], seeds: range) -> dict:
    """The paired verdicts of ``a`` against ``b`` on the datasets of ``seeds`` both finished."""
    paired = sorted(s for s in seeds if a.get(s) and b.get(s))
    out: dict = {"paired": len(paired)}
    for name, higher in MEASURES.items():
        va, vb = [a[s][name] for s in paired], [b[s][name] for s in paired]
        c = compare(va, vb, higher_is_better=higher) if paired else {"verdict": "tie"}
        if len(paired) < MIN_PAIRS:
            c["verdict"] = "too few pairs"
        c["andrey_mean"] = round(float(np.mean(va)), 4) if paired else None
        c["other_mean"] = round(float(np.mean(vb)), 4) if paired else None
        c["D_by_seed"] = {str(s): round(x - y, 4) for s, x, y in zip(paired, va, vb, strict=True)}
        out[name] = c
    return out


def floor(fits: list[dict]) -> dict:
    """``errors`` of each dataset minus the empty answer's, with its bootstrap interval."""
    c = compare([f["errors"] for f in fits], [f["empty_errors"] for f in fits])
    return {
        "paired": c["paired"],
        "mean": c["mean_D"],
        "bootstrap_95": c["bootstrap_95"],
        "pass": c["paired"] >= MIN_PAIRS and c["bootstrap_95"][1] < 0,
    }


def answer_floor(records: list[dict], truths: dict) -> dict:
    """The cell's pooled :func:`andrey.metrics.target_answer_scores`: per part (adjacency, marks),
    the net count and the floor's (the better of abstaining and the base-rate answer)."""
    from andrey import GraphStructure
    from andrey.metrics import target_answer_scores

    pooled = {p: {"net": 0, "floor": 0.0, "queries": 0} for p in ("adjacency", "marks")}
    for r in records:
        truth = truths[(r["network"], r["seed"])]
        answer = GraphStructure.from_numpy(entries_to_marks(r["marks"], len(truth)), kind="pag")
        oracle = GraphStructure.from_numpy(truth, kind="pag")
        for t in r["scores"]:
            s = target_answer_scores(answer, oracle, int(t))
            for part, p in pooled.items():
                p["net"] += s[part]["net"]
                p["floor"] += s[part]["floor_rate"] * s[part]["queries"]
                p["queries"] += s[part]["queries"]
    for p in pooled.values():
        p["floor"] = round(p["floor"], 2)
        p["pass"] = p["net"] > p["floor"]
    return pooled


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 4) if values else None


def cell(
    runs: dict, network: str, n: int, first: range, fresh: range, truths: dict | None
) -> list[dict]:
    """The cell's rows: ``mmb_by_mmb`` against the authors' code (judged) and against each other
    solution (context)."""
    by_solution = {
        name: {s: dataset(recs) for (net, m, s), recs in fits.items() if (net, m) == (network, n)}
        for name, fits in runs.items()
    }
    mine = by_solution.get(ANDREY, {})
    rows = []
    for other, theirs in sorted(by_solution.items()):
        if other in (ANDREY, EMPTY) or not theirs:
            continue
        some = next(r for recs in runs[ANDREY].values() for r in recs if r["network"] == network)
        row: dict = {"network": network, "d": some["d_observed"], "n": n, "against": other}
        row["judged"] = other == REFERENCE
        ok_a = {s for s in first if mine.get(s)}
        ok_b = {s for s in first if theirs.get(s)}
        row["coverage"] = {
            "datasets": len({s for s in first if s in mine or s in theirs}),
            "andrey_short": len(ok_b - ok_a),
            "reference_short": len(ok_a - ok_b),
        }
        row["first"] = judge(mine, theirs, first)
        fresh_judged = judge(mine, theirs, fresh)
        row["fresh"] = fresh_judged if fresh_judged["paired"] else None
        both = sorted(ok_a & ok_b)
        row["floor"] = {
            "andrey_minus_empty": floor([mine[s] for s in both]),
            "other_minus_empty": floor([theirs[s] for s in both]),
        }
        for name in MEASURES:
            verdict = row["first"][name]["verdict"]
            if verdict == "worse":
                again = row["fresh"][name]["verdict"] if row["fresh"] else None
                verdict = {"worse": "worse", None: "worse, not rerun"}.get(again, "tie")
            row[f"{name}_result"] = verdict
        results = [row[f"{name}_result"] for name in MEASURES]
        row["result"] = next(
            (r for r in ("worse", "worse, not rerun", "better", "too few pairs") if r in results),
            "tie",
        )
        for side, fits in (
            ("andrey", [mine[s] for s in both]),
            ("other", [theirs[s] for s in both]),
        ):
            tests = [f["tests"] for f in fits if f["tests"] is not None]
            row[f"{side}_tests_per_fit"] = round(float(np.mean(tests)), 1) if tests else None
            row[f"{side}_s"] = _median([f["wall_s"] for f in fits])
            row[f"{side}_errors_mean"] = (
                round(float(np.mean([f["errors"] for f in fits])), 4) if fits else None
            )
        row["empty_errors_mean"] = (
            round(float(np.mean([mine[s]["empty_errors"] for s in both])), 4) if both else None
        )
        if truths is not None and row["judged"]:
            row["answers"] = {
                side: answer_floor(
                    [
                        r
                        for (net, m, s), recs in runs[name].items()
                        if (net, m) == (network, n) and s in both
                        for r in recs
                    ],
                    truths,
                )
                for side, name in (("andrey", ANDREY), ("reference", REFERENCE))
            }
        rows.append(row)
    return rows


def _truths(runs: dict, meta: dict) -> dict:
    """The truth of every dataset of the runs, ``{(network, seed): marks}``."""
    keys = {(net, s) for fits in runs.values() for (net, _, s) in fits}
    out = {}
    for net, seed in sorted(keys):
        draw = latent_networks.sample(
            networks.load_network(net),
            n=2,
            seed=seed,
            hidden=meta["hidden"][net],
            targets=meta["targets"][net],
        )
        out[(net, seed)] = latent_networks.truth(draw)
    return out


def summarize(runs: dict, first: range, fresh: range, truths: dict | None) -> list[dict]:
    """Every cell where ``mmb_by_mmb`` has records, in the networks' order."""
    cells = {(net, n) for (net, n, _) in runs.get(ANDREY, {})}
    order = {name: i for i, name in enumerate(latent_networks.PAPER_TARGETS)}
    rows = []
    for net, n in sorted(cells, key=lambda c: (order.get(c[0], len(order)), c[0], c[1])):
        rows.extend(cell(runs, net, n, first, fresh, truths))
    return rows


def _summary_side(judged: dict, measure: str) -> dict:
    """One measure of :func:`judge` as the release summary's error fields, F1 as its loss 1 - F1."""
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
    """The judged rows as rows of the release summary for one measure."""
    out = []
    for r in rows:
        if not r["judged"]:
            continue
        fl = r["floor"]
        applies = fl["other_minus_empty"]["bootstrap_95"][1] < 0
        row = {
            "network": r["network"],
            "d": r["d"],
            "n": r["n"],
            "seeds": r["coverage"]["datasets"],
            **_summary_side(r["first"], measure),
            "verdict": r["first"][measure]["verdict"],
            "coverage": r["coverage"],
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
            "andrey_tests_per_fit": r["andrey_tests_per_fit"],
            "comparator_tests_per_fit": r["other_tests_per_fit"],
            "result": r[f"{measure}_result"],
        }
        if r["fresh"] and r["first"][measure]["verdict"] == "worse":
            row["replication"] = {"seeds": fresh, **_summary_side(r["fresh"], measure)}
        out.append(row)
    return out


def _seeds(text: str) -> range:
    lo, hi = (int(x) for x in text.split("-"))
    return range(lo, hi + 1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MMB-by-MMB lane: mmb_by_mmb judged per cell")
    p.add_argument("runs", nargs="+", help="run folders of mmb_by_mmb_cpu.py")
    p.add_argument("--first", default="0-19", help="seeds judged first (default: 0-19)")
    p.add_argument("--fresh", default="20-39", help="seeds of the reruns (default: 20-39)")
    p.add_argument("--json", default=None, help="also write the rows to this file")
    p.add_argument(
        "--answers",
        action="store_true",
        help="add the floor of andrey.metrics.target_answer_scores (draws every dataset's truth)",
    )
    args = p.parse_args(argv)
    runs = load(args.runs)
    truths = None
    if args.answers:
        meta = json.loads((Path(args.runs[0]) / "run_meta.json").read_text())
        for run in args.runs[1:]:
            other = json.loads((Path(run) / "run_meta.json").read_text())
            meta["hidden"] |= other["hidden"]
            meta["targets"] |= other["targets"]
        truths = _truths(runs, meta)
    rows = summarize(runs, _seeds(args.first), _seeds(args.fresh), truths)
    print(
        f"{'network':9s} {'n':>5s} {'against':20s} {'pairs':>5s} {'F1 A/B':>13s} "
        f"{'dist A/B':>13s} {'result':>8s} {'floor':>6s} {'tests A/B':>17s}"
    )
    for r in rows:
        f, d = r["first"]["f1"], r["first"]["distance"]
        print(
            f"{r['network']:9s} {r['n']:5d} {r['against']:20s} {r['first']['paired']:5d} "
            f"{f['andrey_mean']!s:>6s}/{f['other_mean']!s:<6s} "
            f"{d['andrey_mean']!s:>6s}/{d['other_mean']!s:<6s} {r['result']:>8s} "
            f"{str(r['floor']['andrey_minus_empty']['pass']):>6s} "
            f"{r['andrey_tests_per_fit']!s:>8s}/{r['other_tests_per_fit']!s:<8s}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
