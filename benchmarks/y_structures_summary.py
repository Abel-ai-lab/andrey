#!/usr/bin/env python
"""Judge Andrey's Y-structure claims against the authors' code per cell, from the records of
``y_structures_cpu.py``.

Each run folder is named by its selection rule, ``RULE=GLOB`` (:data:`~andrey_bench.selection_models.RULES`);
the rule ``authors`` is judged and any other is context. A cell is one setting, one dataset kind
(``biased``, with selection, or ``unbiased``), and one form (``yst``, ``yst_ext``): Andrey's fit and
the authors' fit on the same datasets. Per dataset:

- ``claim_error``: the false claims plus the true ancestral pairs not claimed, the wrong answers
  of :func:`andrey.metrics.answer_scores` with the claims as the answers; the empty answer's is the
  number of true pairs, so ``claim_error`` minus it is ``fp - tp``;
- ``claim_ap_loss``: 1 minus the average precision of the ranking by claim score.

:func:`~andrey_bench.pairing.compare` gives the mean of the per-dataset difference, Andrey minus the
authors' code, with its percentile-bootstrap 95% interval; a verdict needs
:data:`~andrey_bench.pairing.MIN_PAIRS` paired datasets. A cell is worse when the interval lies
wholly on the worse side of 0 on the first seeds and again on the fresh seeds (``--fresh``); a cell
worse on the first seeds only is a tie, and a cell better on the first seeds is better. The floor
of the release summary is ``claim_error`` minus the empty answer's, which a method passes when the
interval lies wholly below 0 where the authors' code does the same. The cards' floor is the
precision of the pooled claims minus the pooled base rate, the precision of claiming every pair,
whose interval over the datasets has to lie wholly above 0.

    $ANDREY_BENCH_PYTHON benchmarks/y_structures_summary.py authors=RUNS text=RUNS --json cells.json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from andrey_bench.pairing import BOOTSTRAP, MIN_PAIRS, compare

#: The rule whose cells are judged.
JUDGED_RULE = "authors"
#: The forms, by the stem of the solution names (``andrey.<stem>``, ``sbcd.<stem>``).
FORMS: dict[str, str] = {"yst": "Y-structures", "yst_ext": "Y-structures extended"}
#: The settings of the paper, in the order of its figures, with their datasets.
SETTINGS: tuple[str, ...] = ("fixed", "small", "large")
#: The release summary's regime of each dataset kind.
REGIMES: dict[str, str] = {"biased": "selection", "unbiased": "unselected"}
#: The release summary's metrics, each an error (fewer is better), by measure.
SUMMARY_METRICS: dict[str, str] = {"error": "claim_error", "ap": "claim_ap_loss"}


def load(runs: list[str]) -> dict[str, dict[tuple, dict]]:
    """Records by rule, then by ``(setting, selection, solution, seed)``; each ``RULE=GLOB``."""
    out: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for run in runs:
        rule, pattern = run.split("=", 1)
        for folder in sorted(glob.glob(pattern)):
            path = Path(folder) / "records.jsonl"
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                r = json.loads(line)
                out[rule][r["setting"], r["selection"], r["solution"], r["seed"]] = r
    return out


def measures(record: dict) -> dict[str, float]:
    """``claim_error``, ``empty_error``, ``claim_ap_loss``, and the claims' counts of one fit."""
    s = record["scores"]
    return {
        "claim_error": float(s["fp"] + s["positives"] - s["tp"]),
        "empty_error": float(s["positives"]),
        "claim_ap_loss": 1.0 - float(s["average_precision"]),
    }


def _pairs(a: dict, b: dict, seeds: range) -> list[tuple[dict, dict]]:
    """The fits of both sides on the seeds' datasets where both finished."""
    out = []
    for seed in seeds:
        ra, rb = a.get(seed), b.get(seed)
        if ra and rb and ra["status"] == "ok" and rb["status"] == "ok":
            assert ra["d"] == rb["d"] and ra["acceptance"] == rb["acceptance"], seed
            out.append((ra, rb))
    return out


def _compare(values_a: list[float], values_b: list[float]) -> dict:
    out = compare(values_a, values_b)
    if out["paired"] < MIN_PAIRS:
        out["verdict"] = "too few pairs"
    return out


def judge(pairs: list[tuple[dict, dict]]) -> dict:
    """The paired verdicts of Andrey against the authors' code on each measure."""
    out: dict = {"paired": len(pairs)}
    ma = [measures(ra) for ra, _ in pairs]
    mb = [measures(rb) for _, rb in pairs]
    for name in ("claim_error", "claim_ap_loss"):
        va, vb = [m[name] for m in ma], [m[name] for m in mb]
        c = _compare(va, vb) if pairs else {"paired": 0, "verdict": "too few pairs"}
        c["andrey_mean"] = round(float(np.mean(va)), 4) if pairs else None
        c["reference_mean"] = round(float(np.mean(vb)), 4) if pairs else None
        out[name] = c
    out["same_claims"] = sum(
        {(c[0], c[1]) for c in ra["claims"]} == {(c[0], c[1]) for c in rb["claims"]}
        for ra, rb in pairs
    )
    return out


def floor(fits: list[dict]) -> dict:
    """``claim_error`` of each fit minus the empty answer's, with its bootstrap interval."""
    m = [measures(r) for r in fits]
    c = compare([x["claim_error"] for x in m], [x["empty_error"] for x in m])
    return {
        "paired": c["paired"],
        "mean": c["mean_D"],
        "bootstrap_95": c["bootstrap_95"],
        "pass": c["paired"] >= MIN_PAIRS and c["bootstrap_95"][1] < 0,
    }


def precision_floor(fits: list[dict]) -> dict:
    """The pooled precision of the claims minus the pooled base rate, with the percentile
    bootstrap 95% interval over the datasets; ``pass`` when the interval lies above 0. A resample
    with no claim counts precision 0."""
    tp, fp, pos, pairs = (
        np.array([r["scores"][k] for r in fits], dtype=float)
        for k in ("tp", "fp", "positives", "pairs")
    )

    def margin(i: np.ndarray) -> np.ndarray:
        t, f = tp[i].sum(-1), fp[i].sum(-1)
        precision = np.divide(t, t + f, out=np.zeros_like(t), where=t + f > 0)
        return precision - pos[i].sum(-1) / pairs[i].sum(-1)

    n = len(fits)
    picks = np.random.default_rng(0).integers(0, n, (BOOTSTRAP, n))
    low, high = np.percentile(margin(picks), [2.5, 97.5])
    claimed = tp.sum() + fp.sum()
    return {
        "paired": n,
        "precision": round(float(tp.sum() / claimed), 4) if claimed else None,
        "base_rate": round(float(pos.sum() / pairs.sum()), 4),
        "mean": round(float(margin(np.arange(n))), 4),
        "bootstrap_95": [round(float(low), 4), round(float(high), 4)],
        "pass": bool(n >= MIN_PAIRS and claimed and low > 0),
    }


def _side(fits: list[dict]) -> dict:
    """One side's pooled claims, answered datasets, mean average precision, and median time."""
    tp = sum(r["scores"]["tp"] for r in fits)
    fp = sum(r["scores"]["fp"] for r in fits)
    tests = [r["n_tests"] for r in fits if r.get("n_tests") is not None]
    return {
        "tp": tp,
        "fp": fp,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "answered": sum(r["scores"]["claims"] > 0 for r in fits),
        "claim_error": round(float(np.mean([measures(r)["claim_error"] for r in fits])), 4),
        "average_precision": round(
            float(np.mean([r["scores"]["average_precision"] for r in fits])), 4
        ),
        "wall_s": round(statistics.median(r["wall_s"] for r in fits), 4),
        "tests": round(statistics.median(tests), 1) if tests else None,
    }


def _published(first: dict, fresh: dict | None) -> str:
    """The published verdict of one measure: worse only when the fresh seeds agree."""
    if first["verdict"] != "worse":
        return first["verdict"]
    if fresh is None:
        return "worse, not rerun"
    return "worse" if fresh["verdict"] == "worse" else "tie"


def cell(
    setting: str, selection: str, form: str, records: dict, first: range, fresh: range
) -> dict:
    """One cell: its verdicts on the first and the fresh seeds, both floors, and each side's
    pooled claims."""

    def side(arm: str) -> dict[int, dict]:
        return {
            k[3]: r for k, r in records.items() if k[:3] == (setting, selection, f"{arm}.{form}")
        }

    a, b = side("andrey"), side("sbcd")
    pairs = _pairs(a, b, first)
    some = next(iter(a.values()))
    ok_a = {s for s, r in a.items() if s in first and r["status"] == "ok"}
    ok_b = {s for s, r in b.items() if s in first and r["status"] == "ok"}
    row: dict = {
        "setting": setting,
        "selection": selection,
        "form": form,
        "d": some["d"],
        "n": some["n"],
        "coverage": {
            "datasets": len({s for s in a if s in first}),
            "andrey_short": len(ok_b - ok_a),
            "reference_short": len(ok_a - ok_b),
        },
        "first": judge(pairs),
    }
    fresh_pairs = _pairs(a, b, fresh)
    row["fresh"] = judge(fresh_pairs) if fresh_pairs else None
    row["floor"] = {
        "andrey_minus_empty": floor([ra for ra, _ in pairs]),
        "reference_minus_empty": floor([rb for _, rb in pairs]),
    }
    row["precision_floor"] = precision_floor([ra for ra, _ in pairs])
    row["andrey"] = _side([ra for ra, _ in pairs])
    row["reference"] = _side([rb for _, rb in pairs])
    row["base_rate"] = row["precision_floor"]["base_rate"]
    row["empty_error"] = round(float(np.mean([measures(ra)["empty_error"] for ra, _ in pairs])), 4)
    for name in ("claim_error", "claim_ap_loss"):
        fresh_m = row["fresh"][name] if row["fresh"] else None
        row[f"{name}_result"] = _published(row["first"][name], fresh_m)
    results = [row["claim_error_result"], row["claim_ap_loss_result"]]
    row["result"] = next(
        (r for r in ("worse", "worse, not rerun", "better", "too few pairs") if r in results), "tie"
    )
    return row


def summarize(records: dict, first: range, fresh: range) -> list[dict]:
    """Every cell of every rule, the judged rule's first, each with the rule it ran under."""
    rows = []
    for rule in sorted(records, key=lambda r: r != JUDGED_RULE):
        recs = records[rule]
        for setting in SETTINGS:
            for selection in REGIMES:
                for form in FORMS:
                    if any(k[:2] == (setting, selection) and k[2].endswith(form) for k in recs):
                        row = cell(setting, selection, form, recs, first, fresh)
                        rows.append({"rule": rule, "judged": rule == JUDGED_RULE, **row})
    return rows


def _summary_side(judged: dict, measure: str) -> dict:
    """One measure of :func:`judge` as the release summary's error fields, ``andrey_<metric>_mean``
    and ``comparator_<metric>_mean``."""
    metric = SUMMARY_METRICS[measure]
    m = judged["claim_error" if measure == "error" else "claim_ap_loss"]
    return {
        "paired": judged["paired"],
        f"andrey_{metric}_mean": m["andrey_mean"],
        f"comparator_{metric}_mean": m["reference_mean"],
        "mean_D": m["mean_D"],
        "bootstrap_95": m["bootstrap_95"],
        "above_zero": m["bootstrap_95"][0] > 0,
    }


def summary_rows(rows: list[dict], measure: str, fresh: str) -> list[dict]:
    """The judged cells of one form and dataset kind as rows of the release summary."""
    out = []
    name = "claim_error" if measure == "error" else "claim_ap_loss"
    for r in rows:
        fl = r["floor"]
        applies = fl["reference_minus_empty"]["bootstrap_95"][1] < 0
        row = {
            "setting": r["setting"],
            "d": r["d"],
            "n": r["n"],
            "seeds": r["coverage"]["datasets"],
            **_summary_side(r["first"], measure),
            "verdict": r["first"][name]["verdict"],
            "coverage": r["coverage"],
            "floor": {
                "applies": applies,
                "pass": fl["andrey_minus_empty"]["bootstrap_95"][1] < 0 if applies else None,
                "andrey_minus_empty": {
                    k: fl["andrey_minus_empty"][k] for k in ("paired", "mean", "bootstrap_95")
                },
                "reference_minus_empty": {
                    k: fl["reference_minus_empty"][k] for k in ("paired", "mean", "bootstrap_95")
                },
            },
            "precision_floor": r["precision_floor"],
            "andrey_claims": {k: r["andrey"][k] for k in ("tp", "fp", "precision", "answered")},
            "comparator_claims": {
                k: r["reference"][k] for k in ("tp", "fp", "precision", "answered")
            },
            "same_claims": r["first"]["same_claims"],
            "andrey_s": r["andrey"]["wall_s"],
            "comparator_s": r["reference"]["wall_s"],
            "result": r[f"{name}_result"],
        }
        if r["fresh"] and r["first"][name]["verdict"] == "worse":
            row["replication"] = {"seeds": fresh, **_summary_side(r["fresh"], measure)}
        out.append(row)
    return out


def _seeds(text: str) -> range:
    first, last = (int(x) for x in text.split("-"))
    return range(first, last + 1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Y-structures lane: the claims judged per cell")
    p.add_argument("runs", nargs="+", help="RULE=GLOB of run folders of y_structures_cpu.py")
    p.add_argument("--first", default="0-19", help="seeds judged first (default: 0-19)")
    p.add_argument("--fresh", default="20-39", help="seeds of the reruns (default: 20-39)")
    p.add_argument("--json", default=None, help="also write the cells to this file")
    p.add_argument(
        "--summary-rows",
        default=None,
        help="also write the judged cells as release-summary rows, by form, regime, and metric",
    )
    args = p.parse_args(argv)
    rows = summarize(load(args.runs), _seeds(args.first), _seeds(args.fresh))
    if args.summary_rows:
        tables: dict = defaultdict(dict)
        judged = [r for r in rows if r["judged"]]
        for form in FORMS:
            for selection, regime in REGIMES.items():
                mine = [r for r in judged if r["form"] == form and r["selection"] == selection]
                for measure, metric in SUMMARY_METRICS.items():
                    tables[form][f"{regime}|{metric}"] = summary_rows(mine, measure, args.fresh)
        Path(args.summary_rows).write_text(json.dumps(tables, indent=1))
    for r in rows:
        f, a, b = r["first"], r["andrey"], r["reference"]
        fl = r["precision_floor"]
        print(
            f"{r['rule']:<8} {r['setting']:<6} {r['selection']:<9} {r['form']:<8} n={f['paired']:>2} "
            f"andrey {a['tp']}/{a['fp']} ref {b['tp']}/{b['fp']} same {f['same_claims']} "
            f"err {f['claim_error']['verdict']:<5} ap {f['claim_ap_loss']['verdict']:<5} "
            f"-> {r['result']} precision {fl['precision']} base {fl['base_rate']} "
            f"{'pass' if fl['pass'] else 'FLOOR'}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
