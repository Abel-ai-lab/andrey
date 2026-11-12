#!/usr/bin/env python
"""Compare the solutions of the AGES lane per setting, from the records of ``ages_cpu.py``.

A setting is one ``(qs, qw, n)`` of the lane; its datasets are paired by seed. Three measures are
compared, each per dataset: ``shd`` (the lane's pair count, lower is better), ``correct`` (edges
directed as the true DAG directs them, higher is better), and ``wrong`` (directed edges the true
DAG does not direct that way, lower is better). Precision and recall, the paper's measures, are
``correct`` over the directed edges and over the true edges, pooled over a setting's datasets.

:func:`~andrey_bench.pairing.compare` gives the mean of the per-dataset difference with its
percentile-bootstrap 95% interval over resamples of the datasets. :func:`cell` reports, per setting:

- Andrey's AGES against pcalg's: ``identical`` when every paired dataset has the same graph;
  otherwise ``worse`` when a measure's interval lies wholly on the worse side of 0, ``better`` when
  one lies wholly on the better side and none on the worse, and ``tie`` else;
- the floor: ``shd`` of each AGES minus the empty graph's;
- AGES against GES at the same penalty, on the three measures, and the unshielded colliders of
  the AGES graph that the GES CPDAG lacks.

    $ANDREY_BENCH_PYTHON benchmarks/ages_summary.py outputs/ages-cpu [more runs] --json cells.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from andrey_bench.pairing import compare

ANDREY, REFERENCE, GES = "andrey.ages.numpy", "pcalg.ages", "andrey.ges.numpy.bic"

#: The compared measures and whether a higher value is better.
MEASURES: dict[str, bool] = {"shd": False, "correct": True, "wrong": False}

_TAIL, _ARROW = 1, 2


def load(folders: list[str]) -> list[dict]:
    """The finished fits of every run folder, each with its ``wrong`` count."""
    records = []
    for folder in folders:
        for line in (Path(folder) / "records.jsonl").read_text().splitlines():
            record = json.loads(line)
            if record["status"] == "ok" and record["repeat"] == 0:
                record["wrong"] = record["directed"] - record["correct"]
                records.append(record)
    return records


def colliders(marks: list[list[int]]) -> set[tuple[int, int, int]]:
    """The unshielded colliders ``a -> c <- b`` of a mark matrix, as ``(a, c, b)`` with ``a < b``."""
    m = np.asarray(marks)
    into = (m == _TAIL) & (m.T == _ARROW)
    found = set()
    for c in range(m.shape[0]):
        parents = np.flatnonzero(into[:, c])
        for i, a in enumerate(parents):
            for b in parents[i + 1 :]:
                if m[a, b] == 0 and m[b, a] == 0:
                    found.add((int(a), c, int(b)))
    return found


def _pooled(fits: list[dict]) -> dict:
    correct, directed, true_edges = (
        sum(r[k] for r in fits) for k in ("correct", "directed", "true_edges")
    )
    return {
        "fits": len(fits),
        "precision": round(correct / directed, 4) if directed else None,
        "recall": round(correct / true_edges, 4) if true_edges else None,
        "shd_mean": round(statistics.mean(r["shd"] for r in fits), 4),
        "shd_median": statistics.median(r["shd"] for r in fits),
        "shd_cpdag_mean": round(statistics.mean(r["shd_cpdag"] for r in fits), 4),
        "shd_dag_mean": round(statistics.mean(r["shd_dag"] for r in fits), 4),
        "directed_mean": round(statistics.mean(r["directed"] for r in fits), 4),
        "wall_s": round(statistics.median(r["wall_s"] for r in fits), 4),
        "peak_rss_mb": round(max(r["peak_rss_mb"] for r in fits), 1),
    }


def _measures(a: dict[int, dict], b: dict[int, dict]) -> dict:
    seeds = sorted(a.keys() & b.keys())
    return {
        name: compare(
            [a[s][name] for s in seeds], [b[s][name] for s in seeds], higher_is_better=higher
        )
        for name, higher in MEASURES.items()
    }


def cell(records: list[dict]) -> dict:
    """One setting's comparison, from its finished fits."""
    arm = {
        name: {r["seed"]: r for r in records if r["solution"] == name}
        for name in (ANDREY, REFERENCE, GES)
    }
    mine, ref, ges = arm[ANDREY], arm[REFERENCE], arm[GES]
    first = records[0]
    out: dict = {
        "qs": first["qs"],
        "qw": first["qw"],
        "d": first["d"],
        "n": first["n"],
        "seeds": sorted(mine),
        "solutions": {name: _pooled(list(fits.values())) for name, fits in arm.items() if fits},
        "empty_shd_mean": round(statistics.mean(r["empty_shd"] for r in mine.values()), 4),
        "empty_shd_median": statistics.median(r["empty_shd"] for r in mine.values()),
    }
    both = sorted(mine.keys() & ref.keys())
    if both:
        same = sum(mine[s]["marks"] == ref[s]["marks"] for s in both)
        measures = _measures(mine, ref)
        verdicts = {m["verdict"] for m in measures.values()}
        verdict = "worse" if "worse" in verdicts else "better" if "better" in verdicts else "tie"
        ratio = [ref[s]["wall_s"] / mine[s]["wall_s"] for s in both]
        out["reference"] = {
            "paired": len(both),
            "same_graph": same,
            "verdict": "identical" if same == len(both) else verdict,
            "measures": measures,
            "time_ratio": round(statistics.median(ratio), 2),
        }
        empty = [mine[s]["empty_shd"] for s in both]
        out["floor"] = {
            "andrey_minus_empty": compare([mine[s]["shd"] for s in both], empty),
            "reference_minus_empty": compare([ref[s]["shd"] for s in both], empty),
        }
    with_ges = sorted(mine.keys() & ges.keys())
    if with_ges:
        extra = [len(colliders(mine[s]["marks"]) - colliders(ges[s]["marks"])) for s in with_ges]
        skeleton = sum(
            (np.asarray(mine[s]["marks"]) != 0).tolist()
            == (np.asarray(ges[s]["marks"]) != 0).tolist()
            for s in with_ges
        )
        out["ges"] = {
            "paired": len(with_ges),
            "same_graph": sum(mine[s]["marks"] == ges[s]["marks"] for s in with_ges),
            "same_skeleton": skeleton,
            "measures": _measures(mine, ges),
            "new_colliders": sum(extra),
            "datasets_with_new_colliders": sum(e > 0 for e in extra),
            "time_ratio": round(
                statistics.median(mine[s]["wall_s"] / ges[s]["wall_s"] for s in with_ges), 2
            ),
        }
    return out


def summarize(records: list[dict]) -> list[dict]:
    """Every setting's :func:`cell`, by sample size from the largest, then by setting."""
    keys = sorted({(-r["n"], r["qw"] == 0, r["qs"], r["qw"]) for r in records})
    return [
        cell([r for r in records if (r["n"], r["qs"], r["qw"]) == (-n, qs, qw)])
        for n, _, qs, qw in keys
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AGES lane: the solutions compared per setting")
    p.add_argument("runs", nargs="+", help="run folders of ages_cpu.py")
    p.add_argument("--json", default=None, help="also write the settings to this file")
    args = p.parse_args(argv)
    cells = summarize(load(args.runs))
    print(
        f"{'qs':>4s} {'qw':>4s} {'n':>6s} {'same':>6s} {'verdict':>9s} {'shd':>6s} {'empty':>6s} "
        f"{'floor_hi':>8s} {'prec':>6s} {'recall':>6s} {'ges_prec':>8s} {'ges_rec':>7s} "
        f"{'correct':>8s} {'wrong':>8s}"
    )
    for c in cells:
        ref, ges, mine = c.get("reference"), c.get("ges"), c["solutions"][ANDREY]
        base = c["solutions"].get(GES, {})

        def show(value: float | None) -> str:
            return "   nan" if value is None else f"{value:6.3f}"

        print(
            f"{c['qs']:4.1f} {c['qw']:4.1f} {c['n']:6d} "
            + (
                f"{ref['same_graph']:3d}/{ref['paired']:<2d} {ref['verdict']:>9s} "
                if ref
                else " " * 17
            )
            + f"{mine['shd_mean']:6.3g} {c['empty_shd_mean']:6.3g} "
            + (f"{c['floor']['andrey_minus_empty']['bootstrap_95'][1]:8.3g} " if ref else " " * 9)
            + f"{show(mine['precision'])} {show(mine['recall'])} "
            + f"  {show(base.get('precision'))}  {show(base.get('recall'))} "
            + (
                f"{ges['measures']['correct']['verdict']:>8s} {ges['measures']['wrong']['verdict']:>8s}"
                if ges
                else ""
            )
        )
    if args.json:
        Path(args.json).write_text(json.dumps(cells, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
