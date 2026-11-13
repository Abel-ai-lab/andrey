#!/usr/bin/env python
"""Set PCMB beside the published results of its paper, per cell, from the records of ``blanket_cpu.py``.

PCMB has no reference implementation, so it is judged on Table 6 of Pena, Nilsson, Bjorkegren, and
Tegner (2007): the mean and standard deviation over 10 databases of the precision, the recall, and
the distance from perfect precision and recall of the set learned at every node of the Alarm and
Pigs networks, at significance level 0.01 (:data:`PAPER`). A fit's per-target scores are the lane's
(``precision`` is ``tp / est``, ``recall`` is ``tp / true``, ``distance`` is
``hypot(1 - precision, 1 - recall)``); a dataset's value is their mean over the targets, and a
cell's value the mean over its datasets. Two readings of an empty answer, which has no precision:

- ``skip`` (the headline, the reading of the comparison with the paper's rows): the target has no
  precision or distance and counts in the recall as 0;
- ``one``: the precision is 1, so the distance is 1 minus the recall, which is 1 for an empty
  answer, as the other summaries of the set learners count it.

The floor is each dataset's mean errors (missing plus extra members, ``fp + fn``) minus the empty
set's, the true set's size: :func:`~andrey_bench.pairing.compare` gives the mean of the
differences with its percentile-bootstrap 95% interval, and a cell passes when the interval lies
wholly below 0 over at least :data:`~andrey_bench.pairing.MIN_PAIRS` datasets. The summary keeps
the per-dataset differences, the paired quantity, and every other measure as a mean.

    $ANDREY_BENCH_PYTHON benchmarks/pcmb_summary.py RUN [RUN ...] --json cells.json
    $ANDREY_BENCH_PYTHON benchmarks/pcmb_summary.py RUN [RUN ...] --summary summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from andrey_bench.pairing import MIN_PAIRS, compare

SOLUTION = "andrey.pcmb"

#: Table 6 of the paper, PCMB's rows: ``(network, rows)`` to the mean and the standard deviation
#: over 10 databases of the precision, the recall, and the distance.
PAPER: dict[tuple[str, int], dict[str, tuple[float, float]]] = {
    ("alarm", 100): {"precision": (0.79, 0.04), "recall": (0.49, 0.05), "distance": (0.51, 0.04)},
    ("alarm", 200): {"precision": (0.95, 0.03), "recall": (0.56, 0.05), "distance": (0.38, 0.06)},
    ("alarm", 500): {"precision": (0.94, 0.01), "recall": (0.72, 0.04), "distance": (0.26, 0.03)},
    ("alarm", 1000): {"precision": (0.99, 0.01), "recall": (0.79, 0.01), "distance": (0.18, 0.02)},
    ("alarm", 2000): {"precision": (1.00, 0.00), "recall": (0.83, 0.02), "distance": (0.14, 0.01)},
    ("alarm", 5000): {"precision": (1.00, 0.00), "recall": (0.86, 0.02), "distance": (0.11, 0.02)},
    ("pigs", 100): {"precision": (0.83, 0.01), "recall": (0.82, 0.02), "distance": (0.29, 0.02)},
    ("pigs", 200): {"precision": (0.97, 0.01), "recall": (0.96, 0.01), "distance": (0.07, 0.01)},
    ("pigs", 500): {"precision": (0.98, 0.00), "recall": (1.00, 0.00), "distance": (0.02, 0.00)},
}

#: The networks' variable counts, for the rows' ``d``.
VARIABLES = {"alarm": 37, "pigs": 441}


def load(folders: list[str]) -> dict[tuple[str, int], dict[int, dict]]:
    """The finished PCMB fits of the target-mode runs, by ``(network, rows)`` then seed."""
    cells: dict[tuple[str, int], dict[int, dict]] = {}
    for folder in folders:
        for line in (Path(folder) / "records.jsonl").read_text().splitlines():
            record = json.loads(line)
            if (
                record["solution"] != SOLUTION
                or record["mode"] != "target"
                or record["status"] != "ok"
            ):
                continue
            _, network, n = record["cell"].split(":")
            cells.setdefault((network, int(n)), {})[record["seed"]] = record
    return cells


def database(per_target: dict[str, dict]) -> dict[str, float]:
    """One dataset's means over its targets: the precision and distance (``skip``), the recall,
    the distance with an empty answer counted as 1 (``one``), the errors, the empty set's errors,
    and the number of empty answers."""
    precision, distance, one, recall, errors, empty = [], [], [], [], [], 0
    for score in per_target.values():
        if not score["true"]:
            raise ValueError("a target with an empty true set has no recall")
        r = score["tp"] / score["true"]
        recall.append(r)
        errors.append(score["fp"] + score["fn"])
        if score["est"]:
            p = score["tp"] / score["est"]
            precision.append(p)
            distance.append(math.hypot(1 - p, 1 - r))
            one.append(distance[-1])
        else:
            empty += 1
            one.append(1 - r)
    return {
        "precision": statistics.fmean(precision) if precision else math.nan,
        "recall": statistics.fmean(recall),
        "distance": statistics.fmean(distance) if distance else math.nan,
        "distance_one": statistics.fmean(one),
        "errors": statistics.fmean(errors),
        "empty_errors": statistics.fmean(s["true"] for s in per_target.values()),
        "empty_answers": float(empty),
    }


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def cell(network: str, n: int, fits: dict[int, dict]) -> dict:
    """One cell's row: Andrey's means and the paper's beside each other, the floor, and the time."""
    seeds = sorted(fits)
    data = [database(fits[s]["per_target"]) for s in seeds]
    column = {key: [x[key] for x in data] for key in data[0]}
    # a dataset whose targets all answer empty has no precision or distance under ``skip``
    defined = {key: [v for v in values if not math.isnan(v)] for key, values in column.items()}
    mean = {key: statistics.fmean(v) if v else math.nan for key, v in defined.items()}
    paper = PAPER[network, n]
    floor = compare(column["errors"], column["empty_errors"])
    row: dict = {
        "network": network,
        "d": VARIABLES[network],
        "n": n,
        "seeds": len(seeds),
        "andrey_distance_mean": _round(mean["distance"]),
        "andrey_distance_sd": _round(
            statistics.stdev(defined["distance"]) if len(defined["distance"]) > 1 else 0
        ),
        "paper_distance_mean": paper["distance"][0],
        "paper_distance_sd": paper["distance"][1],
        "andrey_precision_mean": _round(mean["precision"]),
        "paper_precision_mean": paper["precision"][0],
        "paper_precision_sd": paper["precision"][1],
        "andrey_recall_mean": _round(mean["recall"]),
        "paper_recall_mean": paper["recall"][0],
        "paper_recall_sd": paper["recall"][1],
        "andrey_empty_answers_mean": _round(mean["empty_answers"], 2),
        "andrey_distance_one_mean": _round(mean["distance_one"]),
        "andrey_errors_mean": _round(mean["errors"]),
        "empty_errors_mean": _round(mean["empty_errors"]),
        "floor": {
            "applies": True,
            "pass": floor["paired"] >= MIN_PAIRS and floor["bootstrap_95"][1] < 0,
            "andrey_minus_empty": {
                "paired": floor["paired"],
                "mean": floor["mean_D"],
                "bootstrap_95": floor["bootstrap_95"],
                "per_dataset": [
                    _round(a - b) for a, b in zip(column["errors"], column["empty_errors"])
                ],
            },
        },
        "andrey_s": _round(statistics.median(fits[s]["wall_s"] for s in seeds), 1),
    }
    return row


def cells(folders: list[str]) -> list[dict]:
    """Every cell the runs hold a PCMB fit for, the networks in the paper's order, by rows."""
    found = load(folders)
    unknown = sorted(set(found) - set(PAPER))
    if unknown:
        raise ValueError(f"the paper's Table 6 has no row for {unknown}")
    return [cell(*key, found[key]) for key in sorted(found, key=lambda k: (k[0], k[1]))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="run folders holding records.jsonl")
    parser.add_argument("--json", type=Path, help="write the rows here")
    args = parser.parse_args(argv)
    rows = cells(args.runs)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")
    for row in rows:
        print(
            f"{row['network']:5} {row['n']:5} k={row['seeds']:2} "
            f"distance {row['andrey_distance_mean']:.3f} ({row['andrey_distance_sd']:.3f}) "
            f"paper {row['paper_distance_mean']:.2f} ({row['paper_distance_sd']:.2f}) "
            f"precision {row['andrey_precision_mean']:.2f}/{row['paper_precision_mean']:.2f} "
            f"recall {row['andrey_recall_mean']:.2f}/{row['paper_recall_mean']:.2f} "
            f"floor {'pass' if row['floor']['pass'] else 'FAIL'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
