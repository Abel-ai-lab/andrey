#!/usr/bin/env python
"""Summarize the missing-value lanes of ``missing_cpu.py`` into paired rows.

One row per lane, cell, and method arm. A cell is a graph family, a size, a mask, and a rate; its
datasets differ by seed. Per row: every arm's mean SHD on the datasets it finished and its statuses,
the paired comparison of the method with each other arm (the reference where the lane has one,
test-wise PC, listwise deletion, mean imputation), the empty graph as the floor, the complete-data
fit of the same method, and the fit times.

A reference is another package's implementation of the same method, and its row is judged. In
``m2-bnlearn`` it is bnlearn's ``structural.em`` with the same search inside:
``bnlearn.structural_em.hc`` for ``hc`` and ``bnlearn.structural_em.tabu`` for ``tabu``, with the
complete-data control ``bnlearn.hc`` or ``bnlearn.tabu`` against Andrey's plain search, and the
``max_iter`` 10 arm of the reference as ``reference_iter10``. A method with no reference (``ges``,
``boss``, ``grasp``, ``xges``, ``exact_search``) carries ``same_data`` instead: the comparison with
another package's structural EM on the same datasets, with ``package`` naming it and ``control``
its complete-data pair (plain search against ``bnlearn.hc``). It is not a reference and is never
judged.

The SHD of every pair is ``shd``, except in ``m2-bnlearn``: bnlearn returns a DAG, whose ``shd`` is
the directed distance, and Andrey's searches return CPDAGs, whose ``shd`` is the distance between
the equivalence classes. There every arm is scored on ``class_shd``, ``mec_shd`` for a DAG arm and
``shd`` for a CPDAG arm, the distance between the CPDAGs of the estimate and the truth.

A comparison is the per-seed difference ``D = SHD(method) - SHD(other)`` on the datasets both
finished, with the 95% percentile-bootstrap interval of its mean over 10,000 resamples
(``default_rng(0)``): worse when the interval lies wholly above 0, better when wholly below, a tie
otherwise; fewer than 10 paired datasets are reported, not judged. A dataset the other arm finished
and the method did not is listed in ``failures``: an error or a timeout is never scored as a graph.
The reference comparison is also made on the skeleton F1 alone. A dataset's error is recorded with
its arm and never moves a verdict. ``--replications`` adds each row's rerun on fresh seeds: a worse
verdict against the reference counts only when the rerun agrees, and a better one is the first
run's.

    $ANDREY_BENCH_PYTHON benchmarks/missing_summary.py --runs outputs/*/ --data outputs/data \
        --out missing-data.json
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BOOTSTRAP = 10_000
MIN_PAIRS = 10
#: The level of the MVPC authors' ``demo.R``, the one their code ran at.
REFERENCE_ALPHA = 0.01
#: The mechanisms each method's claim covers; the others are reported as limits.
CLAIMED = {"mvpc": ("mcar", "mar", "mnar"), "em": ("mcar", "mar")}
#: The lanes at the papers' sizes, where the floor is judged. ``m2-bnlearn`` mixes sizes and is not
#: one of them: the floor is reported there, never applied.
PAPER_LANES = ("m1-paper", "m2-a", "m2-b", "m2-c")
#: Another package's arm for the same method: lane -> method -> arm.
REFERENCES = {
    "m1-paper": {"pc": "mvpc-r.permc"},
    "m2-a": {"notears": "missdag.miss_dag_gaussian.notears"},
    "m2-bnlearn": {"hc": "bnlearn.structural_em.hc", "tabu": "bnlearn.structural_em.tabu"},
}
REFERENCES["m2-c"] = REFERENCES["m2-a"]
#: The reference's complete-data arm, the control against the method's plain search.
COMPLETE_REFERENCE = {
    "m1-paper": {"pc": "mvpc-r.testwise"},
    "m2-a": {"notears": "missdag.notears"},
    "m2-bnlearn": {"hc": "bnlearn.hc", "tabu": "bnlearn.tabu"},
}
COMPLETE_REFERENCE["m2-c"] = COMPLETE_REFERENCE["m2-a"]
#: The reference's arm with more EM iterations, reported beside it.
MORE_ITERATIONS = {
    "bnlearn.structural_em.hc": "bnlearn.structural_em.hc.iter10",
    "bnlearn.structural_em.tabu": "bnlearn.structural_em.tabu.iter10",
}
#: For a method with no reference, another package's arm fitted on the same datasets: lane ->
#: method -> arm, and the complete-data arm of each.
SAME_DATA = {
    "m2-bnlearn": dict.fromkeys(
        ("ges", "boss", "grasp", "xges", "exact_search"), "bnlearn.structural_em.hc"
    )
}
SAME_DATA_COMPLETE = {"bnlearn.structural_em.hc": "bnlearn.hc"}
#: The lanes that score on ``class_shd``: they pair a DAG arm with CPDAG arms.
CLASS_LANES = ("m2-bnlearn",)
CELL = ["lane", "graph", "density", "d", "n", "mechanism", "form", "rate"]
OTHERS = ("reference", "reference_iter10", "same_data", "testwise", "listwise", "mean")


def bootstrap(values: np.ndarray) -> list[float] | None:
    """Percentile bootstrap 95% interval of the mean, ``default_rng(0)``."""
    if len(values) == 0:
        return None
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(0)
    means = v[rng.integers(0, len(v), size=(BOOTSTRAP, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def verdict(interval: list[float] | None, paired: int) -> str | None:
    """``worse``, ``better``, or ``tie`` of an interval of ``D``; ``None`` below the pair floor."""
    if interval is None or paired < MIN_PAIRS:
        return None
    return "worse" if interval[0] > 0 else "better" if interval[1] < 0 else "tie"


def masks(lane: str, seeds: range) -> dict[str, dict[str, Any]]:
    """The mask of every dataset of `lane` over `seeds`, by dataset id, from the driver's plan."""
    import missing_cpu

    from andrey_bench.datasets import MVPC_PAPER

    out: dict[str, dict[str, Any]] = {}
    for unit in missing_cpu.plan(lane, len(seeds), seeds.start):
        key = unit.key
        if key.functional == MVPC_PAPER:
            # The authors' MCAR reference of a draw is a mask of its own; the complete data are not.
            mode, data = key.missing, key.missing_form
            mechanism = {"quantile": mode, "reference": "mcar", "complete": "complete"}[data]
            form = "" if data == "complete" else f"{data} of {mode}"
            mask = {"graph": "randomDAG", "mechanism": mechanism, "form": form, "draw": mode}
            rate = None
        else:
            mechanism = key.missing or "complete"
            mask = {"graph": key.topology, "mechanism": mechanism, "form": key.missing_form}
            mask["draw"] = ""
            rate = key.missing_rate if key.missing else None
        out[key.digest] = mask | {"density": key.density, "rate": rate}
    return out


def load(dirs: list[str], store: str) -> pd.DataFrame:
    """The records of the part folders `dirs`, each with its dataset's mask and complete rows.

    The mask comes from the driver's plan over the seeds the records hold, and the rows listwise
    deletion keeps from the dataset files in `store`, counted once and kept in its
    ``complete_rows.json``.
    """
    frames = []
    for folder in dirs:
        path = Path(folder) / "runs.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame["lane"] = re.sub(r"-\d+$", "", Path(folder).name)
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table = table.drop_duplicates(["lane", "dataset_id", "name"], keep="last")
    known: dict[str, dict[str, Any]] = {}
    for lane, group in table.groupby("lane"):
        seeds = range(int(group["data_seed"].min()), int(group["data_seed"].max()) + 1)
        known |= masks(str(lane), seeds)
    joined = pd.DataFrame([known[i] for i in table["dataset_id"]], index=table.index)
    table = pd.concat([table.drop(columns=["density"]), joined], axis=1)
    counts = Path(store) / "complete_rows.json"
    kept = json.loads(counts.read_text()) if counts.exists() else {}
    known_rows = len(kept)
    for dataset in table.loc[table["mechanism"] != "complete", "dataset_id"].unique():
        if dataset not in kept:
            data = np.load(Path(store) / f"ds_{dataset}.npz")["data"]
            kept[dataset] = int((~np.isnan(data).any(axis=1)).sum())
    if len(kept) > known_rows:
        counts.write_text(json.dumps(kept))
    table["complete_rows"] = table["dataset_id"].map(kept)
    table["class_shd"] = class_shd(table)
    return table


def class_shd(table: pd.DataFrame) -> pd.Series:
    """The SHD between equivalence classes: a DAG arm's ``mec_shd``, a CPDAG arm's ``shd``.

    A DAG whose ``mec_shd`` is absent (a cyclic graph has no CPDAG) stays ``NaN``, never its
    directed ``shd``.
    """
    mec = table["mec_shd"] if "mec_shd" in table else np.nan
    return pd.Series(
        np.where(table["output_type"] == "dag", mec, table["shd"]), index=table.index, dtype=float
    )


def shd_column(lane: str) -> str:
    """The SHD column the pairs of `lane` are compared on."""
    return "class_shd" if lane in CLASS_LANES else "shd"


def arms_of(lane: str, name: str) -> dict[str, str] | None:
    """The arms of the row of method arm `name`, by role, or ``None`` if `name` is no method arm."""
    if (m := re.fullmatch(r"andrey\.pc\.mvpc(\.alpha[\d.]+)\.seed0", name)) is not None:
        suffix, plain = m[1], f"andrey.pc.default{m[1]}"
        arms = {"method": name, "testwise": plain, "complete": plain}
        if float(suffix.removeprefix(".alpha")) == REFERENCE_ALPHA and (
            theirs := REFERENCES.get(lane, {}).get("pc")
        ):
            arms["reference"] = theirs
    elif (m := re.fullmatch(r"andrey\.(\w+)\.em(\.ebic_gamma\d)?", name)) is not None:
        plain = f"andrey.{m[1]}.default{m[2] or ''}"
        arms = {"method": name, "testwise": "andrey.pc.default", "complete": plain}
        if theirs := REFERENCES.get(lane, {}).get(m[1]):
            arms["reference"] = theirs
            if more := MORE_ITERATIONS.get(theirs):
                arms["reference_iter10"] = more
        elif same := SAME_DATA.get(lane, {}).get(m[1]):
            arms["same_data"] = same
    else:
        return None
    return arms | {"listwise": f"listwise.{plain}", "mean": f"impute_mean.{plain}"}


def _statuses(frame: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in frame["status"].value_counts().items()}


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def compare(mine: pd.Series, other: pd.Series, finished: pd.Index) -> dict[str, Any]:
    """The paired comparison of two arms' SHD by seed; `finished` are the other arm's seeds."""
    both = mine.index.intersection(other.index)
    d = (mine[both] - other[both]).to_numpy()
    interval = bootstrap(d)
    return {
        "paired": len(both),
        "mean_D": _round(d.mean()) if len(both) else None,
        "bootstrap_95": interval,
        "verdict": verdict(interval, len(both)),
        "failures": int(len(finished.difference(mine.index))),
    }


def cell_rows(cell: dict[str, Any], frame: pd.DataFrame, complete: pd.DataFrame) -> list[dict]:
    """The rows of one cell: `frame` holds its masked records, `complete` its complete-data ones."""
    lane = cell["lane"]
    shd = shd_column(lane)
    by_arm = {name: group.set_index("data_seed") for name, group in frame.groupby("name")}
    whole = {name: group.set_index("data_seed") for name, group in complete.groupby("name")}
    empty = by_arm["baseline.empty"][shd]
    kept = frame.drop_duplicates("data_seed")["complete_rows"].dropna()
    rows = []
    for name in sorted(by_arm):
        if (arms := arms_of(lane, name)) is None:
            continue
        kind = "mvpc" if ".mvpc." in name else "em"
        method = "pc" if kind == "mvpc" else name.split(".")[1]
        ok = {}
        for role, arm in arms.items():
            records = whole.get(arm) if role == "complete" else by_arm.get(arm)
            ok[role] = None if records is None else records[records["status"] == "ok"]
        mine = ok["method"][shd]
        floor = bootstrap((mine - empty[mine.index]).to_numpy())
        claimed = cell["mechanism"] in CLAIMED[kind]
        row: dict[str, Any] = {
            "regime": "|".join(str(cell[k]) for k in CELL) + f"|{name}",
            **cell,
            "arm": name,
            "method": method,
            "missing": kind,
            "claim": "in" if claimed else "out",
            "datasets": int(frame["data_seed"].nunique()),
            "complete_rows_median": _round(kept.median()) if len(kept) else None,
            "empty_mean": _round(empty.mean()),
            "status": _statuses(by_arm[name]),
            "floor": {"andrey_minus_empty": floor, "pass": floor is not None and floor[1] < 0},
        }
        if shd != "shd":
            row["shd_column"] = shd
        for role in (
            "method",
            "reference",
            "reference_iter10",
            "same_data",
            "testwise",
            "listwise",
            "mean",
            "complete",
        ):
            if (records := ok.get(role)) is None:
                continue
            source = whole if role == "complete" else by_arm
            row[role] = {
                "arm": arms[role],
                "finished": len(records),
                "status": _statuses(source[arms[role]]),
                "mean_shd": _round(records[shd].mean()),
                "mean_skeleton_f1": _round(records["skeleton_f1"].mean()),
                "fit_s_median": _round(records["wall_s"].median(), 4),
            }
            if role == "same_data":
                row[role]["package"] = arms[role].split(".")[0]
            if role in OTHERS:
                row[role] |= compare(mine, records[shd], records.index)
                if role == "reference":
                    # The skeleton alone, which the orientation rules of the two packages do not
                    # reach: D = F1(reference) - F1(method), so above 0 is worse here too.
                    both = mine.index.intersection(records.index)
                    f1 = records["skeleton_f1"][both] - ok["method"]["skeleton_f1"][both]
                    row[role]["skeleton"] = {
                        "mean_D": _round(f1.mean(), 4) if len(both) else None,
                        "bootstrap_95": (interval := bootstrap(f1.to_numpy())),
                        "verdict": verdict(interval, len(both)),
                    }
                    interval = bootstrap((records[shd] - empty[records.index]).to_numpy())
                    row[role]["minus_empty"] = interval
                    both = mine.index.intersection(records.index)
                    ratio = ok["method"]["wall_s"][both] / records["wall_s"][both]
                    row[role]["speed_ratio_median"] = _round(ratio.median(), 4)
        # The same two packages without the method under test: both NOTEARS on the complete data
        # (M2), both test-wise PC on the masked data (M1), both searches without EM (M2 bnlearn).
        source = by_arm if lane == "m1-paper" else whole
        plain = source.get(arms["complete"])
        for role, theirs in (
            ("reference", COMPLETE_REFERENCE.get(lane, {}).get(method)),
            ("same_data", SAME_DATA_COMPLETE.get(arms.get("same_data", ""))),
        ):
            other = source.get(theirs)
            if role not in row or plain is None or other is None:
                continue
            a = plain[plain["status"] == "ok"][shd]
            b = other[other["status"] == "ok"]
            control = {"arm": arms["complete"], "other": theirs} | compare(a, b[shd], b.index)
            if role == "reference":
                row["control"] = control
            else:
                row[role]["control"] = control
        row["judged"] = claimed and "reference" in row and row["reference"]["verdict"] is not None
        if row["judged"]:
            ref = row["reference"]
            floor_applies = lane in PAPER_LANES and ref["minus_empty"][1] < 0
            row["floor"]["applies"] = floor_applies
            if floor_applies and not row["floor"]["pass"]:
                row["result"] = "fail: not better than the empty graph"
            else:
                row["result"] = ref["verdict"]
        else:
            row["result"] = "not judged"
        row["rerun"] = row["judged"] and row["result"] not in ("tie", "better")
        rows.append(row)
    return rows


def all_rows(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Every row of `table`, cell by cell."""
    masked = table[table["mechanism"] != "complete"].fillna({"rate": -1.0})
    whole = table[table["mechanism"] == "complete"]
    on = ["lane", "graph", "density", "d", "n"]
    complete = {key: group for key, group in whole.groupby(on)}
    rows = []
    for key, frame in masked.groupby(CELL, sort=True):
        cell = dict(zip(CELL, key))
        cell = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in cell.items()}
        cell["rate"] = None if cell["rate"] < 0 else cell["rate"]
        same = complete.get(tuple(cell[k] for k in on), whole.iloc[:0])
        if cell["lane"] == "m1-paper":
            # A complete draw belongs to the mode (MAR or MNAR) of the generator call it came from.
            same = same[same["draw"].isin(frame["draw"].unique())]
        rows += cell_rows(cell, frame, same)
    return rows


REPLICATION_KEYS = ("datasets", "status", "floor", "reference", "result")


def summarize(args) -> int:
    """Pair the records of ``--runs`` into rows; ``--replications`` adds each row's rerun."""
    table = load(args.runs, args.data)
    rows = all_rows(table)
    if args.replications:
        again = {r["regime"]: r for r in all_rows(load(args.replications, args.data))}
        for row in rows:
            if (rep := again.get(row["regime"])) is None:
                continue
            row["replication"] = {k: rep[k] for k in REPLICATION_KEYS if k in rep}
            if row["rerun"] and row["result"] == "worse":
                agrees = rep.get("reference", {}).get("verdict") == "worse"
                row["result"] = "worse" if agrees else "tie (did not replicate)"
                row["rerun"] = False
    judged = [r for r in rows if r["judged"]]
    results: dict[str, int] = {}
    for row in judged:
        results[row["result"]] = results.get(row["result"], 0) + 1
    summary = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lane": "missing-data",
        "parity_rule": "paired bootstrap",
        "bootstrap": {
            "resamples": BOOTSTRAP,
            "interval": "percentile 95%",
            "rng": "default_rng(0)",
        },
        "min_pairs": MIN_PAIRS,
        "reference_alpha": REFERENCE_ALPHA,
        "records": len(table),
        "fit_hours": round(float(table["wall_s"].sum()) / 3600, 1),
        "status": _statuses(table),
        "judged": len(judged),
        "results": results,
        "andrey_failures_in_judged_rows": sum(r["reference"]["failures"] for r in judged),
        "rerun": [r["regime"] for r in rows if r["rerun"]],
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=1) + "\n")
    print(f"{len(rows)} rows; status {summary['status']}; judged {results}")
    print(f"reruns: {len(summary['rerun'])}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True, help="part folders of missing_cpu.py")
    parser.add_argument("--data", required=True, help="the dataset store the parts wrote")
    parser.add_argument("--replications", nargs="+", help="part folders of the reruns")
    parser.add_argument("--out", required=True)
    return summarize(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
