"""A target-centered method judged per cell: the paired comparisons with global PC, the replication
that a worse cell needs, the floor, and the paper's counts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import direct_summary  # noqa: E402

CELL = ("alarm", 1000, 0.02)


def _counts(extra: int = 0, missing: int = 0, under: int = 0) -> dict:
    return {
        "extra": extra,
        "missing": missing,
        "reversed": under,
        "wrongdir": 0,
        "over": 0,
        "under": under,
        "total": extra + missing + under,
    }


def _record(solution: str, seed: int, target: int, counts: dict | None, **extra) -> dict:
    alpha = None if solution == direct_summary.EMPTY else CELL[2]
    return {
        "cell": "net:alarm:1000",
        "solution": solution,
        "seed": seed,
        "target": target,
        "scope": "target" if solution == "andrey.cmb" else "graph",
        "params": {} if alpha is None else {"alpha": alpha},
        "status": "ok" if counts is not None else "timeout",
        "scores": None if counts is None else {"cpdag": counts},
        "wall_s": 1.0,
        "n_tests": 10,
    } | extra


def _run(tmp_path: Path, missing_of: dict[str, list[int]], seeds: range = range(40)) -> Path:
    """Two targets per dataset; ``missing_of[solution][seed // 20]`` edges missed at each."""
    lines = []
    for seed in seeds:
        for target in (0, 1):
            for solution, missing in missing_of.items():
                lines.append(_record(solution, seed, target, _counts(missing=missing[seed // 20])))
            lines.append(_record(direct_summary.EMPTY, seed, target, _counts(missing=3)))
    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "records.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return run


def _row(run: Path) -> dict:
    per = direct_summary.dataset_means(direct_summary.load([str(run)]))
    return direct_summary.cell_row(per, "cmb", CELL, list(range(20)), list(range(20, 40)))


def test_fewer_errors_than_pc_on_every_dataset_is_better_and_passes_the_floor(tmp_path):
    run = _run(
        tmp_path, {"andrey.cmb": [1, 1], "andrey.pc.gsq": [2, 2], "bnlearn.pc-stable.graph": [1, 1]}
    )
    row = _row(run)
    assert row["against"]["andrey.pc.gsq"]["first"]["mean_D"] == pytest.approx(-1.0)
    assert row["against"]["andrey.pc.gsq"]["verdict"] == "better"
    assert row["against"]["bnlearn.pc-stable.graph"]["verdict"] == "tie"
    assert row["floor"]["mean_D"] == pytest.approx(-2.0) and row["floor"]["pass"]
    assert row["solutions"]["andrey.cmb"]["total"] == pytest.approx(1.0)
    assert row["paper"]["total"] == 1.81


def test_a_worse_cell_counts_only_when_the_fresh_seeds_repeat_it(tmp_path):
    once = _row(
        _run(
            tmp_path / "a",
            {"andrey.cmb": [2, 1], "andrey.pc.gsq": [1, 1], "bnlearn.pc-stable.graph": [1, 1]},
        )
    )
    assert once["against"]["andrey.pc.gsq"]["first"]["verdict"] == "worse"
    assert once["against"]["andrey.pc.gsq"]["fresh"]["verdict"] == "tie"
    assert once["against"]["andrey.pc.gsq"]["verdict"] == "tie"
    twice = _row(
        _run(
            tmp_path / "b",
            {"andrey.cmb": [2, 2], "andrey.pc.gsq": [1, 1], "bnlearn.pc-stable.graph": [1, 1]},
        )
    )
    assert twice["against"]["andrey.pc.gsq"]["verdict"] == "worse"
    unrun = _row(
        _run(
            tmp_path / "c",
            {"andrey.cmb": [2, 2], "andrey.pc.gsq": [1, 1], "bnlearn.pc-stable.graph": [1, 1]},
            seeds=range(20),
        )
    )
    assert unrun["against"]["andrey.pc.gsq"]["verdict"] == "worse, not replicated"


def test_a_dataset_with_a_timeout_is_left_out_of_its_pairs_never_scored_empty(tmp_path):
    run = _run(
        tmp_path, {"andrey.cmb": [1, 1], "andrey.pc.gsq": [2, 2], "bnlearn.pc-stable.graph": [1, 1]}
    )
    records = (run / "records.jsonl").read_text()
    timeout = _record("andrey.cmb", 0, 2, None)
    (run / "records.jsonl").write_text(records + json.dumps(timeout) + "\n")
    row = _row(run)
    assert row["solutions"]["andrey.cmb"]["datasets"] == 19
    assert row["against"]["andrey.pc.gsq"]["first"]["paired"] == 19


def test_the_lenient_reading_leaves_out_the_edges_left_undirected(tmp_path):
    lines = []
    for seed in range(20):
        lines.append(_record("andrey.cmb", seed, 0, _counts(under=1)))
        lines.append(_record("andrey.pc.gsq", seed, 0, _counts(missing=1)))
        lines.append(_record(direct_summary.EMPTY, seed, 0, _counts(missing=2)))
    run = tmp_path / "run"
    run.mkdir()
    (run / "records.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    row = _row(run)
    pc = row["against"]["andrey.pc.gsq"]
    assert pc["first"]["verdict"] == "tie"
    assert pc["lenient"]["mean_D"] == pytest.approx(-1.0)
    assert row["against"]["bnlearn.pc-stable.graph"]["first"]["verdict"] == "too few pairs"
