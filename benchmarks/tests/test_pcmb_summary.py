"""PCMB beside the published results of its paper: the readings of an empty answer, the floor, the
cell's row, and the paper's table.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pcmb_summary  # noqa: E402


def _score(est: int, true: int, tp: int) -> dict:
    return {"est": est, "true": true, "tp": tp, "fp": est - tp, "fn": true - tp}


def _record(seed: int, scores: list[dict], cell: str = "net:alarm:500", **extra) -> dict:
    return {
        "solution": pcmb_summary.SOLUTION,
        "mode": "target",
        "status": "ok",
        "cell": cell,
        "seed": seed,
        "wall_s": 2.0 + seed,
        "per_target": {str(i): s for i, s in enumerate(scores)},
        **extra,
    }


def test_an_empty_answer_has_no_precision_under_skip_and_distance_one_under_one():
    """A target whose answer is empty counts in the recall as 0 and leaves the precision and the
    distance of ``skip`` to the other targets; ``one`` gives it the distance 1."""
    got = pcmb_summary.database(
        {
            "a": _score(est=2, true=2, tp=2),  # exact: distance 0
            "b": _score(est=2, true=4, tp=1),  # precision 1/2, recall 1/4
            "c": _score(est=0, true=4, tp=0),  # empty
        }
    )
    d_b = math.hypot(0.5, 0.75)
    assert got["precision"] == pytest.approx((1 + 0.5) / 2)
    assert got["recall"] == pytest.approx((1 + 0.25 + 0) / 3)
    assert got["distance"] == pytest.approx(d_b / 2)
    assert got["distance_one"] == pytest.approx((0 + d_b + 1) / 3)
    assert got["empty_answers"] == 1
    assert got["errors"] == pytest.approx((0 + 4 + 4) / 3)  # missing plus extra
    assert got["empty_errors"] == pytest.approx((2 + 4 + 4) / 3)  # the true set's size


def test_a_target_with_an_empty_true_set_is_refused():
    with pytest.raises(ValueError, match="empty true set"):
        pcmb_summary.database({"a": _score(est=1, true=0, tp=0)})


def test_a_cell_sets_the_means_beside_the_papers_and_keeps_the_paired_floor_per_dataset():
    exact = [_score(2, 2, 2), _score(3, 3, 3)]  # no errors against an empty set's 2.5
    half = [_score(2, 4, 2), _score(1, 2, 1)]  # 1.5 errors against an empty set's 3
    fits = {s: _record(s, exact if s % 2 else half) for s in range(10)}
    row = pcmb_summary.cell("alarm", 500, fits)
    paper = pcmb_summary.PAPER["alarm", 500]
    assert (row["network"], row["d"], row["n"], row["seeds"]) == ("alarm", 37, 500, 10)
    assert (row["paper_distance_mean"], row["paper_distance_sd"]) == paper["distance"]
    assert (row["paper_precision_mean"], row["paper_recall_sd"]) == (
        paper["precision"][0],
        paper["recall"][1],
    )
    assert row["andrey_precision_mean"] == pytest.approx(1.0)
    assert row["andrey_recall_mean"] == pytest.approx((1.0 + 0.5) / 2)
    assert row["andrey_empty_answers_mean"] == 0
    assert row["andrey_s"] == pytest.approx(6.5)  # the median of 2 to 11
    floor = row["floor"]
    assert floor["andrey_minus_empty"]["paired"] == 10
    assert floor["andrey_minus_empty"]["per_dataset"] == [-1.5, -2.5] * 5  # by seed
    assert floor["pass"] is True
    assert floor["andrey_minus_empty"]["bootstrap_95"][1] < 0


def test_a_cell_whose_errors_equal_the_empty_sets_fails_the_floor():
    only_empty = [_score(0, 2, 0), _score(0, 3, 0)]
    row = pcmb_summary.cell("alarm", 500, {s: _record(s, only_empty) for s in range(10)})
    assert row["floor"]["pass"] is False
    assert row["floor"]["andrey_minus_empty"]["mean"] == 0
    assert row["andrey_empty_answers_mean"] == 2
    assert row["andrey_distance_one_mean"] == 1


def test_a_cell_with_fewer_datasets_than_a_verdict_needs_does_not_pass_the_floor():
    exact = [_score(2, 2, 2), _score(3, 3, 3)]
    row = pcmb_summary.cell("pigs", 100, {s: _record(s, exact, "net:pigs:100") for s in range(4)})
    assert row["seeds"] == 4
    assert row["floor"]["pass"] is False


def test_the_runs_are_read_by_cell_and_seed_and_only_finished_pcmb_target_fits_count(tmp_path):
    scores = [_score(2, 2, 2), _score(3, 3, 3)]
    other = [
        _record(0, scores, "net:alarm:100", solution="andrey.iamb"),
        _record(1, scores, "net:alarm:100", status="error"),
        _record(2, scores, "net:alarm:100", mode="graph"),
    ]
    for name, seeds in (("a", [0, 1]), ("b", [2, 3])):
        run = tmp_path / name
        run.mkdir()
        rows = [_record(s, scores, "net:alarm:100") for s in seeds] + other
        (run / "records.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    found = pcmb_summary.load([str(tmp_path / "a"), str(tmp_path / "b")])
    assert sorted(found) == [("alarm", 100)]
    assert sorted(found["alarm", 100]) == [0, 1, 2, 3]


def test_a_cell_the_paper_has_no_row_for_is_refused(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    record = _record(0, [_score(1, 1, 1)], "net:alarm:300")
    (run / "records.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="Table 6"):
        pcmb_summary.cells([str(run)])


def test_the_papers_rows_cover_the_cells_of_both_networks_in_the_papers_sizes():
    assert sorted(pcmb_summary.PAPER) == sorted(
        [("alarm", n) for n in (100, 200, 500, 1000, 2000, 5000)]
        + [("pigs", n) for n in (100, 200, 500)]
    )
    for values in pcmb_summary.PAPER.values():
        assert set(values) == {"precision", "recall", "distance"}
        assert all(0 <= mean <= 1 and sd >= 0 for mean, sd in values.values())
