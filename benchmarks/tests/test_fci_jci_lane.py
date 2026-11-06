"""The FCI-JCI lane: its simulation, its truth, its Andrey adapter, its scores, and its rows.

Runs without R and Java: the reference arms and the pcalg check of the truth need them and run in
the lane's own jobs.
"""

from __future__ import annotations

import importlib
import json

import fci_jci_cpu as lane
import numpy as np
import pytest

from andrey_bench.adapters.andrey_fci_jci import AndreyFCIJCI, with_index
from andrey_bench.regime_models import jci_diagonal, merged_index_dag

NULL, TAIL, ARROW, CIRCLE = 0, 1, 2, 3


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("surgical", [False, True])
def test_the_simulation_follows_the_papers_design(seed, surgical):
    p, q = 6, 2
    d = jci_diagonal(p, q, 0.5, 0.5, 50, seed, surgical)
    A = d.adjacency
    assert d.data.shape == (150, p) and d.context.shape == (150, q)
    assert np.array_equal(d.regime, np.repeat([0, 1, 2], 50))
    assert not np.tril(A[:p, :p]).any()  # system edges i -> j with i < j
    assert not A[:, p : p + q].any()  # nothing causes a context variable
    for k in range(q):
        assert np.flatnonzero(A[p + k]).tolist() == [d.targets[k]]
        assert np.array_equal(d.context[:, k], (d.regime == k + 1).astype(float))
    hidden = A[p + q :]
    assert len(hidden) == d.hidden
    assert (hidden[:, :p].sum(axis=1) == 2).all() and not hidden[:, p:].any()
    assert len(set(d.targets.tolist())) == q


def test_the_simulation_is_the_dry_runs_draw():
    """Seed 0 at the small and larger settings: the targets and confounders the design records."""
    small, larger = jci_diagonal(4, 1, 0.5, 0.5, 500, 0), jci_diagonal(10, 1, 0.25, 0.25, 500, 0)
    assert (small.targets.tolist(), small.hidden) == ([3], 4)
    assert (larger.targets.tolist(), larger.hidden) == ([9], 9)
    again = jci_diagonal(4, 1, 0.5, 0.5, 500, 0)
    assert np.array_equal(small.data, again.data)


def test_a_perfect_intervention_cuts_the_target_from_its_causes():
    d = jci_diagonal(4, 1, 0.5, 0.5, 4000, 0, surgical=True)
    target = d.targets[0]
    rows = d.regime == 1
    parents = np.flatnonzero(d.adjacency[:, target][:4])
    assert parents.size
    corr = np.corrcoef(d.data[rows].T)
    assert np.abs(corr[target, parents]).max() < 0.1
    assert abs(d.data[rows, target].mean() - 1.0) < 0.1  # noise N(1, 1), nothing else


def test_the_merged_index_points_into_every_target():
    d = jci_diagonal(5, 3, 0.5, 0.5, 10, 2)
    merged = merged_index_dag(d.adjacency, 5, 3)
    assert merged.shape[0] == d.adjacency.shape[0] - 2
    assert set(np.flatnonzero(merged[5]).tolist()) == set(d.targets.tolist())
    assert np.array_equal(merged[6:, :5], d.adjacency[8:, :5])
    assert np.array_equal(merged[:5, :5], d.adjacency[:5, :5])


def test_the_truth_passes_its_checks_and_carries_the_jci_marks():
    data = lane.draw(lane.CELLS["S4"], 0)
    true = lane.truth(data)
    assert all(true["checks"].values())
    p = 4
    # The index's edge in the design's dry run: x3 <-o C (JCI1) and x3 <-- C (JCI123).
    assert (true["1"][3, p], true["1"][p, 3]) == (ARROW, CIRCLE)
    assert (true["123"][3, p], true["123"][p, 3]) == (ARROW, TAIL)
    assert true["plain"][p, 3] == CIRCLE  # without JCI the index end stays open
    record = lane.dataset_record(data, true, None)
    assert record["truth_ok"] and record["empty"]["1"]["shd"] == 2 * record["true_edges"]
    json.dumps(record)


def test_the_adapter_returns_the_marks_over_the_data_and_the_environment():
    data = lane.draw(lane.CELLS["S4"], 0)
    solution = AndreyFCIJCI()
    solution.setup()
    for arm in ("andrey.jci1", "andrey.jci123", "andrey.jci1.reversed"):
        x, params = lane.inputs(arm, data, 0.01)
        est = lane.estimate(arm, solution.fit(x, {**solution.params(), **params}))
        assert est.shape == (5, 5)
        solution.to_structure(est).validate()
        index = np.flatnonzero(est[4])
        assert index.size and (est[index, 4] == ARROW).all()
        if arm == "andrey.jci123":
            assert (est[4, index] == TAIL).all()
    # The design's dry run: Andrey's default PAG has endpoint SHD 9 against the JCI1 truth.
    true = lane.truth(data)
    x, params = lane.inputs("andrey.jci1", data, 0.01)
    est = solution.fit(x, {**solution.params(), **params})
    assert lane.score_arm("andrey.jci1", est, true, 4)["shd"] == 9


def test_the_index_moves_to_its_column():
    pag = np.array([[0, CIRCLE], [ARROW, 0]], dtype=np.int8)
    marks = with_index(pag, [[1, "bidirected"]], 0)
    assert marks[1, 2] == CIRCLE and marks[2, 1] == ARROW
    assert marks[2, 0] == ARROW and marks[0, 2] == ARROW


def test_the_adapter_needs_one_environment_column():
    with pytest.raises(ValueError, match="one environment column"):
        AndreyFCIJCI().fit(np.zeros((10, 3)), {"context_vars": []})


def test_the_scores_count_endpoints():
    true = np.zeros((3, 3), dtype=np.int8)
    true[0, 1], true[1, 0] = TAIL, ARROW  # 0 -> 1
    true[1, 2], true[2, 1] = ARROW, ARROW  # 1 <-> 2
    est = np.zeros((3, 3), dtype=np.int8)
    est[0, 1], est[1, 0] = CIRCLE, ARROW  # 0 o-> 1
    s = lane.scores(est, true)
    assert s["shd"] == 3  # the circle at 0, and both marks of the missing 1 <-> 2
    assert (s["adjacency_precision"], s["adjacency_recall"]) == (1.0, 0.5)
    assert (s["arrowhead_precision"], s["arrowhead_recall"]) == (1.0, 1 / 3)
    assert (s["tail_precision"], s["tail_recall"], s["tail_f1"]) == (None, 0.0, 0.0)
    assert lane.scores(np.zeros_like(true), true)["shd"] == 4


def test_the_possible_dsep_sizes_are_recorded_and_the_engine_restored():
    module = importlib.import_module("andrey.constraint.fci")
    real = module._possible_dsep
    data = lane.draw(lane.CELLS["L10"], 0)
    x, params = lane.inputs("andrey.jci1", data, 0.01)
    marks, seen = lane.refit("andrey.jci1", x, {**AndreyFCIJCI().params(), **params})
    assert module._possible_dsep is real
    assert seen["calls"] > 0 and 0 < seen["max"] <= 10
    assert marks.shape == (11, 11)


def test_every_cell_runs_the_gated_arms():
    for cell in lane.CELLS.values():
        arms = lane.cell_arms(cell)
        assert {"andrey.jci1", "andrey.jci123"} <= set(arms) <= set(lane.ARMS)
        assert cell.claim in ("in", "out")
    assert {c.name for c in lane.CELLS.values() if c.claim == "in"} == {"S4", "S4p", "L10", "X100"}


def _records(shift: float, seeds=range(20)):
    recs, empty = {}, {}
    rng = np.random.default_rng(1)
    for s in seeds:
        base = float(rng.integers(5, 10))
        empty[("S4", s)] = {v: {"shd": 14, "shd_data": 12} for v in ("1", "123")}
        for arm, value in (("andrey.jci1", base + shift), ("pcalg.jci1", base)):
            recs[("S4", s, arm, 0.01)] = {
                "cell": "S4",
                "seed": s,
                "arm": arm,
                "alpha": 0.01,
                "status": "ok",
                "shd": value,
                "marks_sha1": str(value),
                "wall_s": 1.0,
            }
    return recs, empty


@pytest.mark.parametrize(("shift", "verdict"), [(0.0, "tie"), (2.0, "worse"), (-2.0, "better")])
def test_a_row_is_judged_by_its_paired_interval(shift, verdict):
    recs, empty = _records(shift)
    row = lane.paired_row(lane.CELLS["S4"], 0.01, "andrey.jci1", "pcalg.jci1", recs, empty)
    assert row["verdict"] == verdict and row["claim"] == "in" and row["paired"] == 20
    assert row["floor"]["applies"] and row["coverage"]["andrey_short"] == 0
    assert row["same_pag"] == (20 if shift == 0 else 0)


def test_a_timeout_is_never_a_score():
    recs, empty = _records(0.0)
    recs[("S4", 3, "andrey.jci1", 0.01)] = {"status": "timeout", "seed": 3}
    row = lane.paired_row(lane.CELLS["S4"], 0.01, "andrey.jci1", "pcalg.jci1", recs, empty)
    assert row["paired"] == 19 and row["coverage"]["andrey_short"] == 1


def test_a_row_whose_reference_does_not_beat_the_empty_graph_is_not_gated():
    recs, empty = _records(0.0)
    for key in empty:
        empty[key] = {v: {"shd": 3, "shd_data": 3} for v in ("1", "123")}
    row = lane.paired_row(lane.CELLS["S4"], 0.01, "andrey.jci1", "pcalg.jci1", recs, empty)
    assert row["claim"] == "out" and row["reason"] == "reference not better than the empty graph"
