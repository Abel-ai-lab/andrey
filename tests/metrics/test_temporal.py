"""Temporal metrics: per-lag scoring, the lag-0 capability rule, and the self-loop-free summary."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.core import GraphStructure, TemporalStructure
from andrey.metrics import temporal_scores


def _lag(marks, *, self_loops: bool = False) -> GraphStructure:
    return GraphStructure.from_numpy(np.asarray(marks, dtype=np.int8), allow_self_loops=self_loops)


CONTEMP = [[0, 1], [2, 0]]  # lag-0: 0 -> 1
LAG1 = [[2, 2], [1, 0]]  # lag-1: self-loop 0->0 and 1->0 (arrowhead at 0, tail at 1)


def test_temporal_identity_is_perfect():
    t = TemporalStructure.from_lag_graphs([_lag(CONTEMP), _lag(LAG1, self_loops=True)], lags=[0, 1])
    scores = temporal_scores(t, t)
    assert scores["summary"]["f1"] == 1.0
    assert scores["contemporaneous"]["f1"] == 1.0
    assert scores["lagged"]["f1"] == 1.0
    assert scores["per_lag"][0]["shd"] == 0
    assert scores["per_lag"][1]["shd"] == 0


def test_contemporaneous_absent_without_lag0():
    # Granger-style stack: lags start at 1, so there is no contemporaneous block to score.
    t = TemporalStructure.from_lag_graphs([_lag(LAG1, self_loops=True)], lags=[1])
    scores = temporal_scores(t, t)
    assert "contemporaneous" not in scores
    assert 0 not in scores["per_lag"]


def test_lag0_not_scored_when_estimate_lacks_it():
    # Estimate has no lag-0 (a class limit); the truth's lag-0 must not be charged as misses.
    truth = TemporalStructure.from_lag_graphs(
        [_lag(CONTEMP), _lag(LAG1, self_loops=True)], lags=[0, 1]
    )
    estimate = TemporalStructure.from_lag_graphs([_lag(LAG1, self_loops=True)], lags=[1])
    scores = temporal_scores(estimate, truth)
    assert "contemporaneous" not in scores
    assert 0 not in scores["per_lag"]
    assert scores["summary"]["f1"] == 1.0  # lag-1 matches; the un-scored lag-0 does not count


def test_summary_excludes_self_loops_but_per_lag_keeps_them():
    est = TemporalStructure.from_lag_graphs([_lag([[2, 0], [0, 0]], self_loops=True)], lags=[1])
    truth = TemporalStructure.from_lag_graphs([_lag([[0, 0], [0, 2]], self_loops=True)], lags=[1])
    scores = temporal_scores(est, truth)
    assert scores["summary"]["f1"] == 1.0  # only self-loops differ, excluded from the summary
    assert scores["per_lag"][1]["shd"] == 2  # per-lag keeps the diagonal: two self-loops differ


def test_coefficient_mae_per_lag():
    g = _lag(CONTEMP)
    est = TemporalStructure.from_lag_graphs(
        [g], lags=[0], lag_weights=np.array([[[0.0, 0.8], [0.0, 0.0]]])
    )
    truth = TemporalStructure.from_lag_graphs(
        [g], lags=[0], lag_weights=np.array([[[0.0, 1.0], [0.0, 0.0]]])
    )
    scores = temporal_scores(est, truth)
    assert scores["coefficient_mae_per_lag"][0] == pytest.approx(0.2)


def test_node_count_mismatch_raises():
    small = TemporalStructure.from_lag_graphs([_lag(CONTEMP)], lags=[0])
    big = TemporalStructure.from_lag_graphs([_lag([[0, 1, 0], [2, 0, 0], [0, 0, 0]])], lags=[0])
    with pytest.raises(ValueError):
        temporal_scores(small, big)


def test_label_mismatch_raises():
    a = TemporalStructure.from_lag_graphs([_lag(CONTEMP)], lags=[0], labels=("x", "y"))
    b = TemporalStructure.from_lag_graphs([_lag(CONTEMP)], lags=[0], labels=("y", "x"))
    with pytest.raises(ValueError, match="label"):
        temporal_scores(a, b)
