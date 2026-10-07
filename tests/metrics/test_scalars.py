"""Weight and pairwise-direction metrics."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.metrics import coefficient_mae, decision_rate, direction_accuracy

# ---- coefficient MAE ----------------------------------------------------------------------------


def test_coefficient_mae_over_true_edges_only():
    true_w = [[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]
    est_w = [[0.0, 0.9, 5.0], [0.0, 0.0, 2.2], [0.0, 0.0, 0.0]]  # the 5.0 is off a non-true edge
    # errors on true edges: |0.9-1.0| = 0.1, |2.2-2.0| = 0.2 -> mean 0.15; the 5.0 is ignored.
    assert coefficient_mae(est_w, true_w) == pytest.approx(0.15)


def test_coefficient_mae_zero_on_exact():
    w = [[0.0, 1.5], [0.0, 0.0]]
    assert coefficient_mae(w, w) == 0.0


def test_coefficient_mae_no_true_edges_is_zero():
    assert coefficient_mae([[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]) == 0.0


def test_coefficient_mae_from_structure_output():
    from andrey.core import GraphStructure, StructureOutput

    marks = np.array([[0, 1], [2, 0]], dtype=np.int8)  # 0 -> 1
    g = GraphStructure.from_numpy(marks, kind="dag")
    est = StructureOutput.new(g, weighted_adjacency=np.array([[0.0, 0.8], [0.0, 0.0]]))
    true = StructureOutput.new(g, weighted_adjacency=np.array([[0.0, 1.0], [0.0, 0.0]]))
    assert coefficient_mae(est, true) == pytest.approx(0.2)


# ---- direction accuracy -------------------------------------------------------------------------


def test_direction_accuracy_hand_case():
    assert direction_accuracy([1, 1, -1], [1, -1, -1]) == pytest.approx(2 / 3)


def test_direction_accuracy_ignores_abstentions():
    # only the two decided pairs (both correct) count.
    assert direction_accuracy([1, 0, -1], [1, 1, -1]) == 1.0


def test_direction_accuracy_weighted():
    acc = direction_accuracy([1, -1], [1, 1], weights=[3.0, 1.0])
    assert acc == pytest.approx(0.75)  # the correct pair carries weight 3 of 4


def test_direction_accuracy_no_decision_is_zero():
    assert direction_accuracy([0, 0], [1, -1]) == 0.0


def test_decision_rate():
    assert decision_rate([1, 0, -1, 0]) == 0.5
    assert decision_rate([1, 1, 1]) == 1.0
