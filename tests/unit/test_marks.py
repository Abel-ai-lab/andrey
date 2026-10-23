"""Round-trip tests for the shared endpoint-mark builders in ``andrey.api._marks``."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.api._marks import directed_marks, marks_from_signed
from andrey.core import ARROW, CIRCLE, NULL, TAIL, GraphStructure


def test_directed_single_edge_is_dag():
    B = np.zeros((3, 3))
    B[1, 0] = 0.7  # B[i, j] != 0 => edge j -> i, that is, 0 -> 1
    g = GraphStructure.from_numpy(directed_marks(B))
    assert g.kind == "dag"
    assert g.endpoints(0, 1) == (TAIL, ARROW)  # tail at cause 0, arrowhead at effect 1
    assert g.endpoints(0, 2) == (NULL, NULL)


def test_directed_both_directions_is_bidirected_pag():
    B = np.zeros((2, 2))
    B[0, 1] = 0.5  # edge 1 -> 0
    B[1, 0] = 0.5  # edge 0 -> 1  => the pair carries both directions
    g = GraphStructure.from_numpy(directed_marks(B))
    assert g.kind == "pag"
    assert g.endpoints(0, 1) == (ARROW, ARROW)


def test_directed_drops_diagonal_and_nonfinite():
    B = np.zeros((3, 3))
    B[0, 0] = 1.0  # self-loop -> dropped
    B[2, 1] = np.nan  # non-finite -> not a directed edge
    B[1, 0] = 0.4  # real edge 0 -> 1
    M = directed_marks(B)
    assert np.array_equal(np.diagonal(M), np.zeros(3))
    g = GraphStructure.from_numpy(M)  # would raise if diagonal or asymmetric support leaked
    assert g.endpoints(1, 2) == (NULL, NULL)
    assert g.endpoints(0, 1) == (TAIL, ARROW)


def test_marks_reject_bad_input():
    with pytest.raises(ValueError):
        directed_marks(np.zeros((2, 3)))  # non-square


def test_marks_from_signed_directed_plus_nan_pair():
    """Directed support + a per-algorithm mark on every NaN pair, in one call."""
    B = np.array([[0.0, 0.5, np.nan], [0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    # B[0, 1] = 0.5 => edge 1 -> 0 (arrow at 0); the 0-2 pair is NaN both ways (confounded).
    bidirected = marks_from_signed(B, nan_mark="bidirected")
    assert (bidirected[0, 1], bidirected[1, 0]) == (ARROW, TAIL)  # directed 1 -> 0 preserved
    assert (bidirected[0, 2], bidirected[2, 0]) == (ARROW, ARROW)  # NaN pair -> bidirected (RCD)
    circle = marks_from_signed(B, nan_mark="circle")
    assert (circle[0, 2], circle[2, 0]) == (CIRCLE, CIRCLE)  # NaN pair -> o-o (BottomUp)
    with pytest.raises(ValueError, match="nan_mark"):
        marks_from_signed(B, nan_mark="bogus")
