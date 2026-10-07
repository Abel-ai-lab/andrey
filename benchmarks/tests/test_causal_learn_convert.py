"""Conversion of causal-learn outputs to the arrays the adapters return.

The CPDAG goldens pin the function the causal-learn GES, PC, BOSS, and GRaSP adapters call inside
``fit``, the remap is what FCI's PAG endpoints pass through, and the LiNGAM conversion reads the
weight matrix's direction. A convention drift fails here rather than mis-scoring a competitor.
"""

from __future__ import annotations

import numpy as np
import pytest

from andrey.core import ARROW, CIRCLE, TAIL
from andrey_bench.adapters import causal_learn_convert as convert
from andrey_bench.contracts import structure_from_adjacency

# --- GeneralGraph CPDAG -> 0/1 adjacency ---------------------------------------


def test_conversion_directed_edge():
    # causal-learn X0 -> X1:  graph[0,1] = -1 (tail), graph[1,0] = 1 (arrow).
    cl = np.array([[0, -1], [1, 0]])
    expected = np.array([[0, 1], [0, 0]], dtype=np.int8)  # directed 0 -> 1
    np.testing.assert_array_equal(convert.generalgraph_to_adjacency(cl), expected)


def test_conversion_collider():
    # X0 -> X1 <- X2:  arrows into node 1 from both 0 and 2.
    cl = np.array([[0, -1, 0], [1, 0, 1], [0, -1, 0]])
    expected = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.int8)
    np.testing.assert_array_equal(convert.generalgraph_to_adjacency(cl), expected)


def test_conversion_undirected_edge():
    # causal-learn undirected X0 - X1:  tail-tail, graph[0,1] = graph[1,0] = -1.
    cl = np.array([[0, -1], [-1, 0]])
    expected = np.array([[0, 1], [1, 0]], dtype=np.int8)  # symmetric == undirected
    np.testing.assert_array_equal(convert.generalgraph_to_adjacency(cl), expected)


def test_conversion_roundtrips_through_structure_from_adjacency():
    # The point of the convention: cl-matrix -> adjacency -> Andrey CPDAG with the right marks.
    cl = np.array([[0, -1, 0], [1, 0, 1], [0, -1, 0]])  # collider 0 -> 1 <- 2
    marks = structure_from_adjacency(convert.generalgraph_to_adjacency(cl), kind="cpdag").to_numpy()
    assert marks[1, 0] == ARROW and marks[0, 1] == TAIL  # arrowhead at 1, tail at 0  (0 -> 1)
    assert marks[1, 2] == ARROW and marks[2, 1] == TAIL  # arrowhead at 1, tail at 2  (2 -> 1)


# --- endpoint remap ------------------------------------------------------------


# One matrix exercising every mark: 0->1 directed, 2--3 undirected, 0<->4 bidirected, 1 o->4 circle.
def _signed_all_marks() -> np.ndarray:
    S = np.zeros((5, 5), dtype=int)
    S[0, 1], S[1, 0] = -1, 1  # 0 -> 1  (tail at 0, arrow at 1)
    S[2, 3], S[3, 2] = -1, -1  # 2 -- 3  (undirected)
    S[0, 4], S[4, 0] = 1, 1  # 0 <-> 4 (bidirected)
    S[1, 4], S[4, 1] = 2, 1  # 1 o-> 4 (circle at 1, arrow at 4)
    return S


def _unsigned_all_marks() -> np.ndarray:
    U = np.zeros((5, 5), dtype=np.int8)
    U[0, 1], U[1, 0] = TAIL, ARROW
    U[2, 3], U[3, 2] = TAIL, TAIL
    U[0, 4], U[4, 0] = ARROW, ARROW
    U[1, 4], U[4, 1] = CIRCLE, ARROW
    return U


def test_remap_to_unsigned_matches_table():
    out = convert.remap_to_unsigned(_signed_all_marks())
    assert out.dtype == np.int8
    assert np.array_equal(out, _unsigned_all_marks())


@pytest.mark.parametrize("bad", [3, 4, 5, -2])
def test_remap_to_unsigned_rejects_unmappable_code(bad):
    S = np.array([[0, bad], [-1, 0]])  # STAR / compound endpoints have no Andrey mark
    with pytest.raises(ValueError, match="unmappable"):
        convert.remap_to_unsigned(S)


def test_remap_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        convert.remap_to_unsigned(np.zeros((2, 3), dtype=int))


# --- LiNGAM weights ------------------------------------------------------------


def test_lingam_weights_become_a_row_source_adjacency():
    B = np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, -0.5, 0.0]])  # x1 = 0.8 x0, x2 = -0.5 x1
    assert np.array_equal(convert.lingam_to_adjacency(B), [[0, 1, 0], [0, 0, 1], [0, 0, 0]])


def test_lingam_conversion_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        convert.lingam_to_adjacency(np.zeros((2, 3)))
