"""R5 orients a circle cycle only when the whole path, closing edge included, is uncovered."""

from __future__ import annotations

import importlib

import numpy as np

from andrey import data
from andrey.core import ARROW, CIRCLE, TAIL
from andrey.data.latent import marginal
from qa import quality_gate as qg

fci = importlib.import_module("andrey.constraint.fci")


def _circles(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    marks = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        marks[u, v] = marks[v, u] = CIRCLE
    return marks


def _undirected(marks: np.ndarray) -> int:
    n = len(marks)
    return sum(
        1 for i in range(n) for j in range(i + 1, n) if marks[i, j] == TAIL and marks[j, i] == TAIL
    )


def test_r5_orients_the_smallest_uncovered_circle_cycle():
    """A chordless four-cycle (an inner path of two nodes): all four edges become undirected."""
    marks = _circles(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    assert fci._rule_r5(marks, False)
    assert _undirected(marks) == 4


def test_r5_orients_an_uncovered_circle_cycle():
    """A chordless five-cycle of circle edges becomes five undirected edges."""
    marks = _circles(5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    assert fci._rule_r5(marks, False)
    assert _undirected(marks) == 5


def test_r5_leaves_a_cycle_whose_end_triples_are_shielded():
    """The inner path is uncovered, but chords shield the triples through the closing edge.

    Cycle 0-1-2-3-4-0 with chords 0-2 and 4-2 (the Sachs pattern: mek-raf-pka-akt-erk with chords
    mek-pka and erk-pka). With a = 0, b = 4 and inner path 1-2-3, the triples (0, 1, 2) and
    (2, 3, 4) are shielded, so the whole path is covered and R5 must not fire.
    """
    marks = _circles(5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (4, 2)])
    before = marks.copy()
    assert not fci._rule_r5(marks, False)
    np.testing.assert_array_equal(marks, before)


def test_r5_leaves_a_cycle_with_one_directed_chord():
    """One directed chord ``2 -> 4`` shields only the triple (2, 3, 4) and makes no shorter circle
    cycle. Every way around the five-cycle then has a shielded triple or a chord between the ends
    that must be non-adjacent, so R5 must not fire from either side of the chord.
    """
    marks = _circles(5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    marks[4, 2], marks[2, 4] = ARROW, TAIL
    before = marks.copy()
    assert not fci._rule_r5(marks, False)
    np.testing.assert_array_equal(marks, before)


def test_the_sachs_truth_pag_has_no_undirected_edges():
    """Without selection variables the true PAG has no tail-tail edge; Sachs's is all circles."""
    parents, children = qg.arc_indices(qg.SACHS_NODES, qg.SACHS_ARCS)
    dag = data.graphs.dag_truth(parents, children, len(qg.SACHS_NODES))
    pag = np.asarray(marginal(dag, "pag", max_nodes=len(qg.SACHS_NODES)).to_numpy())
    assert _undirected(pag) == 0
    assert set(np.unique(pag)) <= {0, CIRCLE}
