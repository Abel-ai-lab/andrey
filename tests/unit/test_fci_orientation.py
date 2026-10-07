"""Uncovered possibly-directed path requirements for FCI's tail orientation rules."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from andrey.constraint.fci import _exists_uncovered_pd_path, _rule_r9, _rule_r10
from andrey.core.structure import ARROW, CIRCLE, TAIL


def _circle_graph(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    marks = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        marks[u, v] = marks[v, u] = CIRCLE
    return marks


def test_r9_prunes_unreachable_diamond_chain():
    """A failed R9 query must not enumerate the chain's exponentially many routes."""
    a, c, e, start = range(4)
    edges = [(a, c), (e, c), (a, start)]
    for i in range(40):
        x, y, end = range(4 + 3 * i, 7 + 3 * i)
        edges.extend([(start, x), (start, y), (x, end), (y, end)])
        start = end
    marks = _circle_graph(124, edges)
    marks[c, a] = marks[c, e] = ARROW
    expected = marks.copy()

    assert _rule_r9(marks, False) is False
    np.testing.assert_array_equal(marks, expected)


@pytest.mark.parametrize("covered", [False, True])
def test_r9_requires_uncovered_path_from_a(covered):
    a, b, m, c = range(4)
    marks = _circle_graph(4, [(a, b), (b, m), (m, c), (a, c)])
    marks[c, a] = ARROW
    if covered:
        marks[a, m] = marks[m, a] = CIRCLE
    expected = marks.copy()
    if not covered:
        expected[a, c] = TAIL

    assert _rule_r9(marks, False) is (not covered)
    np.testing.assert_array_equal(marks, expected)


@pytest.mark.parametrize("swap_routes", [False, True])
def test_r9_explores_different_paths_to_the_same_node(swap_routes):
    a, b, p, q, v, w, c = range(7)
    if swap_routes:
        p, q = q, p
    marks = _circle_graph(
        7, [(a, c), (a, b), (b, p), (b, q), (p, v), (q, v), (v, w), (w, c), (p, w)]
    )
    marks[c, a] = ARROW
    marks[p, w] = ARROW  # Blocks P-W as a forward step and covers P-V-W.
    expected = marks.copy()
    expected[a, c] = TAIL  # A-B-Q-V-W-C remains uncovered.

    assert _rule_r9(marks, False)
    np.testing.assert_array_equal(marks, expected)


@pytest.mark.parametrize("edge", [(0, 1), (1, 2), (2, 3)])
@pytest.mark.parametrize("blocking_end", ["near_arrow", "far_tail"])
def test_pd_path_rejects_backward_marks(edge, blocking_end):
    marks = _circle_graph(4, [(0, 1), (1, 2), (2, 3)])
    u, v = edge
    if blocking_end == "near_arrow":
        marks[u, v] = ARROW
    else:
        marks[v, u] = TAIL
    assert not _exists_uncovered_pd_path(marks, 0, 1, 3)


def test_pd_path_rejects_covered_interior_triple():
    marks = _circle_graph(5, [(0, 1), (1, 2), (2, 3), (3, 4), (1, 3)])
    marks[1, 3] = ARROW  # No shortcut 1 -> 3; the longer route is covered.
    assert not _exists_uncovered_pd_path(marks, 0, 1, 4)


def test_pd_path_cannot_revisit_a():
    marks = _circle_graph(3, [(0, 1), (0, 2)])
    assert not _exists_uncovered_pd_path(marks, 0, 1, 2)


def test_pd_path_accepts_direct_target():
    marks = _circle_graph(2, [(0, 1)])
    assert _exists_uncovered_pd_path(marks, 0, 1, 1)


def test_r10_accepts_paths_with_one_edge():
    a, c, b, d = range(4)
    marks = _circle_graph(4, [(a, c), (a, b), (a, d)])
    marks[c, a] = ARROW
    for parent in (b, d):
        marks[parent, c], marks[c, parent] = TAIL, ARROW
    expected = marks.copy()
    expected[a, c] = TAIL

    assert _rule_r10(marks, False)
    np.testing.assert_array_equal(marks, expected)


@pytest.mark.parametrize("swap_children", [False, True])
@pytest.mark.parametrize("covered_path", [None, "b", "d"])
def test_r10_requires_both_paths_uncovered_from_a(covered_path, swap_children):
    a, c, b, d, mu, nu = range(6)
    if swap_children:  # The child with the lower index leads to D.
        mu, nu = nu, mu
    marks = _circle_graph(6, [(a, c), (a, mu), (mu, b), (a, nu), (nu, d)])
    marks[c, a] = ARROW
    for parent in (b, d):
        marks[parent, c], marks[c, parent] = TAIL, ARROW
    if covered_path == "b":
        # A-MU-B is covered; B-NU prevents the shortcut A-B from pairing with A-NU.
        marks[a, b] = marks[b, a] = CIRCLE
        marks[b, nu] = marks[nu, b] = CIRCLE
    elif covered_path == "d":
        # Mirror case: only the path ending at D is covered.
        marks[a, d] = marks[d, a] = CIRCLE
        marks[d, mu] = marks[mu, d] = CIRCLE
    expected = marks.copy()
    if covered_path is None:
        expected[a, c] = TAIL

    assert _rule_r10(marks, False) is (covered_path is None)
    np.testing.assert_array_equal(marks, expected)


def test_r10_needs_two_children():
    """One child reaching both tails is not a pair."""
    a, c, b, d, mu, nu = range(6)
    marks = _circle_graph(6, [(a, c), (a, mu), (a, nu), (mu, b), (mu, d)])
    marks[c, a] = ARROW
    for parent in (b, d):
        marks[parent, c], marks[c, parent] = TAIL, ARROW
    expected = marks.copy()

    assert _rule_r10(marks, False) is False
    np.testing.assert_array_equal(marks, expected)


def test_r10_skips_path_searches_for_adjacent_children(monkeypatch):
    fci = importlib.import_module("andrey.constraint.fci")
    a, c, b, d, x, y = range(6)
    marks = _circle_graph(6, [(a, c), (a, x), (a, y), (x, y), (x, b), (y, d)])
    calls = []
    monkeypatch.setattr(fci, "_exists_uncovered_pd_path", lambda *args: calls.append(args) or True)

    assert fci._r10_connects(marks, [x, y], b, d, a) is False
    assert calls == []
