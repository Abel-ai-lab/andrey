"""FCI's orientation rules: their rounds, the separating sets GFCI derives for them, and the tail
rules R7-R10 with their uncovered possibly-directed paths."""

from __future__ import annotations

import functools
import importlib

import numpy as np
import pytest

from andrey import data
from andrey.constraint.fci import (
    _circle_pag,
    _exists_uncovered_pd_path,
    _fci_orient,
    _orient_colliders,
    _rule_r7,
    _rule_r9,
    _rule_r10,
)
from andrey.core.structure import ARROW, CIRCLE, TAIL
from andrey.data.latent import marginal
from andrey.search.gfci import _derive_sepsets


def _circle_graph(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    marks = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        marks[u, v] = marks[v, u] = CIRCLE
    return marks


def _no_test(x, y, condition_set=()):
    raise AssertionError("no rule needs a test here")


def test_rules_leave_an_endpoint_two_firings_mark_differently_a_circle():
    """R1 orients ``1 -> 2`` from ``0 o-> 1`` and ``2 -> 1`` from ``3 o-> 2``."""
    marks = _circle_graph(4, [(0, 1), (1, 2), (2, 3)])
    marks[1, 0] = marks[2, 3] = ARROW
    expected = marks.copy()
    _fci_orient(marks, _no_test, 0.5, {(0, 2): (), (1, 3): ()})
    np.testing.assert_array_equal(marks, expected)
    # One firing at a time, the firing at node 1 comes first.
    expected[1, 2], expected[2, 1] = TAIL, ARROW
    marks = _circle_graph(4, [(0, 1), (1, 2), (2, 3)])
    marks[1, 0] = marks[2, 3] = ARROW
    _fci_orient(marks, _no_test, 0.5, {(0, 2): (), (1, 3): ()}, in_column_order=True)
    np.testing.assert_array_equal(marks, expected)


def _named_test(names: np.ndarray):
    """A test whose answer depends only on the queried nodes' ``names``, never on their labels."""

    def test(x, y, condition_set=()):
        pair = sorted((int(names[x]), int(names[y])))
        return 0.9 if hash((*pair, *sorted(int(names[v]) for v in condition_set))) % 3 else 0.01

    return test


def _orient(skeleton, sepsets, names, in_column_order):
    marks = _circle_pag(skeleton)
    _orient_colliders(marks, sepsets)
    _fci_orient(marks, _named_test(names), 0.5, sepsets, in_column_order=in_column_order)
    return marks


def test_rules_ignore_node_labels():
    """Relabeling the nodes relabels the PAG, with separating sets and test answers that need not
    agree with any graph; one firing at a time, some relabeling changes it."""
    rng = np.random.default_rng(0)
    in_order_changed = False
    for d in range(5, 9):
        for _ in range(30):
            skeleton = np.triu(rng.random((d, d)) < 0.45, 1)
            skeleton |= skeleton.T
            sepsets = {}
            for x, y in zip(*np.nonzero(np.triu(~skeleton, 1))):
                drawn = tuple(v for v in range(d) if v not in (x, y) and rng.random() < 0.4)
                sepsets[(int(x), int(y))] = sepsets[(int(y), int(x))] = drawn
            names = np.arange(d)
            want = _orient(skeleton, sepsets, names, False)
            want_in_order = _orient(skeleton, sepsets, names, True)
            for _ in range(4):
                order = rng.permutation(d)  # node k of the relabeled graph is node order[k]
                back = np.argsort(order)
                relabeled = {
                    (int(back[x]), int(back[y])): tuple(int(back[v]) for v in s)
                    for (x, y), s in sepsets.items()
                }
                skeleton_k = skeleton[np.ix_(order, order)]
                found = _orient(skeleton_k, relabeled, order, False)
                np.testing.assert_array_equal(found[np.ix_(back, back)], want)
                found = _orient(skeleton_k, relabeled, order, True)
                in_order_changed |= not np.array_equal(found[np.ix_(back, back)], want_in_order)
    assert in_order_changed


def test_rules_complete_the_true_pag_as_one_firing_at_a_time(monkeypatch):
    """Under exact tests no firings conflict, so the true PAGs the data module builds stay."""
    truths = [
        data.sample_scm(graph=graph, d=14, n=10, seed=seed, density=3.0, latents=2).graph
        for graph in ("erdos_renyi", "scale_free")
        for seed in range(6)
    ]
    rounds = [marginal(truth, "pag").to_numpy() for truth in truths]
    fci = importlib.import_module("andrey.constraint.fci")
    monkeypatch.setattr(fci, "_fci_orient", functools.partial(_fci_orient, in_column_order=True))
    for truth, want in zip(truths, rounds):
        np.testing.assert_array_equal(marginal(truth, "pag").to_numpy(), want)


def test_gfci_records_every_smallest_separating_set():
    """``0`` and ``1`` share the neighbors ``2`` and ``3``; each alone separates them."""
    cpdag = np.zeros((4, 4), dtype=np.int8)
    for u, v in [(0, 2), (2, 1), (0, 3), (3, 1)]:
        cpdag[u, v] = cpdag[v, u] = TAIL

    def test(x, y, condition_set=()):
        return 0.9 if len(condition_set) == 1 else 0.01

    assert _derive_sepsets(cpdag, test, 0.5)[(0, 1)] == (2, 3)
    assert _derive_sepsets(cpdag, test, 0.5, in_column_order=True)[(0, 1)] == (2,)


def test_r7_needs_a_third_node():
    """A lone ``a --o b`` has no ``c`` to pair with, so its circle stays."""
    marks = _circle_graph(2, [(0, 1)])
    marks[0, 1] = TAIL
    expected = marks.copy()

    assert _rule_r7(marks, False) is False
    np.testing.assert_array_equal(marks, expected)


def test_r7_orients_only_the_edge_to_c():
    """``a --o b o-o c`` with ``a``, ``c`` non-adjacent: only the endpoint at ``b`` toward ``c``."""
    a, b, c = range(3)
    marks = _circle_graph(3, [(a, b), (b, c)])
    marks[a, b] = TAIL
    expected = marks.copy()
    expected[b, c] = TAIL

    assert _rule_r7(marks, False)
    np.testing.assert_array_equal(marks, expected)


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
