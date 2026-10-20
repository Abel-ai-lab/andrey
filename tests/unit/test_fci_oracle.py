"""FCI with an exact d-separation test recovers the true PAG, whatever the node labels.

Each graph is a latent DAG whose nodes ``0 .. n_obs - 1`` are observed and the rest hidden. The
test answers every query by d-separation in the DAG, conditioning on the selection nodes too, so
FCI's output must equal the PAG of the observed nodes. The expected PAGs come from enumerating
every MAG in the true MAG's equivalence class.
"""

from __future__ import annotations

import importlib

import networkx as nx
import numpy as np
import pytest

from andrey.core.structure import ARROW, CIRCLE, TAIL

fci_module = importlib.import_module("andrey.constraint.fci")
remove_by_possible_dsep = fci_module._remove_by_possible_dsep

_, T, A, C = 0, TAIL, ARROW, CIRCLE


def _oracle_fci(monkeypatch, n_obs, edges, selection=(), removed=None, data=None, **params):
    """Run ``fci`` with a d-separation test on the DAG ``edges`` given ``selection``.

    ``removed``, when given, receives the number of edges the possible-d-separation step removes.
    ``data`` (zeros by default) only sets the order the possible-d-separation step visits the nodes
    in.
    """
    dag = nx.DiGraph(edges)
    dag.add_nodes_from(range(n_obs))
    conditioned = set(selection)

    def test(x, y, condition_set=()):
        return float(nx.is_d_separator(dag, {x}, {y}, set(condition_set) | conditioned))

    def counted(marks, *args, **kwargs):
        before = np.count_nonzero(marks)
        remove_by_possible_dsep(marks, *args, **kwargs)
        removed.append((before - np.count_nonzero(marks)) // 2)

    data = np.zeros((10, n_obs)) if data is None else data

    def make_test(name, X, categorical=None):
        """The d-separation test over ``X``, whose columns are ``data``'s in some order."""
        unused = list(range(n_obs))
        label = []
        for column in X.T:
            j = next(j for j in unused if np.array_equal(column, data[:, j]))
            unused.remove(j)
            label.append(j)
        return lambda x, y, condition_set=(): test(
            label[x], label[y], [label[v] for v in condition_set]
        )

    monkeypatch.setattr(fci_module, "make_indep_test", make_test)
    if removed is not None:
        monkeypatch.setattr(fci_module, "_remove_by_possible_dsep", counted)
    pag = fci_module.fci(data, alpha=0.5, **params)
    return np.asarray(pag.to_numpy(), dtype=np.int8)


def _relabel(n_obs, edges, order):
    """``edges`` with observed node ``order[j]`` renamed ``j``; hidden nodes keep their labels."""
    new = {int(old): j for j, old in enumerate(order)}
    return [(new.get(a, a), new.get(b, b)) for a, b in edges]


def test_r8_orients_a_circle_arrow_edge(monkeypatch):
    """R8 turns ``4 o-> 2`` into ``4 -> 2`` through ``4 -> 1 -> 2``.

    The edge's tail end has the larger label; the rule fires for either label order.
    """
    edges = [(0, 4), (0, 2), (4, 1), (4, 2), (1, 5), (1, 2), (6, 5), (3, 2), (5, 2), (7, 4), (7, 3)]
    truth = np.array(
        [
            [_, _, C, _, C, _, _],
            [_, _, T, _, A, T, _],
            [A, A, _, A, A, A, _],
            [_, _, C, _, C, _, _],
            [A, T, T, A, _, _, _],
            [_, A, T, _, _, _, A],
            [_, _, _, _, _, C, _],
        ]
    )
    np.testing.assert_array_equal(_oracle_fci(monkeypatch, 7, edges), truth)


def test_r5_finds_an_uncovered_route_behind_a_covered_one(monkeypatch):
    """Selection makes every edge undirected; R5 must find each uncovered circle cycle.

    A breadth-first search that marks a node visited on its first, covered route never tries the
    uncovered route through the same node, and leaves the circles in place.
    """
    edges = [(0, 2), (0, 3), (0, 10), (1, 5), (1, 7), (2, 9), (3, 7), (3, 8), (4, 2), (4, 8)]
    edges += [(5, 9), (5, 10), (6, 0), (6, 1)]
    truth = np.array(
        [
            [_, T, T, T, T, T],
            [T, _, _, T, _, T],
            [T, _, _, _, T, T],
            [T, T, _, _, T, _],
            [T, _, T, T, _, _],
            [T, T, T, _, _, _],
        ]
    )
    np.testing.assert_array_equal(_oracle_fci(monkeypatch, 6, edges, (7, 8, 9, 10)), truth)


def test_possible_dsep_tests_every_subset(monkeypatch):
    """Only ``{2, 3, 6}``, the first three-node subset of the possible-d-sep set, separates 4 and 5.

    Skipping that subset keeps the false edge ``4 - 5`` and leaves circles where 0 meets 4 and 5.
    """
    edges = [(2, 3), (2, 6), (3, 5), (6, 4), (7, 0), (7, 5), (9, 0), (9, 4), (10, 5), (10, 6)]
    edges += [(11, 3), (11, 4)]
    truth = np.array(
        [
            [_, _, _, _, A, A, _],
            [_, _, _, _, _, _, _],
            [_, _, _, C, _, _, C],
            [_, _, A, _, A, T, _],
            [A, _, _, A, _, _, A],
            [A, _, _, A, _, _, A],
            [_, _, A, _, T, A, _],
        ]
    )
    np.testing.assert_array_equal(_oracle_fci(monkeypatch, 7, edges), truth)


# Seven observed nodes 0-6 with hidden common causes from node 7 up; R8 fires on each.
R8_GRAPHS = {
    "one hidden common cause": [(0, 4), (0, 2), (4, 1), (4, 2), (1, 6), (1, 5), (1, 2), (6, 5)]
    + [(3, 5), (3, 2), (5, 2), (7, 4), (7, 3)],
    "two hidden common causes": [(2, 6), (2, 5), (2, 1), (2, 3), (6, 1), (4, 1), (5, 0), (5, 3)]
    + [(0, 1), (0, 3), (1, 3), (7, 3), (7, 4), (8, 5), (8, 4)],
    "three hidden common causes": [(2, 0), (2, 4), (2, 1), (0, 6), (0, 1), (5, 4), (6, 4), (6, 1)]
    + [(4, 1), (7, 5), (7, 4), (8, 3), (8, 2), (9, 2), (9, 5)],
}


@pytest.mark.parametrize("name", sorted(R8_GRAPHS))
def test_r8_graphs_do_not_depend_on_labels(monkeypatch, name):
    """On graphs where R8 fires, every labeling gives the same PAG once mapped back."""
    edges = R8_GRAPHS[name]
    reference = _oracle_fci(monkeypatch, 7, edges)
    rng = np.random.default_rng(0)
    for _ in range(12):
        order = rng.permutation(7)
        pag = _oracle_fci(monkeypatch, 7, _relabel(7, edges, order))
        back = np.empty_like(pag)
        back[np.ix_(order, order)] = pag
        np.testing.assert_array_equal(back, reference)


# Six observed nodes 0-5 with hidden common causes from node 6 up. On each, the
# possible-d-separation step removes an edge.
ORACLE_GRAPHS = {
    "one extra edge": [(6, 2), (6, 5), (7, 1), (7, 0), (4, 5), (4, 1), (5, 0), (1, 2), (3, 0)]
    + [(8, 1), (8, 3)],
    "four hidden common causes": [(6, 0), (6, 1), (7, 3), (7, 4), (2, 1), (2, 3), (1, 4), (3, 0)]
    + [(8, 3), (8, 5), (9, 4), (9, 5)],
    "a fourth parent": [(6, 0), (6, 3), (7, 1), (7, 2), (5, 3), (5, 1), (3, 2), (1, 0), (4, 0)]
    + [(4, 2), (8, 4), (8, 3)],
}


@pytest.mark.parametrize("name", sorted(ORACLE_GRAPHS))
def test_pag_ignores_node_labels_under_exact_tests(monkeypatch, name):
    """With exact tests, every labeling gives the same PAG once mapped back.

    Each node keeps its data column, so the possible-d-separation step visits the nodes in the same
    order under every labeling.
    """
    edges = ORACLE_GRAPHS[name]
    x = np.random.default_rng(1).standard_normal((10, 6))
    removed = []
    reference = _oracle_fci(monkeypatch, 6, edges, removed=removed, data=x)
    assert removed[0] > 0
    rng = np.random.default_rng(0)
    for _ in range(24):
        order = rng.permutation(6)
        pag = _oracle_fci(monkeypatch, 6, _relabel(6, edges, order), data=x[:, order])
        back = np.empty_like(pag)
        back[np.ix_(order, order)] = pag
        np.testing.assert_array_equal(back, reference)
