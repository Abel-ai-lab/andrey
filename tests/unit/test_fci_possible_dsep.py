"""FCI's ``possible_dsep``: the default keeps its recorded PAGs; ``"order_free"`` ignores labels.

``"order_free"`` builds possible-d-separation sets that do not depend on the node labels, and with
exact tests its PAGs do not either.
"""

from __future__ import annotations

import importlib
from itertools import permutations

import networkx as nx
import numpy as np
import pytest

import andrey
from andrey.constraint.fci import _possible_dsep
from andrey.core.structure import ARROW, CIRCLE, TAIL
from andrey.data import sample_scm

fci_module = importlib.import_module("andrey.constraint.fci")
remove_by_possible_dsep = fci_module._remove_by_possible_dsep


def _pag() -> np.ndarray:
    """``0 o-o 1``, ``0 <-> 3``, ``2 <-> 3``, ``4 --o 2``, and ``4 o-> 3``."""
    marks = np.zeros((5, 5), dtype=np.int8)
    for u, v, at_u, at_v in [
        (0, 1, CIRCLE, CIRCLE),
        (0, 3, ARROW, ARROW),
        (2, 3, ARROW, ARROW),
        (2, 4, CIRCLE, TAIL),
        (3, 4, ARROW, CIRCLE),
    ]:
        marks[u, v], marks[v, u] = at_u, at_v
    return marks


def test_order_free_set_ignores_node_labels():
    """The order-free set is the same under every relabeling, mapped back to the original labels.

    Node 3 is adjacent to 0. Nodes 2 and 4 are reached through the collider at 3, and each is a
    possible-parent target of the other: ``4 --o 2`` and ``2 o-- 4`` are both semidirected. The
    default set misses one of them under some labelings.
    """
    marks = _pag()
    assert set(_possible_dsep(marks, 0, 1, order_free=True)) == {2, 3, 4}
    fast_sets = set()
    for order in permutations(range(5)):
        label = np.array(order)  # node i is called label[i]
        node = np.argsort(label)
        relabeled = marks[np.ix_(node, node)]
        x, y = int(label[0]), int(label[1])
        found = _possible_dsep(relabeled, x, y, order_free=True)
        assert {int(node[v]) for v in found} == {2, 3, 4}, order
        fast_sets.add(frozenset(int(node[v]) for v in _possible_dsep(relabeled, x, y)))
    assert len(fast_sets) > 1


def _oracle_fci(monkeypatch, n_obs, edges, possible_dsep):
    """Run ``fci`` with a d-separation test on the DAG ``edges``; also count the edges the
    possible-d-separation step removes."""
    dag = nx.DiGraph(edges)
    dag.add_nodes_from(range(n_obs))

    def test(x, y, condition_set=()):
        return float(nx.is_d_separator(dag, {x}, {y}, set(condition_set)))

    removed = []

    def counted(marks, *args, **kwargs):
        before = np.count_nonzero(marks)
        remove_by_possible_dsep(marks, *args, **kwargs)
        removed.append((before - np.count_nonzero(marks)) // 2)

    monkeypatch.setattr(fci_module, "make_indep_test", lambda name, X, categorical=None: test)
    monkeypatch.setattr(fci_module, "_remove_by_possible_dsep", counted)
    pag = fci_module.fci(np.zeros((10, n_obs)), alpha=0.5, possible_dsep=possible_dsep)
    return np.asarray(pag.to_numpy(), dtype=np.int8), removed[0]


def _relabel(edges, order):
    """``edges`` with observed node ``order[j]`` renamed ``j``; hidden nodes keep their labels."""
    new = {int(old): j for j, old in enumerate(order)}
    return [(new.get(a, a), new.get(b, b)) for a, b in edges]


# Six observed nodes 0-5 with hidden common causes from node 6 up. On each, the
# possible-d-separation step removes an edge, and the default and order-free sets differ.
ORACLE_GRAPHS = {
    "one extra edge": [(6, 2), (6, 5), (7, 1), (7, 0), (4, 5), (4, 1), (5, 0), (1, 2), (3, 0)]
    + [(8, 1), (8, 3)],
    "four hidden common causes": [(6, 0), (6, 1), (7, 3), (7, 4), (2, 1), (2, 3), (1, 4), (3, 0)]
    + [(8, 3), (8, 5), (9, 4), (9, 5)],
    "a fourth parent": [(6, 0), (6, 3), (7, 1), (7, 2), (5, 3), (5, 1), (3, 2), (1, 0), (4, 0)]
    + [(4, 2), (8, 4), (8, 3)],
}


@pytest.mark.parametrize("name", sorted(ORACLE_GRAPHS))
def test_order_free_pag_ignores_node_labels_under_exact_tests(monkeypatch, name):
    """With exact tests, every labeling gives the same order-free PAG once mapped back."""
    edges = ORACLE_GRAPHS[name]
    reference, removed = _oracle_fci(monkeypatch, 6, edges, "order_free")
    assert removed > 0
    rng = np.random.default_rng(0)
    for _ in range(24):
        order = rng.permutation(6)
        pag, _ = _oracle_fci(monkeypatch, 6, _relabel(edges, order), "order_free")
        back = np.empty_like(pag)
        back[np.ix_(order, order)] = pag
        np.testing.assert_array_equal(back, reference)


# Default PAGs recorded before ``possible_dsep`` existed, on draws where ``"order_free"`` removes
# one more edge. Row i holds ``to_numpy()``'s marks at node i: 0 none, 1 tail, 2 arrowhead,
# 3 circle.
RECORDED = {
    "fci": (
        {"d": 15, "n": 750, "density": 4.0, "seed": 3},
        ["00000000201200", "00000000020200", "00000002000021", "00003000003000"]
        + ["00030030000000", "00000000000000", "00003000000000", "00200000220001"]
        + ["20000002000100", "02000002001000", "20020000020020", "22000000200000"]
        + ["00200000002000", "00200002000000"],
    ),
    "gfci": (
        {"d": 8, "n": 400, "density": 4.0, "seed": 7},
        ["0231002", "2030200", "2201222", "2020020"] + ["0230020", "0022200", "3030000"],
    ),
}


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize("method", sorted(RECORDED))
def test_default_keeps_the_recorded_pag(method):
    draw, rows = RECORDED[method]
    x = sample_scm(graph="erdos_renyi", latents=1, scale="standardize", **draw).data
    recorded = np.array([[int(mark) for mark in row] for row in rows])
    fit = getattr(andrey, method)
    np.testing.assert_array_equal(fit(x).structure.to_numpy(), recorded)
    np.testing.assert_array_equal(fit(x, possible_dsep="fast").structure.to_numpy(), recorded)
    order_free = fit(x, possible_dsep="order_free").structure.to_numpy()
    assert np.count_nonzero(order_free) < np.count_nonzero(recorded)


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize("method", ["fci", "gfci"])
@pytest.mark.parametrize("value", ["order-free", "FAST"])
def test_unknown_possible_dsep_raises(method, value):
    with pytest.raises(ValueError, match="possible_dsep must be 'fast' or 'order_free'"):
        getattr(andrey, method)(
            np.random.default_rng(0).standard_normal((50, 3)), possible_dsep=value
        )
