"""RFCI with an exact d-separation test is sound, and equals FCI where Colombo et al. (2012) say.

Each graph is a latent DAG whose nodes ``0 .. n_obs - 1`` are observed and the rest hidden. The
test answers every query by d-separation in the DAG. RFCI's PAG must keep every adjacency of the
true MAG, every tail and arrowhead it sets must hold in the DAG, and an edge FCI's PAG lacks must
join two nodes neither of which causes the other. Where RFCI keeps exactly FCI's adjacencies, the
two PAGs are equal. FCI with the same test gives the true PAG.
"""

from __future__ import annotations

import importlib

import networkx as nx
import numpy as np
import pytest

from andrey.core.structure import ARROW, CIRCLE, NULL, TAIL

rfci_module = importlib.import_module("andrey.constraint.rfci")
fci_module = importlib.import_module("andrey.constraint.fci")

_, T, A, C = NULL, TAIL, ARROW, CIRCLE


def _oracle(module, monkeypatch, n_obs, edges, data=None):
    """Run ``module``'s search with a d-separation test on the DAG ``edges``.

    ``data`` (standard normal by default) only sets the columns' tie ranks; each column keeps its
    node, found by value, so a permuted ``data`` relabels the nodes the test answers for.
    """
    dag = nx.DiGraph(edges)
    dag.add_nodes_from(range(n_obs))
    data = np.random.default_rng(1).standard_normal((10, n_obs)) if data is None else data
    answers = {}

    def separated(x, y, condition_set):
        key = (min(x, y), max(x, y), frozenset(condition_set))
        if key not in answers:
            answers[key] = float(nx.is_d_separator(dag, {x}, {y}, set(condition_set)))
        return answers[key]

    def make_test(name, X, categorical=None):
        label = [next(j for j in range(n_obs) if np.array_equal(c, data[:, j])) for c in X.T]
        return lambda x, y, condition_set=(): separated(
            label[x], label[y], [label[v] for v in condition_set]
        )

    monkeypatch.setattr(module, "make_indep_test", make_test)
    search = module.rfci if module is rfci_module else module.fci
    return np.asarray(search(data, alpha=0.5).to_numpy(), dtype=np.int8)


def _rfci(monkeypatch, n_obs, edges, data=None):
    return _oracle(rfci_module, monkeypatch, n_obs, edges, data)


def _fci(monkeypatch, n_obs, edges):
    return _oracle(fci_module, monkeypatch, n_obs, edges)


def _assert_sound(n_obs, edges, rfci_pag, fci_pag):
    """``rfci_pag`` keeps FCI's adjacencies, its marks hold in the DAG, and its extra edges join
    two non-ancestors; where it keeps exactly FCI's adjacencies, it equals ``fci_pag``."""
    dag = nx.DiGraph(edges)
    dag.add_nodes_from(range(n_obs))
    ancestors = {x: nx.ancestors(dag, x) for x in range(n_obs)}
    kept, true = rfci_pag != NULL, fci_pag != NULL
    assert not (true & ~kept).any(), "an adjacency of the true PAG is missing"
    for x, y in zip(*np.nonzero(kept)):
        if rfci_pag[x, y] == ARROW:
            assert x not in ancestors[y], f"arrowhead at {x} on {x}-{y}, an ancestor of {y}"
        if rfci_pag[x, y] == TAIL:
            assert x in ancestors[y], f"tail at {x} on {x}-{y}, not an ancestor of {y}"
            assert true[x, y], f"tail at {x} on {x}-{y}, an edge FCI removes"
    if (kept == true).all():
        np.testing.assert_array_equal(rfci_pag, fci_pag)


# Colombo et al. (2012), Figure 2(a): X1..X6 are nodes 0-5, L1 and L2 nodes 6 and 7.
EXAMPLE_1 = [(2, 1), (2, 3), (2, 5), (1, 5), (3, 5), (6, 1), (6, 0), (7, 3), (7, 4), (1, 4)]
EXAMPLE_1 += [(3, 0), (4, 5)]
# Figure 3(a): X1..X5 are nodes 0-4, L1 and L2 nodes 5 and 6.
EXAMPLE_2 = [(2, 1), (2, 3), (5, 1), (5, 0), (6, 3), (6, 4), (1, 4), (3, 0)]


def _from_letters(observed: str, directed: str, bidirected: str) -> list[tuple[int, int]]:
    """A latent DAG from ``"ab cd"``-style lists: ``a -> b``, and ``c <- L -> d`` with ``L``
    hidden."""
    node = {name: i for i, name in enumerate(observed)}
    edges = [(node[a], node[b]) for a, b in directed.split()]
    for hidden, (a, b) in enumerate(bidirected.split(), start=len(observed)):
        edges += [(hidden, node[a]), (hidden, node[b])]
    return edges


# The supplement's Figure 8 (Example 3), with L1-L4 as the hidden pairs ai, id, bk, fj.
EXAMPLE_3_NODES = "abcdefgijkl"
EXAMPLE_3 = _from_letters(
    EXAMPLE_3_NODES,
    "bi fi ig ak ca cb dj ef ed kj jg gl al bl cl kl jl fl el dl",
    "ai id bk fj",
)
# The supplement's Figure 10 (Example 4), its dashed edges as hidden pairs.
EXAMPLE_4_NODES = "abcdefgijklp"
EXAMPLE_4 = _from_letters(
    EXAMPLE_4_NODES,
    "bl al aj fa fg fp gb gl gp ij jl jp jc kp bd de dl el cp",
    "ab ap bp cd cl ep fl gj jk kl",
)


def test_paper_example_1_equals_fci(monkeypatch):
    """The unshielded-triple step removes ``X1 - X5``, which FCI removes by possible-d-sep."""
    dag = nx.DiGraph(EXAMPLE_1)
    assert nx.is_d_separator(dag, {0}, {4}, {1, 2, 3})
    assert nx.is_d_separator(dag, {0}, {5}, {1, 2, 3})
    pag = _rfci(monkeypatch, 6, EXAMPLE_1)
    assert pag[0, 4] == NULL
    np.testing.assert_array_equal(pag, _fci(monkeypatch, 6, EXAMPLE_1))


def test_paper_example_2_keeps_one_extra_bidirected_edge(monkeypatch):
    """RFCI's PAG is FCI's plus ``X1 <-> X5``, which no unshielded triple or path tests."""
    pag, fci_pag = _rfci(monkeypatch, 5, EXAMPLE_2), _fci(monkeypatch, 5, EXAMPLE_2)
    assert (pag[0, 4], pag[4, 0]) == (A, A)
    pag[0, 4] = pag[4, 0] = NULL
    np.testing.assert_array_equal(pag, fci_pag)


def _separated(nodes: str, edges: list[tuple[int, int]], x: str, y: str, given: str) -> bool:
    return nx.is_d_separator(
        nx.DiGraph(edges), {nodes.index(x)}, {nodes.index(y)}, {nodes.index(v) for v in given}
    )


def test_supplement_example_3_needs_a_minimal_separating_set(monkeypatch):
    """``Xi - Xk`` is removed with a minimal separating set. The set of every other node but
    ``Xl`` also separates them, but holds ``Xj``, which would make ``Xi *-* Xj *-* Xk`` a
    non-collider, though ``Xj`` causes neither."""
    assert _separated(EXAMPLE_3_NODES, EXAMPLE_3, "l", "i", "abcdefgjk")
    pag = _rfci(monkeypatch, len(EXAMPLE_3_NODES), EXAMPLE_3)
    np.testing.assert_array_equal(pag, _fci(monkeypatch, len(EXAMPLE_3_NODES), EXAMPLE_3))


def test_supplement_example_4_tests_every_subset_on_a_path(monkeypatch):
    """Only a subset of the path's separating set removes ``Xj - Xb``; testing the whole set alone
    would orient ``Xb -> Xp``, though ``Xb`` does not cause ``Xp``."""
    assert _separated(EXAMPLE_4_NODES, EXAMPLE_4, "l", "p", "abcdefgjk")
    assert _separated(EXAMPLE_4_NODES, EXAMPLE_4, "j", "b", "afg")
    pag = _rfci(monkeypatch, len(EXAMPLE_4_NODES), EXAMPLE_4)
    np.testing.assert_array_equal(pag, _fci(monkeypatch, len(EXAMPLE_4_NODES), EXAMPLE_4))
    assert pag[EXAMPLE_4_NODES.index("b"), EXAMPLE_4_NODES.index("p")] != TAIL


# Graphs on which FCI's possible-d-separation step removes an edge (tests/unit/test_fci_oracle.py).
EXTRA_EDGE_GRAPHS = {
    "one extra edge": (6, [(6, 2), (6, 5), (7, 1), (7, 0), (4, 5), (4, 1), (5, 0), (1, 2), (3, 0)]),
    "four hidden common causes": (
        6,
        [(6, 0), (6, 1), (7, 3), (7, 4), (2, 1), (2, 3), (1, 4), (3, 0), (8, 3), (8, 5), (9, 4)]
        + [(9, 5)],
    ),
    "possible-d-sep subset": (
        7,
        [(2, 3), (2, 6), (3, 5), (6, 4), (7, 0), (7, 5), (9, 0), (9, 4), (10, 5), (10, 6)]
        + [(11, 3), (11, 4)],
    ),
}
EXTRA_EDGE_GRAPHS["one extra edge"][1].extend([(8, 1), (8, 3)])


@pytest.mark.parametrize("name", sorted(EXTRA_EDGE_GRAPHS))
def test_edges_fci_removes_by_possible_dsep_stay_and_carry_no_tail(monkeypatch, name):
    n_obs, edges = EXTRA_EDGE_GRAPHS[name]
    pag, fci_pag = _rfci(monkeypatch, n_obs, edges), _fci(monkeypatch, n_obs, edges)
    assert np.count_nonzero((pag != NULL) & (fci_pag == NULL)) == 2
    _assert_sound(n_obs, edges, pag, fci_pag)


def _random_latent_dag(seed: int) -> tuple[int, list[tuple[int, int]]]:
    """A random DAG over 6 to 9 observed nodes, and 2 to 4 hidden nodes with two children each."""
    rng = np.random.default_rng(seed)
    n_obs = 6 + seed % 4
    order = rng.permutation(n_obs)
    edges = [
        (int(order[i]), int(order[j]))
        for i in range(n_obs)
        for j in range(i + 1, n_obs)
        if rng.random() < 0.35
    ]
    for hidden in range(n_obs, n_obs + 2 + seed % 3):
        edges += [(hidden, int(child)) for child in rng.choice(n_obs, 2, replace=False)]
    return n_obs, edges


@pytest.mark.parametrize("seed", range(20))
def test_sound_on_random_latent_graphs(monkeypatch, seed):
    n_obs, edges = _random_latent_dag(seed)
    _assert_sound(n_obs, edges, _rfci(monkeypatch, n_obs, edges), _fci(monkeypatch, n_obs, edges))


@pytest.mark.parametrize(
    ("nodes", "edges"),
    [(EXAMPLE_3_NODES, EXAMPLE_3), (EXAMPLE_4_NODES, EXAMPLE_4)],
    ids=["triple removals", "path removals"],
)
def test_pag_ignores_column_order(monkeypatch, nodes, edges):
    """Five shuffles of the columns give the same PAG once mapped back, on graphs where both
    removal steps run and several separating sets tie."""
    n_obs = len(nodes)
    x = np.random.default_rng(2).standard_normal((10, n_obs))
    reference = _rfci(monkeypatch, n_obs, edges, x)
    rng = np.random.default_rng(0)
    for _ in range(5):
        order = rng.permutation(n_obs)
        new = {int(old): j for j, old in enumerate(order)}
        relabeled = [(new.get(a, a), new.get(b, b)) for a, b in edges]
        pag = _rfci(monkeypatch, n_obs, relabeled, x[:, order])
        back = np.empty_like(pag)
        back[np.ix_(order, order)] = pag
        np.testing.assert_array_equal(back, reference)
