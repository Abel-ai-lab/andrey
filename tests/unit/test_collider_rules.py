"""The conservative and majority collider rules of PC, CD-NOD, FCI, and GFCI.

A vote tests the two ends of an unshielded triple given every subset of either end's neighbors on
the final skeleton and decides the middle node from the share of separating subsets that hold it.
"""

from __future__ import annotations

import importlib

import networkx as nx
import numpy as np
import pytest

import andrey
from andrey import data
from andrey.core import ARROW, CIRCLE, NULL, TAIL, ci
from andrey.metrics import to_cpdag

pc_module = importlib.import_module("andrey.constraint.pc")
fci_module = importlib.import_module("andrey.constraint.fci")

VOTING = ("conservative", "majority")
RULES = ("sepsets", *VOTING)


def _skeleton(n: int, edges: list[tuple[int, int]], mark: int) -> np.ndarray:
    marks = np.full((n, n), NULL, dtype=np.int8)
    for a, b in edges:
        marks[a, b] = marks[b, a] = mark
    return marks


# The triple 0 - 1 - 2 with 0 also adjacent to 3 and 2 to 4: the pair (0, 2) is tested given
# (), (1,), (3,), (1, 3) from node 0 and (), (1,), (4,), (1, 4) from node 2. ``passing`` is the
# set of subsets that separate 0 and 2; the share is of the separating subsets holding node 1.
VOTES = {
    "share 0": ({()}, "collider", "collider"),
    "share 1": ({(1,)}, "non-collider", "non-collider"),
    "share 1/4": ({(), (3,), (1, 3)}, "ambiguous", "collider"),
    "share 2/3": ({(), (1,), (1, 3), (1, 4)}, "ambiguous", "non-collider"),
    "share 1/2": ({(), (1,)}, "ambiguous", "ambiguous"),
}


@pytest.mark.parametrize("engine", ["pc", "fci"])
@pytest.mark.parametrize("rule", VOTING)
@pytest.mark.parametrize("case", sorted(VOTES))
def test_vote_decides_the_middle_node_by_its_share(engine, rule, case):
    passing, conservative, majority = VOTES[case]
    expected = conservative if rule == "conservative" else majority
    edges = [(0, 1), (1, 2), (0, 3), (2, 4)]
    recorded = (3,)  # the skeleton's separating set; it does not vote
    sepsets = {(0, 2): recorded}

    def test(a, c, subset=()):
        return float((a, c) == (0, 2) and tuple(subset) in passing)

    if engine == "pc":
        ambiguous = pc_module.vote_colliders(_skeleton(5, edges, TAIL), test, 0.5, sepsets, rule)
    else:
        ambiguous = fci_module._vote_sepsets(
            _skeleton(5, edges, CIRCLE), test, 0.5, sepsets, rule=rule
        )
    assert ((0, 1, 2) in ambiguous) == (expected == "ambiguous")
    held = {fci_module._sepset(sepsets, 0, 2), fci_module._sepset(sepsets, 2, 0)}
    if expected == "collider":
        assert held == {frozenset({3})}
    elif expected == "non-collider":
        assert held == {frozenset({1, 3})}
    else:
        assert sepsets[0, 2] == recorded


@pytest.mark.parametrize("rule", VOTING)
def test_no_separating_subset(rule):
    """PC leaves the triple ambiguous; FCI lets the recorded separating set decide."""
    edges = [(0, 1), (1, 2)]
    sepsets = {(0, 2): (1,)}
    ambiguous = pc_module.vote_colliders(
        _skeleton(3, edges, TAIL), lambda *args: 0.0, 0.5, dict(sepsets), rule
    )
    assert ambiguous == {(0, 1, 2)}
    fci_sepsets = dict(sepsets)
    ambiguous = fci_module._vote_sepsets(
        _skeleton(3, edges, CIRCLE), lambda *args: 0.0, 0.5, fci_sepsets, rule=rule
    )
    assert not ambiguous
    assert fci_sepsets == sepsets


def _scripted(name: str, separations: dict[tuple[int, int], set[tuple[int, ...]]]):
    """Register a test, as ``name``, that separates each pair exactly given its listed sets."""

    class Scripted:
        def __init__(self, data, **kwargs):
            pass

        def __call__(self, x, y, condition_set=()):
            pair = (min(x, y), max(x, y))
            return float(tuple(sorted(condition_set)) in separations.get(pair, ()))

    return name, Scripted


# A collider 0 -> 2 <- 1 and the edge 2 - 3. The pairs (0, 3) and (1, 3) are separated given ()
# and given (2,), so node 2 is in half of their separating subsets: ambiguous under both voting
# rules. The skeleton removes those pairs given (), so the standard rule makes 3 -> 2 a collider.
BLOCKED = {(0, 1): {()}, (0, 3): {(), (2,)}, (1, 3): {(), (2,)}}


@pytest.mark.parametrize("rule", RULES)
def test_ambiguous_triples_orient_nothing(monkeypatch, rule):
    monkeypatch.setitem(ci._CI_REGISTRY, *_scripted("scripted-blocked", BLOCKED))
    x = np.random.default_rng(0).standard_normal((50, 4))
    out = andrey.pc(x, indep_test="scripted-blocked", collider_rule=rule)
    marks = out.structure.to_numpy()
    assert marks[2, 0] == marks[2, 1] == ARROW  # 0 -> 2 <- 1
    if rule == "sepsets":
        assert marks[2, 3] == ARROW  # 3 -> 2
        assert "ambiguous_triples" not in out.metadata
    else:
        # Meek's R1 would orient 2 -> 3 from 0 -> 2 or 1 -> 2 were the triples non-colliders.
        assert marks[2, 3] == marks[3, 2] == TAIL
        assert out.metadata["ambiguous_triples"] == [[0, 2, 3], [1, 2, 3]]


def _random_dag(seed: int, n: int) -> nx.DiGraph:
    rng = np.random.default_rng(seed)
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n))
    dag.add_edges_from((i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.35)
    return dag


@pytest.mark.parametrize("seed", range(4))
def test_rules_recover_the_cpdag_with_exact_tests(monkeypatch, seed):
    """With d-separation as the test, every rule gives the true CPDAG and no ambiguous triple."""
    n = 8
    dag = _random_dag(seed, n)

    class DSeparation:
        def __init__(self, data, **kwargs):
            pass

        def __call__(self, x, y, condition_set=()):
            return float(nx.is_d_separator(dag, {x}, {y}, set(condition_set)))

    monkeypatch.setitem(ci._CI_REGISTRY, "d-separation", DSeparation)
    truth = to_cpdag(nx.to_numpy_array(dag, nodelist=range(n), dtype=int)).to_numpy()
    x = np.random.default_rng(seed).standard_normal((50, n))
    for rule in RULES:
        out = andrey.pc(x, indep_test="d-separation", collider_rule=rule)
        np.testing.assert_array_equal(out.structure.to_numpy(), truth)
        assert out.metadata.get("ambiguous_triples", []) == []


def _ambiguous_back(triples: list[list[int]], order: np.ndarray) -> list[tuple[int, int, int]]:
    """The ``triples`` of a fit on ``x[:, order]``, named by the original columns."""
    back = [(int(order[a]), int(order[b]), int(order[c])) for a, b, c in triples]
    return sorted((min(a, c), b, max(a, c)) for a, b, c in back)


@pytest.mark.parametrize("method", ["pc", "fci", "gfci", "cdnod"])
@pytest.mark.parametrize("rule", VOTING)
def test_results_do_not_depend_on_the_column_order(method, rule):
    """Five column orders give the same graph and the same ambiguous triples once mapped back."""
    x = data.sample_scm(d=8, n=150, density=3.0, seed=11).data
    domain = np.repeat([0.0, 1.0], 75)
    x[:75, 0] += 1.5  # the index shifts one column, so CD-NOD keeps an edge out of it

    def fit(columns):
        args = (x[:, columns], domain) if method == "cdnod" else (x[:, columns],)
        return getattr(andrey, method)(*args, collider_rule=rule)

    reference = fit(np.arange(8))
    triples = _ambiguous_back(reference.metadata["ambiguous_triples"], np.arange(9))
    if method in ("pc", "fci"):
        assert triples  # the data leave some triple undecided, so the check has something to map
    rng = np.random.default_rng(0)
    for _ in range(5):
        order = rng.permutation(8)
        out = fit(order)
        back = np.empty((8, 8), dtype=np.int8)
        back[np.ix_(order, order)] = out.structure.to_numpy()
        np.testing.assert_array_equal(back, reference.structure.to_numpy())
        # The index stays the last node of the search, numbered 8 in either labeling.
        assert _ambiguous_back(out.metadata["ambiguous_triples"], np.append(order, 8)) == triples


@pytest.mark.parametrize("rule", VOTING)
def test_cdnod_index_is_never_an_ambiguous_middle_node(rule):
    x = data.sample_scm(d=6, n=400, density=3.0, seed=3).data
    domain = np.repeat([0.0, 1.0, 2.0, 3.0], 100)
    x[:, :3] += domain[:, None]
    out = andrey.cdnod(x, domain, collider_rule=rule)
    assert all(b != 6 for _, b, _ in out.metadata["ambiguous_triples"])


@pytest.mark.parametrize("rule", VOTING)
def test_cdnod_with_one_domain_is_pc(rule):
    x = data.sample_scm(d=8, n=150, density=3.0, seed=11).data
    out = andrey.cdnod(x, np.zeros(150), collider_rule=rule)
    reference = andrey.pc(x, collider_rule=rule)
    np.testing.assert_array_equal(out.structure.to_numpy(), reference.structure.to_numpy())
    assert out.metadata["ambiguous_triples"] == reference.metadata["ambiguous_triples"]


@pytest.mark.parametrize("method", ["pc", "cdnod"])
def test_unknown_rule_is_rejected(method):
    x = np.random.default_rng(0).standard_normal((20, 3))
    args = (x, np.repeat([0.0, 1.0], 10)) if method == "cdnod" else (x,)
    with pytest.raises(ValueError, match="collider_rule"):
        getattr(andrey, method)(*args, collider_rule="none")
