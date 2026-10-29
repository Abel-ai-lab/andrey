"""``max_parents`` caps the parents any DAG of the returned CPDAG's class gives a node.

A node's most parents over the class is its directed parents plus the largest set of its undirected
neighbors that are adjacent to one another. The cap holds in the returned CPDAG and so in every DAG
of its class; a move whose rebuilt CPDAG would break it is never taken, in either phase; and the
checks that avoid scoring such a move change nothing the post-move test would have allowed.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import andrey
from andrey import Knowledge
from andrey.core import ARROW, TAIL
from andrey.core.orient import dag2cpdag, pdag2dag
from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore
from andrey.search import _parallel_ges as pg
from andrey.search import operators as ops
from andrey.search.ges import ges
from andrey.search.hc import tie_ranks


def _sem(seed: int, n: int, d: int, p: float) -> np.ndarray:
    """Linear-Gaussian data from a random DAG on ``d`` nodes with edge density ``p``."""
    rng = np.random.default_rng(seed)
    edges = np.triu(rng.random((d, d)) < p, 1)
    b = edges * rng.uniform(0.5, 1.2, (d, d)) * rng.choice([-1, 1], (d, d))
    x = rng.standard_normal((n, d))
    for j in range(d):
        x[:, j] += x @ b[:, j]
    return x


def _in_degrees(dag: np.ndarray) -> np.ndarray:
    """Each node's parent count in a DAG of marks."""
    return (dag == ARROW).sum(axis=1)


def _undirected(n: int, lines: list[tuple[int, int]]) -> np.ndarray:
    """A CPDAG of marks with only the undirected edges ``lines``."""
    adj = np.zeros((n, n), dtype=np.int8)
    for i, j in lines:
        adj[i, j] = adj[j, i] = TAIL
    return adj


def _dags_of_class(cpdag: np.ndarray):
    """Every DAG of the class of ``cpdag``: orientations of its undirected edges that add no cycle
    and leave the same CPDAG."""
    n = cpdag.shape[0]
    base = np.where((cpdag == TAIL) & (cpdag.T == ARROW), cpdag, 0).astype(np.int8)
    base = np.where((cpdag == ARROW) & (cpdag.T == TAIL), cpdag, base)
    lines = [
        (i, j) for i in range(n) for j in range(i + 1, n) if cpdag[i, j] == cpdag[j, i] == TAIL
    ]
    for turns in itertools.product((0, 1), repeat=len(lines)):
        dag = base.copy()
        for (i, j), turn in zip(lines, turns, strict=True):
            tail, head = (j, i) if turn else (i, j)
            dag[tail, head], dag[head, tail] = TAIL, ARROW
        try:
            same_class = np.array_equal(dag2cpdag(dag), cpdag)
        except ValueError:  # a directed cycle
            continue
        if same_class:
            yield dag


def _most_parents_by_listing(cpdag: np.ndarray) -> np.ndarray:
    """Each node's most parents over the DAGs of the class, found by listing them."""
    dags = list(_dags_of_class(cpdag))
    assert dags, "the class holds no DAG"
    return np.max([_in_degrees(dag) for dag in dags], axis=0)


def _random_cpdag(rng: np.random.Generator, n: int, p: float) -> np.ndarray:
    """The CPDAG of a random DAG on ``n`` nodes with edge density ``p``."""
    order = rng.permutation(n)
    dag = np.zeros((n, n), dtype=np.int8)
    for a, b in itertools.combinations(range(n), 2):
        if rng.random() < p:
            dag[order[a], order[b]], dag[order[b], order[a]] = TAIL, ARROW
    return dag2cpdag(dag)


def test_a_nodes_most_parents_is_its_directed_parents_plus_its_largest_clique_of_neighbors() -> (
    None
):
    """A chain and a star leave every node one parent; a triangle two; a collider keeps both."""
    chain = _undirected(3, [(0, 1), (1, 2)])
    assert ops.max_parents_in_class(chain).tolist() == [1, 1, 1]
    assert not ops.exceeds_parent_cap(chain, 1)
    assert ops.exceeds_parent_cap(chain, 0)
    star = _undirected(4, [(0, 1), (0, 2), (0, 3)])
    assert ops.max_parents_in_class(star).tolist() == [1, 1, 1, 1]
    triangle = _undirected(3, [(0, 1), (1, 2), (0, 2)])
    assert ops.max_parents_in_class(triangle).tolist() == [2, 2, 2]
    assert ops.exceeds_parent_cap(triangle, 1) and not ops.exceeds_parent_cap(triangle, 2)
    # u1 -- b -- u2, with a -> b and the collider b -> c <- d: b has a and one neighbor, c two.
    mixed = _undirected(6, [(0, 1), (1, 2)])
    for tail, head in [(3, 1), (1, 4), (5, 4)]:
        mixed[tail, head], mixed[head, tail] = TAIL, ARROW
    assert ops.max_parents_in_class(mixed).tolist() == [1, 2, 1, 0, 2, 0]


@pytest.mark.parametrize("seed", range(40))
def test_the_bound_is_the_most_parents_over_the_class(seed: int) -> None:
    """On a random CPDAG of up to 7 nodes, listing the class gives the clique bound at every node,
    and the cap test agrees with it for every cap."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 8))
    cpdag = _random_cpdag(rng, n, float(rng.uniform(0.2, 0.9)))
    while ((cpdag == TAIL) & (cpdag.T == TAIL)).sum() // 2 > 11:  # too many to list the class
        cpdag = _random_cpdag(rng, n, float(rng.uniform(0.2, 0.9)))
    bound = ops.max_parents_in_class(cpdag)
    assert np.array_equal(bound, _most_parents_by_listing(cpdag))
    for cap in range(n):
        assert ops.exceeds_parent_cap(cpdag, cap) == bool((bound > cap).any())


# (seed, variables, rows, edge density): each gives an uncapped node more than three parents in
# some DAG, and the smaller ones keep their classes small enough to list every DAG.
_CASES = [(2, 5, 400, 0.9), (0, 7, 600, 0.9), (0, 12, 600, 0.5), (0, 25, 500, 0.3)]


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("seed,d,n,p", _CASES)
def test_no_dag_of_the_returned_class_gives_a_node_more_than_k_parents(
    seed: int, d: int, n: int, p: float, k: int
) -> None:
    """Every node is within ``k``, some DAG reaches the bound, and the old gate was not within."""
    x = _sem(seed, n, d, p)
    assert ops.max_parents_in_class(ges(x)[0].to_numpy()).max() > k, "the cap does not bind"
    gated = ges(x, max_parents=k, _parent_rule="gate")[0].to_numpy()
    assert ops.max_parents_in_class(gated).max() > k, "the gate already holds"
    cpdag = ges(x, max_parents=k)[0].to_numpy()
    bound = ops.max_parents_in_class(cpdag)
    assert (bound <= k).all()
    assert (_in_degrees(pdag2dag(cpdag)) <= bound).all()
    if d <= 7:
        assert np.array_equal(_most_parents_by_listing(cpdag), bound)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_a_column_permutation_gives_the_permuted_graph_under_a_cap(k: int) -> None:
    """The cap does not break the search's independence from the column order."""
    x = _sem(5, 500, 10, 0.6)
    perm = np.random.default_rng(k).permutation(x.shape[1])
    plain = ges(x, max_parents=k)[0].to_numpy()
    shuffled = ges(x[:, perm], max_parents=k)[0].to_numpy()
    back = np.empty_like(shuffled)
    back[np.ix_(perm, perm)] = shuffled
    assert not ops.exceeds_parent_cap(plain, k)
    assert np.array_equal(plain, back)


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("seed,d,n,p", [(0, 8, 500, 0.8), (1, 14, 500, 0.5)])
def test_the_prefilter_leaves_out_nothing_the_cap_allows(
    seed: int, d: int, n: int, p: float, k: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scoring only the Inserts whose family fits the cap returns what testing each move gives."""
    x = _sem(seed, n, d, p)
    filtered, filtered_score = ges(x, max_parents=k)
    monkeypatch.setattr(pg, "_over_cap", lambda cap, parents: False)
    tested, tested_score = ges(x, max_parents=k)
    assert np.array_equal(filtered.to_numpy(), tested.to_numpy())
    assert filtered_score == tested_score


@pytest.mark.parametrize("k", [1, 2, 3])
def test_the_cap_holds_under_background_knowledge(k: int) -> None:
    """Required and forbidden edges and tiers leave the cap intact, including their orientation."""
    x = _sem(4, 500, 10, 0.5)
    knowledge = Knowledge(
        required=[(0, 5)], forbidden=[(1, 2), (3, 4)], tiers=[[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    )
    cpdag = andrey.ges(x, max_parents=k, knowledge=knowledge).structure.to_numpy()
    assert ops.max_parents_in_class(ges(x)[0].to_numpy()).max() > k, "the cap does not bind"
    assert not ops.exceeds_parent_cap(cpdag, k)
    assert cpdag[0, 5] != 0, "the required edge is missing"


def test_an_insert_that_makes_a_collider_breaks_a_cap_of_one_but_a_chain_extension_does_not() -> (
    None
):
    """On the chain ``u -- y -- v``, ``x -> y <- u`` leaves ``y`` two parents; joining ``x`` as a
    third undirected neighbor leaves it one."""
    adj = _undirected(4, [(1, 2), (1, 3)])  # y = 1, u = 2, v = 3, x = 0
    assert pg._breaks_cap(adj, ops.insert_cpdag, (0, 1, (2,)), 1)
    assert not pg._breaks_cap(adj, ops.insert_cpdag, (0, 1, ()), 1)
    assert not pg._breaks_cap(adj, ops.insert_cpdag, (0, 1, (2,)), 2)
    assert not pg._breaks_cap(adj, ops.insert_cpdag, (0, 1, (2,)), None)


def test_a_backward_scan_asks_the_cap_before_it_selects_a_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Delete the scan would take is skipped when its CPDAG breaks the cap."""
    rng = np.random.default_rng(0)
    cause, noise = rng.standard_normal((2, 2000))
    x = np.column_stack([cause, cause + 0.5 * rng.standard_normal(2000), noise])
    adj = _undirected(3, [(0, 1), (0, 2)])  # the edge to the independent column should go
    engine = DeltaBICScore(BICScore(x, penalty_discount=1.0))
    ranks = tie_ranks(x).tolist()
    state = pg.neighborhood_state(adj)
    taken = pg.backward_chunk_best(range(3), adj, engine, 3, state, ranks, cap=3)
    assert taken is not None and {taken[1], taken[2]} == {0, 2} and taken[3] == ()
    monkeypatch.setattr(pg, "_breaks_cap", lambda *args: True)
    assert pg.backward_chunk_best(range(3), adj, engine, 3, state, ranks, cap=3) is None


def test_a_target_at_the_cap_has_no_forward_work() -> None:
    """A node whose parents already fill the cap can take no Insert, and the gate estimate knows."""
    adj = np.zeros((4, 4), dtype=np.int8)
    for tail in (0, 1):
        adj[tail, 3], adj[3, tail] = TAIL, ARROW
    state = pg.neighborhood_state(adj)
    assert pg.forward_work_per_target(4, state, None)[3] > 0
    assert pg.forward_work_per_target(4, state, None, 2)[3] == 0
    assert pg.forward_work_per_target(4, state, None, 3)[3] > 0
