"""``max_parents`` caps every node's parents plus undirected neighbors in each CPDAG of the search.

The cap holds in the returned CPDAG and so in every DAG of its class; a move whose rebuilt CPDAG
would break it is never taken, in either phase; and the checks that avoid scoring such a move
change nothing the post-move test would have allowed.
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
    b = (
        np.triu(rng.random((d, d)) < p, 1)
        * rng.uniform(0.5, 1.2, (d, d))
        * rng.choice([-1, 1], (d, d))
    )
    x = rng.standard_normal((n, d))
    for j in range(d):
        x[:, j] += x @ b[:, j]
    return x


def _in_degrees(dag: np.ndarray) -> np.ndarray:
    """Each node's parent count in a DAG of marks."""
    return (dag == ARROW).sum(axis=1)


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


# (seed, variables, rows, edge density): each gives an uncapped node more than three parents, and
# the smaller ones keep their classes small enough to list every DAG.
_CASES = [(2, 5, 400, 0.9), (0, 7, 600, 0.9), (0, 12, 600, 0.5), (0, 25, 500, 0.3)]


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("seed,d,n,p", _CASES)
def test_no_node_exceeds_the_cap_in_the_cpdag_or_in_any_dag_of_its_class(
    seed: int, d: int, n: int, p: float, k: int
) -> None:
    """Parents plus undirected neighbors stay within ``k``; every DAG of the class follows."""
    x = _sem(seed, n, d, p)
    assert ops.parent_counts(ges(x)[0].to_numpy()).max() > k, "the cap does not bind"
    cpdag = ges(x, max_parents=k)[0].to_numpy()
    assert (ops.parent_counts(cpdag) <= k).all()
    assert (_in_degrees(pdag2dag(cpdag)) <= k).all()
    if d <= 7:
        dags = list(_dags_of_class(cpdag))
        assert dags, "the class holds no DAG"
        assert all((_in_degrees(dag) <= k).all() for dag in dags)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_a_column_permutation_gives_the_permuted_graph_under_a_cap(k: int) -> None:
    """The cap does not break the search's independence from the column order."""
    x = _sem(5, 500, 10, 0.6)
    perm = np.random.default_rng(k).permutation(x.shape[1])
    plain = ges(x, max_parents=k)[0].to_numpy()
    shuffled = ges(x[:, perm], max_parents=k)[0].to_numpy()
    back = np.empty_like(shuffled)
    back[np.ix_(perm, perm)] = shuffled
    assert (ops.parent_counts(plain) <= k).all()
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
    assert ops.parent_counts(ges(x)[0].to_numpy()).max() > k, "the cap does not bind"
    assert (ops.parent_counts(cpdag) <= k).all()
    assert cpdag[0, 5] != 0, "the required edge is missing"


def _star_with_a_collider() -> np.ndarray:
    """A CPDAG whose edge ``a -> c`` is the only one holding ``b -> c`` in place.

    ``u1 -- b -- u2`` is undirected, and ``b -> c <- a`` is a collider: ``b`` has two undirected
    neighbors, ``c`` two parents. Deleting ``a -> c`` leaves ``b -- c`` undirected, three
    undirected neighbors for ``b``.
    """
    u1, u2, b, a, c = range(5)
    adj = np.zeros((5, 5), dtype=np.int8)
    for line in [(u1, b), (u2, b)]:
        adj[line] = adj[line[::-1]] = TAIL
    for tail, head in [(b, c), (a, c)]:
        adj[tail, head], adj[head, tail] = TAIL, ARROW
    return adj


def test_a_delete_that_undirects_an_edge_into_a_full_node_breaks_the_cap() -> None:
    """The rebuild after a Delete can raise another node's count, and a scan skips that Delete."""
    adj = _star_with_a_collider()
    assert ops.parent_counts(adj).tolist() == [1, 1, 2, 0, 2]
    delete = (3, 4, ())  # a -> c
    assert ops.parent_counts(ops.delete_cpdag(adj.copy(), *delete)).tolist() == [1, 1, 3, 0, 1]
    assert pg._breaks_cap(adj, ops.delete_cpdag, delete, 2)
    assert not pg._breaks_cap(adj, ops.delete_cpdag, delete, 3)
    assert not pg._breaks_cap(adj, ops.delete_cpdag, delete, None)

    # Data where that delete is the best one: ``a`` is independent of ``c``.
    rng = np.random.default_rng(0)
    u1, u2, b, a, c = (rng.standard_normal(2000) for _ in range(5))
    x = np.column_stack([b + 0.5 * u1, b + 0.5 * u2, b, a, b + 0.5 * c])
    engine = DeltaBICScore(BICScore(x, lambda_value=1.0))
    ranks = tie_ranks(x).tolist()
    state = pg.neighborhood_state(adj)
    uncapped = pg.backward_chunk_best(range(5), adj, engine, 5, state, ranks)
    capped = pg.backward_chunk_best(range(5), adj, engine, 5, state, ranks, cap=2)
    assert uncapped is not None and uncapped[1:] == (3, 4, ())
    assert capped is None or capped[1:] != uncapped[1:]


def test_a_target_at_the_cap_has_no_forward_work() -> None:
    """A node whose parents already fill the cap can take no Insert, and the gate estimate knows."""
    adj = np.zeros((4, 4), dtype=np.int8)
    for tail in (0, 1):
        adj[tail, 3], adj[3, tail] = TAIL, ARROW
    state = pg.neighborhood_state(adj)
    assert pg.forward_work_per_target(4, state, None)[3] > 0
    assert pg.forward_work_per_target(4, state, None, 2)[3] == 0
    assert pg.forward_work_per_target(4, state, None, 3)[3] > 0
