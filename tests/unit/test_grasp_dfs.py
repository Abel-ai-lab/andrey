"""GRaSP's depth-first search tucks the edges of the current DAG and follows each chain of ties.

Each level of the search tries a tuck on every edge ``x -> y`` of the DAG it starts from: any edge
at the first level, covered edges below it. A tuck that leaves the score unchanged starts a level
below it, on a DAG the chain of tucks has not visited.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey.data as data
from andrey.search import grasp as grasp_mod


def _sample() -> np.ndarray:
    """Linear-Gaussian data on 10 variables whose search tries many tucks and ties."""
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=10, avg_degree=3),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    return np.asarray(scm.sample(n=100, seed=0).data)


def test_every_tuck_is_on_an_edge_and_the_last_pass_tries_every_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every tuck moves a child before one of its parents, and the pass that finds no improving
    chain starts one from every edge of the final DAG, once each."""
    orders: list[grasp_mod._Order] = []
    passes: list[list[tuple[int, int]]] = []
    level = [0]
    original_init, original_dfs, original_tuck = (
        grasp_mod._Order.__init__,
        grasp_mod._dfs,
        grasp_mod._tuck,
    )

    def capture_order(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        orders.append(self)

    def dfs(*args, **kwargs):
        if level[0] == 0:
            passes.append([])
        level[0] += 1
        try:
            return original_dfs(*args, **kwargs)
        finally:
            level[0] -= 1

    def tuck(i: int, j: int, order: grasp_mod._Order) -> None:
        x, y = order.get(j), order.get(i)
        assert x in order.get_parents(y), f"tuck of {y} before {x}, which is not its parent"
        if level[0] == 1:
            passes[-1].append((x, y))
        original_tuck(i, j, order)

    monkeypatch.setattr(grasp_mod._Order, "__init__", capture_order)
    monkeypatch.setattr(grasp_mod, "_dfs", dfs)
    monkeypatch.setattr(grasp_mod, "_tuck", tuck)
    grasp_mod.grasp(_sample(), penalty_discount=1.0, random_state=0)

    (order,) = orders
    edges = sorted((x, y) for y in order.order for x in order.get_parents(y))
    assert len(edges) > 5
    assert sorted(passes[-1]) == edges


def _dag(order: grasp_mod._Order) -> frozenset[tuple[int, int]]:
    return frozenset((x, y) for y in order.order for x in order.get_parents(y))


def test_a_tie_extends_the_chain_to_each_new_dag_and_never_to_one_in_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tuck that leaves the score unchanged, with depth left, leads one level deeper unless its
    DAG is already in the chain of tucks that reached it, so no chain visits a DAG twice."""
    chain: list[frozenset[tuple[int, int]]] = []
    depths: list[int] = []
    unexplored = [False]
    extended = [0]
    original_dfs, original_update, original_restore = (
        grasp_mod._dfs,
        grasp_mod._update,
        grasp_mod._restore,
    )

    def dfs(depth, *args, **kwargs):
        order = next(arg for arg in args if isinstance(arg, grasp_mod._Order))
        assert _dag(order) not in chain, "a chain revisits a DAG"
        extended[0] += bool(chain)
        unexplored[0] = False
        chain.append(_dag(order))
        depths.append(depth)
        try:
            return original_dfs(depth, *args, **kwargs)
        finally:
            chain.pop()
            depths.pop()

    def update(i: int, j: int, order: grasp_mod._Order, gsts):
        edge_bump, score_bump = original_update(i, j, order, gsts)
        tie = -grasp_mod._SCORE_EPS < score_bump <= grasp_mod._SCORE_EPS
        unexplored[0] = depths[-1] > 0 and tie and _dag(order) not in chain
        return edge_bump, score_bump

    def restore(order: grasp_mod._Order, start: int, snapshot: tuple) -> None:
        assert not unexplored[0], "a tie to a new DAG is rolled back unexplored"
        original_restore(order, start, snapshot)

    monkeypatch.setattr(grasp_mod, "_dfs", dfs)
    monkeypatch.setattr(grasp_mod, "_update", update)
    monkeypatch.setattr(grasp_mod, "_restore", restore)
    grasp_mod.grasp(_sample(), penalty_discount=1.0, random_state=0)

    assert extended[0] > 5
