"""GRaSP's depth-first search tucks the edges of the current DAG.

Each level of the search tries a tuck on every edge ``x -> y`` of the DAG it starts from: any edge
at the first level, covered edges below it.
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
    grasp_mod.grasp(_sample(), lambda_value=1.0, random_state=0)

    (order,) = orders
    edges = sorted((x, y) for y in order.order for x in order.get_parents(y))
    assert len(edges) > 5
    assert sorted(passes[-1]) == edges
