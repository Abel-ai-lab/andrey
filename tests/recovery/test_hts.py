"""LHTS and NHTS recover their models' graphs: fixed cases, seeded ground truth, and recorded
outputs.

The fixed cases have one certain answer: independent columns give no edge and one layer, and a
cause with a linear (LHTS) or quadratic (NHTS) effect and uniform noise points to it in either
column order. The ground truth is each method's paper simulation at test size: a linear model with
uniform noise for LHTS and the quadratic mechanism ``(sum_k w_k x_k)^2`` with Gaussian noise for
NHTS, on random graphs with as many edges as variables, columns standardized; summed over three
graphs, the fit's errors stay below the empty graph's, and its edges follow its causal order. The
recorded outputs pin the graph and the order on one fixed case each.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.api._adapt import dag_from_adjacency
from andrey.metrics import shd
from tests.recovery.helpers import datagen
from tests.recovery.helpers.harness import assert_recorded_output

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")


def _edges(out: andrey.StructureOutput) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j, kind in out.structure.oriented_edges() if kind == "directed"}


def _graph(d: int, rng: np.random.Generator) -> np.ndarray:
    """A random DAG over ``d`` nodes with ``d`` edges on average, in a random order."""
    order = rng.permutation(d)
    upper = np.triu(rng.random((d, d)) < 2.0 / (d - 1), 1)
    adj = np.zeros((d, d), dtype=bool)
    adj[np.ix_(order, order)] = upper
    return adj


def _simulate(adj: np.ndarray, n: int, mechanism: str, rng: np.random.Generator) -> np.ndarray:
    """Columns in a topological order of ``adj``, each standardized after it is generated."""
    d = adj.shape[0]
    X = np.zeros((n, d))
    left = set(range(d))
    while left:
        for j in sorted(v for v in left if not adj[list(left), v].any()):
            parents = np.flatnonzero(adj[:, j])
            if mechanism == "linear":
                w = rng.choice([-1.0, 1.0], parents.size) * rng.uniform(0.5, 1.5, parents.size)
                X[:, j] = X[:, parents] @ w + rng.uniform(-1, 1, n)
            else:
                w = rng.choice([-1.0, 1.0], parents.size) * rng.uniform(1.5, 2.5, parents.size)
                X[:, j] = (X[:, parents] @ w) ** 2 + rng.standard_normal(n)
            X[:, j] = (X[:, j] - X[:, j].mean()) / X[:, j].std()
            left.remove(j)
    return X


def _truth(adj: np.ndarray) -> andrey.GraphStructure:
    return dag_from_adjacency(adj)


@pytest.mark.parametrize("method", ["lhts", "nhts"])
def test_independent_columns_give_no_edges(method):
    out = getattr(andrey, method)(np.random.default_rng(0).uniform(-1, 1, (300, 3)))
    assert out.structure.oriented_edges() == []
    assert out.metadata["partial_order"] == [[0, 1, 2]]


@pytest.mark.parametrize("reverse", [False, True], ids=["cause first", "effect first"])
@pytest.mark.parametrize("method", ["lhts", "nhts"])
def test_a_cause_points_to_its_effect(method, reverse):
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, 300)
    y = (2 * x if method == "lhts" else (2 * x) ** 2) + 0.5 * rng.uniform(-1, 1, 300)
    X = np.column_stack([y, x] if reverse else [x, y])
    cause, effect = (1, 0) if reverse else (0, 1)
    out = getattr(andrey, method)(X)
    assert _edges(out) == {(cause, effect)}
    assert out.ordering == (cause, effect)


@pytest.mark.parametrize(
    ("method", "mechanism", "d", "n"), [("lhts", "linear", 6, 500), ("nhts", "quadratic", 5, 300)]
)
def test_graphs_of_the_papers_simulation_beat_the_empty_graph(method, mechanism, d, n):
    errors = edges = 0
    for seed in range(3):
        rng = np.random.default_rng(seed)
        adj = _graph(d, rng)
        out = getattr(andrey, method)(_simulate(adj, n, mechanism, rng))
        errors += shd(out.structure, _truth(adj))
        edges += int(adj.sum())
        order = list(out.ordering)
        assert all(order.index(i) < order.index(j) for i, j in _edges(out))
    assert errors < edges


def test_a_rerun_repeats():
    X = datagen.generate("lingam_5v_uniform")[:400]
    first, second = andrey.lhts(X), andrey.lhts(X)
    assert _edges(first) == _edges(second)
    assert first.metadata["partial_order"] == second.metadata["partial_order"]


def test_the_recorded_outputs_repeat():
    assert_recorded_output(
        andrey.lhts(datagen.generate("lingam_5v_uniform")), "LHTS", case="lingam_5v_uniform"
    )
    assert_recorded_output(
        andrey.nhts(datagen.generate("nonmonotone_4v")), "NHTS", case="nonmonotone_4v"
    )
