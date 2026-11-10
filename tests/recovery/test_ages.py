"""AGES: fixed cases, the solution path against separate GES fits, recovery on the paper's
simulation, and the recorded output."""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.search.ages import solution_path
from tests.recovery.helpers import datagen
from tests.recovery.helpers.harness import assert_recorded_output

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")


def weak_edge_sem(seed: int, n_variables: int = 8, n_samples: int = 5000):
    """Data and weights (``[i, j]``: ``i -> j``) of the simulation of Eigenmann, Nandy and Maathuis
    (2017), Section 4.1: an edge is strong (0.8 to 1.2) with probability 0.3 and weak (0.1 to 0.3)
    with probability 0.3, noise variances from 0.5 to 1.5, the columns shuffled."""
    rng = np.random.default_rng(seed)
    rows, cols = np.triu_indices(n_variables, 1)
    kind = rng.random(rows.size)
    size = np.where(kind < 0.3, rng.uniform(0.8, 1.2, rows.size), rng.uniform(0.1, 0.3, rows.size))
    weights = np.zeros((n_variables, n_variables))
    weights[rows, cols] = np.where(kind < 0.6, size * rng.choice([-1.0, 1.0], rows.size), 0.0)
    noise = rng.standard_normal((n_samples, n_variables)) * np.sqrt(
        rng.uniform(0.5, 1.5, n_variables)
    )
    order = rng.permutation(n_variables)
    data = noise @ np.linalg.inv(np.eye(n_variables) - weights)
    return data[:, order], weights[np.ix_(order, order)]


def directed(structure) -> set[tuple[int, int]]:
    return {(i, j) for i, j, kind in structure.oriented_edges() if kind == "directed"}


def skeleton(structure) -> np.ndarray:
    return structure.to_numpy() != 0


# ---- fixed cases ---------------------------------------------------------------------------------


def test_directs_the_collider_a_weak_edge_hides():
    """Example 1.1 of the paper: with the weak edge x0 -> x1 found, the CPDAG of the triangle is
    undirected; the CPDAG without it holds x0 -> x2 <- x1, which AGES takes."""
    x = datagen.generate("weak_triangle_3v")
    x0, x1, x2 = np.argsort(datagen.column_order("weak_triangle_3v")).tolist()
    assert directed(andrey.ges(x, ebic_gamma=0.0).structure) == set()
    out = andrey.ages(x)
    assert directed(out.structure) == {(x0, x2), (x1, x2)}
    assert skeleton(out.structure).sum() == 6
    assert out.metadata["path_aggregated"][0] is True
    assert out.metadata["path_penalties"][0] == 1.0
    assert out.metadata["path_penalties"] == sorted(out.metadata["path_penalties"])


def test_equals_the_ges_cpdag_without_a_weak_edge():
    """A strong collider: the CPDAGs of the path direct nothing the first leaves undirected."""
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal((2, 2000))
    x = np.column_stack([a, b, a + b + rng.standard_normal(2000)])
    out, ges = andrey.ages(x), andrey.ges(x, ebic_gamma=0.0)
    assert np.array_equal(out.structure.to_numpy(), ges.structure.to_numpy())
    assert directed(out.structure) == {(0, 2), (1, 2)}


def test_matches_the_recorded_output_on_weak_triangle_3v():
    out = andrey.ages(datagen.generate("weak_triangle_3v"))
    assert_recorded_output(out, "AGES", case="weak_triangle_3v")


# ---- the solution path ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_each_cpdag_of_the_path_is_the_ges_cpdag_at_its_penalty(seed):
    """A separate GES fit without the forward gate just above each penalty of the path returns
    that entry's CPDAG, and one at a penalty between two entries returns the lower entry's."""
    x, _ = weak_edge_sem(seed)
    path = solution_path(x)
    penalties = [penalty for penalty, _ in path]
    assert penalties[0] == 1.0
    assert len(path) > 3
    between = [(low + high) / 2 for low, high in zip(penalties, penalties[1:], strict=False)]
    for penalty, (_, cpdag) in zip(
        [p * (1 + 1e-9) for p in penalties] + between, path + path[:-1], strict=True
    ):
        ges = andrey.ges(x, ebic_gamma=0.0, penalty_discount=penalty, max_parents=x.shape[1] - 1)
        assert np.array_equal(ges.structure.to_numpy(), cpdag)
    assert not path[-1][1].any()  # the highest penalty leaves the empty graph


# ---- ground truth --------------------------------------------------------------------------------


def test_directs_more_true_edges_than_ges_on_the_papers_simulation():
    """Over 12 graphs with weak edges, AGES keeps the GES skeleton and every direction of its CPDAG,
    directs more edges as the true DAG does, and is as precise within 0.05."""
    correct = {"ges": 0, "ages": 0}
    found = {"ges": 0, "ages": 0}
    for seed in range(12):
        x, weights = weak_edge_sem(seed)
        truth = set(zip(*np.nonzero(weights), strict=True))
        ges = andrey.ges(x, ebic_gamma=0.0, max_parents=x.shape[1] - 1).structure
        out = andrey.ages(x).structure
        assert np.array_equal(skeleton(out), skeleton(ges))
        assert directed(ges) <= directed(out)
        for name, structure in (("ges", ges), ("ages", out)):
            correct[name] += len(directed(structure) & truth)
            found[name] += len(directed(structure))
    assert correct["ages"] > correct["ges"]
    assert correct["ages"] / found["ages"] >= correct["ges"] / found["ges"] - 0.05


# ---- invariance ----------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_does_not_depend_on_the_column_order_or_the_units(seed):
    x, _ = weak_edge_sem(seed)
    rng = np.random.default_rng(100 + seed)
    order = rng.permutation(x.shape[1])
    scale = rng.uniform(0.01, 100.0, x.shape[1])
    base = andrey.ages(x).structure.to_numpy()
    moved = andrey.ages((x * scale)[:, order]).structure.to_numpy()
    assert np.array_equal(moved, base[np.ix_(order, order)])
