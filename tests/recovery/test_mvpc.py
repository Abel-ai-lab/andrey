"""Recovery with Missing-Value PC, ``pc(missing="mvpc")``, on masks whose m-graph is known.

The fixed case is Figure 1b of Tu et al. (2019): ``x -> w <- y``, with ``y`` missing when ``w`` is
low. Test-wise deletion reads only the rows where ``y`` is observed, a sample selected on their
common effect ``w``, so it finds ``x`` and ``y`` dependent and keeps an edge between them; MVPC
finds ``w`` as the parent of ``y``'s indicator and removes it. On seeded ground truth, each
indicator's parent is a collider and the indicator belongs to one of the collider's parents, as in
the paper's simulations: MVPC finds every placed parent, keeps no more extra edges than test-wise
deletion on any dataset, and makes fewer errors over all of them.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey import data
from andrey.core.ci import make_indep_test
from andrey.data import missingness
from andrey.metrics import shd


def figure_1b(seed: int = 0, n: int = 3000) -> np.ndarray:
    """``x -> w <- y`` in columns 0, 2, 1, with ``y`` missing below a quantile of ``w``."""
    rng = np.random.default_rng(seed)
    x, y = rng.standard_normal(n), rng.standard_normal(n)
    X = np.column_stack([x, y, x + y + 0.5 * rng.standard_normal(n)])
    mask = missingness.draw(X, missingness.mar(0.4, form="quantile", parents={1: [2]}), seed=seed)
    return np.where(mask.mask, np.nan, X)


@pytest.mark.parametrize("seed", range(4))
def test_test_wise_deletion_finds_the_pair_of_figure_1b_dependent_and_keeps_its_edge(seed):
    X = figure_1b(seed)
    assert make_indep_test("fisherz", X)(0, 1) < 1e-6
    assert andrey.pc(X).structure.to_numpy()[0, 1] != 0


@pytest.mark.parametrize("seed", range(4))
def test_mvpc_finds_the_indicator_parent_removes_the_edge_and_orients_the_collider(seed):
    out = andrey.pc(figure_1b(seed), missing="mvpc")
    assert out.structure.oriented_edges() == [(0, 2, "directed"), (1, 2, "directed")]
    assert out.metadata["missingness_parents"] == {1: [2]}
    assert out.metadata["corrected_edges"]["removed"] == [[0, 1]]


def test_a_seeded_fit_of_figure_1b_records_its_output():
    out = andrey.pc(figure_1b(0), missing="mvpc", seed=0)
    assert dict(out.metadata) == {
        "algorithm": "pc",
        "conflicts": [],
        "rows_tested": {"min": 1796, "median": 1796.0},
        "missingness_parents": {1: [2]},
        "corrected_edges": {"retested": [[0, 1], [0, 2], [1, 2]], "removed": [[0, 1]]},
        "seed": 0,
        "indep_test": "fisherz",
        "tie_break": "rank",
        "missing": "mvpc",
    }


def test_independent_columns_with_a_mar_mask_give_no_edge():
    X = np.random.default_rng(0).standard_normal((2000, 4))
    mask = missingness.draw(X, missingness.mar(0.3, form="quantile", parents={1: [0]}), seed=0)
    out = andrey.pc(np.where(mask.mask, np.nan, X), missing="mvpc")
    assert out.structure.oriented_edges() == []


def _collider_mask(ds, most: int = 3) -> dict[int, list[int]]:
    """Up to ``most`` indicators, each of one parent of a collider, with that collider as parent."""
    d = ds.data.shape[1]
    adjacency = np.zeros((d, d), dtype=bool)
    adjacency[ds.params.edges[:, 0], ds.params.edges[:, 1]] = True
    parents: dict[int, list[int]] = {}
    for c in range(d):
        pa = np.flatnonzero(adjacency[:, c])
        for a in pa:
            others = [b for b in pa if b != a and not adjacency[a, b] and not adjacency[b, a]]
            causes = {p for ps in parents.values() for p in ps}
            if others and a not in parents and a not in causes and c not in parents:
                parents[int(a)] = [int(c)]
                break
        if len(parents) == most:
            break
    return parents


def _extra_edges(structure, truth) -> int:
    est, true = structure.to_numpy() != 0, truth.to_numpy() != 0
    return int(np.triu(est & ~(true | true.T), 1).sum())


@pytest.fixture(scope="module")
def ground_truth():
    """Each seed's placed parents, found parents, extra edges, and errors, deletion then MVPC."""
    return [_ground_truth_row(seed) for seed in range(8)]


def _ground_truth_row(seed: int) -> tuple:
    """The placed and found parents, extra edges, and errors of deletion then MVPC on ``seed``."""
    ds = data.sample_scm(d=10, n=2000, seed=seed, density=2.5)
    placed = _collider_mask(ds)
    mechanism = missingness.mar(0.3, form="quantile", parents=placed)
    masked = data.sample_scm(d=10, n=2000, seed=seed, density=2.5, missing=mechanism)
    deletion = andrey.pc(masked.data)
    corrected = andrey.pc(masked.data, missing="mvpc")
    return (
        placed,
        corrected.metadata["missingness_parents"],
        [_extra_edges(f.structure, ds.graph) for f in (deletion, corrected)],
        [shd(f.structure, ds.graph) for f in (deletion, corrected)],
    )


def test_mvpc_finds_every_placed_indicator_parent(ground_truth):
    for placed, found, _, _ in ground_truth:
        assert placed and all(set(pa) <= set(found[c]) for c, pa in placed.items())


def test_mvpc_keeps_no_more_extra_edges_than_test_wise_deletion(ground_truth):
    extras = np.array([extra for _, _, extra, _ in ground_truth])
    assert (extras[:, 1] <= extras[:, 0]).all() and extras[:, 1].sum() < extras[:, 0].sum()


def test_mvpc_makes_fewer_errors_than_test_wise_deletion_over_the_datasets(ground_truth):
    errors = np.array([e for _, _, _, e in ground_truth])
    assert errors[:, 1].sum() < errors[:, 0].sum()
