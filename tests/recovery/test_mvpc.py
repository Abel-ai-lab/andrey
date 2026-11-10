"""Recovery with Missing-Value PC, ``pc(missing="mvpc")``, on masks whose m-graph is known.

The fixed case is Figure 1b of Tu et al. (2019): ``x -> w <- y``, with ``y`` missing when ``w`` is
low. Test-wise deletion reads only the rows where ``y`` is observed, a sample selected on their
common effect ``w``, so it finds ``x`` and ``y`` dependent and keeps an edge between them; MVPC
finds ``w`` as the parent of ``y``'s indicator and removes it. On seeded ground truth, each
indicator's parent is a collider and the indicator belongs to one of the collider's parents, as in
the paper's simulations: MVPC finds every placed parent, keeps no more extra edges than test-wise
deletion on any dataset, and makes fewer errors over all of them. Two fixed cases check the steps
those graphs leave untested. In the first, an indicator's parent can itself go missing (MNAR); the
parents of that parent's indicator are common effects of the tested pair, which breaks condition
(ii) of Tu et al. (2019), so the case checks the detection and the true edges, not the removal. In
the second, only the indicators a pair both causes make it a candidate for the corrected test.
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
        "corrected_edges": {
            "retested": [[0, 1], [0, 2], [1, 2]],
            "removed": [[0, 1]],
            "uncorrected": [],
        },
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


# The parents of the indicators of y, w, and v in :func:`two_colliders`.
TWO_COLLIDERS = {1: [2], 2: [3], 3: [2]}


def two_colliders(seed: int, n: int = 3000) -> np.ndarray:
    """``x -> w <- y`` and ``x -> v <- y`` in columns 0, 1, 2, 3, with ``y`` missing when ``w`` is
    low, ``w`` when ``v`` is, and ``v`` when ``w`` is: the parent of ``y``'s indicator can go
    missing, and the parent of its indicator is the other collider (MNAR)."""
    rng = np.random.default_rng(seed)
    x, y = rng.standard_normal(n), rng.standard_normal(n)
    w = x + y + 0.5 * rng.standard_normal(n)
    v = x - 0.5 * y + 0.5 * rng.standard_normal(n)
    X = np.column_stack([x, y, w, v])
    mechanism = missingness.mnar(0.4, form="quantile", parents=TWO_COLLIDERS)
    return np.where(missingness.draw(X, mechanism, seed=seed).mask, np.nan, X)


@pytest.fixture(scope="module")
def two_collider_fits():
    """Each seed's test-wise deletion and MVPC fits of :func:`two_colliders`."""
    return [(andrey.pc(X), andrey.pc(X, missing="mvpc")) for X in map(two_colliders, range(8))]


def test_mvpc_finds_indicator_parents_that_can_go_missing_and_keeps_every_true_edge(
    two_collider_fits,
):
    # The query x, y reads W = {w, v}: w, the parent of y's indicator, and v, the parent of w's.
    # The indicators of w and v are caused by v and w, common effects of x and y, so x and y are
    # dependent on the rows where W is observed, whose distribution PermC's virtual data follow.
    # The case breaks condition (ii) of Tu et al. (2019): the corrected test keeps x - y on some
    # seeds, and on more of them with more rows, so its removal is not asserted.
    true_edges = [(0, 2), (1, 2), (0, 3), (1, 3)]
    for deletion, corrected in two_collider_fits:
        assert deletion.structure.to_numpy()[0, 1] != 0
        found = corrected.metadata["missingness_parents"]
        assert all(set(pa) <= set(found[c]) for c, pa in TWO_COLLIDERS.items())
        assert all(corrected.structure.to_numpy()[e] != 0 for e in true_edges)


def chain(seed: int, n: int = 5000) -> np.ndarray:
    """``x -> s -> t -> y`` in columns 0, 2, 3, 1, with ``s`` and ``t`` missing when ``x + y`` is
    high.

    Every test that separates ``x`` and ``y`` reads ``s`` or ``t``, so test-wise deletion reads
    rows selected on a common effect of ``x`` and ``y`` and keeps an edge between them. In the
    deletion skeleton ``x`` and ``y`` share no adjacent variable: only the indicators they both
    cause make the pair a candidate. The mask is built by hand, as the logistic form's weights,
    scaled to standard deviation 1, select too weakly to keep the edge.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    s = x + 0.6 * rng.standard_normal(n)
    t = s + 0.6 * rng.standard_normal(n)
    y = t + 0.6 * rng.standard_normal(n)
    X = np.column_stack([x, y, s, t])
    high = x + y > np.quantile(x + y, 0.6)
    X[high & (rng.random(n) < 0.9), 2] = np.nan
    X[high & (rng.random(n) < 0.9), 3] = np.nan
    return X


@pytest.mark.parametrize("seed", range(4))
def test_mvpc_retests_a_pair_that_shares_only_an_indicator_and_recovers_the_chain(seed):
    X = chain(seed)
    assert andrey.pc(X).structure.to_numpy()[0, 1] != 0
    out = andrey.pc(X, missing="mvpc")
    assert out.structure.oriented_edges() == [
        (0, 2, "undirected"),
        (1, 3, "undirected"),
        (2, 3, "undirected"),
    ]
    found = out.metadata["missingness_parents"]
    assert {0, 1} <= set(found[2]) and {0, 1} <= set(found[3])
    assert [0, 1] in out.metadata["corrected_edges"]["removed"]


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
