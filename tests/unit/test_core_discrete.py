"""The categorical primitives: the chi-square and G-square tests, the BDeu score, and the codes.

Each statistic is checked against an independent computation: SciPy's contingency-table test for
one table, a per-configuration loop for a conditional test, and the BDeu formula written out with
``math.lgamma``.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest
from scipy.stats import chi2, chi2_contingency

from andrey.core import categorical
from andrey.core.ci import ChiSquare
from andrey.core.score import BDeuScore


def _codes(seed: int = 0, n: int = 400, cards=(2, 3, 4, 3)) -> np.ndarray:
    """Categorical columns where column 1 copies column 0 in most rows, so some pairs depend."""
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.integers(0, k, n) for k in cards])
    x[:, 1] = np.where(rng.random(n) < 0.6, x[:, 0], x[:, 1])
    return x.astype(np.float64)


def _table(x: np.ndarray, a: int, b: int) -> np.ndarray:
    """The observed table of columns ``a`` and ``b``, without empty rows or columns."""
    va, ia = np.unique(x[:, a], return_inverse=True)
    vb, ib = np.unique(x[:, b], return_inverse=True)
    table = np.zeros((va.size, vb.size))
    np.add.at(table, (ia, ib), 1)
    return table


@pytest.mark.parametrize(("g_square", "lambda_"), [(False, None), (True, "log-likelihood")])
def test_a_marginal_test_matches_scipy(g_square, lambda_):
    x = _codes()
    test = ChiSquare(x, g_square=g_square)
    for a, b in combinations(range(x.shape[1]), 2):
        stat, p, dof, _ = chi2_contingency(_table(x, a, b), correction=False, lambda_=lambda_)
        assert test(a, b) == pytest.approx(p, rel=1e-9, abs=1e-300)


@pytest.mark.parametrize(("g_square", "lambda_"), [(False, None), (True, "log-likelihood")])
def test_a_conditional_test_sums_one_table_per_configuration(g_square, lambda_):
    """The statistic and degrees of freedom add up over the conditioning configurations, each
    table counting only the values present in it."""
    x = _codes(seed=1, n=600)
    statistic = dof = 0.0
    for config in np.unique(x[:, [2, 3]], axis=0):
        rows = x[(x[:, [2, 3]] == config).all(axis=1)]
        table = _table(rows, 0, 1)
        if min(table.shape) > 1:
            s, _, k, _ = chi2_contingency(table, correction=False, lambda_=lambda_)
            statistic, dof = statistic + s, dof + k
    assert ChiSquare(x, g_square=g_square)(0, 1, (2, 3)) == pytest.approx(
        chi2.sf(statistic, dof), rel=1e-9
    )


def test_a_pair_with_no_degrees_of_freedom_has_p_value_one():
    """When ``y`` takes one value in every configuration of the conditioning set, no table has a
    degree of freedom, and the test cannot reject independence."""
    x = _codes()
    x[:, 3] = x[:, 2]  # y is a copy of the conditioning column
    for g_square in (False, True):
        assert ChiSquare(x, g_square=g_square)(0, 3, (2,)) == 1.0


def test_dependence_and_independence_are_told_apart():
    x = _codes(n=2000)
    test = ChiSquare(x)
    assert test(0, 1) < 1e-10  # column 1 copies column 0
    assert test(2, 3) > 0.01  # independent draws


def test_p_values_do_not_depend_on_codes_or_argument_order():
    x = _codes()
    recoded = x * np.array([3.0, -2.0, 1.0, 7.0]) + 5.0
    for g in (False, True):
        a, b = ChiSquare(x, g_square=g), ChiSquare(recoded, g_square=g)
        assert a(1, 0, (3, 2, 3)) == a(0, 1, (2, 3))
        assert b(0, 1, (2, 3)) == pytest.approx(a(0, 1, (2, 3)), rel=1e-12)


def test_a_sparse_table_counts_only_the_occupied_cells(monkeypatch):
    """Above the dense-table limit the cells that occur are counted directly; same p-value."""
    from andrey.core import ci

    x = _codes(n=300, cards=(4, 5, 6, 6))
    dense = ChiSquare(x)(0, 1, (2, 3))
    monkeypatch.setattr(ci, "_DENSE_CELLS", 0)
    assert ChiSquare(x)(0, 1, (2, 3)) == pytest.approx(dense, rel=1e-12)


def _bdeu(counts: np.ndarray, q: int, a: float = 1.0) -> float:
    """Log BDeu marginal likelihood of ``counts`` (one row per parent configuration)."""
    r = counts.shape[1]
    total = 0.0
    for row in counts:
        total += math.lgamma(a / q) - math.lgamma(row.sum() + a / q)
        for c in row:
            total += math.lgamma(c + a / (r * q)) - math.lgamma(a / (r * q))
    return total


def test_bdeu_matches_the_formula():
    """Three variables, so the structure prior gives each other variable probability 1/2."""
    x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1]], float)
    score = BDeuScore(x)
    half = math.log(0.5)
    # Node 0 with no parent: counts (4, 2).
    assert score.score(0, []) == pytest.approx(-(_bdeu(np.array([[4, 2]]), 1) + 2 * half))
    # Node 0 given node 1: (2, 0) where node 1 is 0, (2, 2) where it is 1.
    assert score.score(0, [1]) == pytest.approx(-(_bdeu(np.array([[2, 0], [2, 2]]), 2) + 2 * half))


def test_bdeu_structure_prior_penalizes_each_parent():
    """From five variables up, p = 1 / (d - 1) makes each parent cost log((1 - p) / p) more."""
    x = _codes(n=300, cards=(2, 3, 2, 3, 2))
    with_prior = BDeuScore(x)
    p = 1 / 4
    penalty = math.log((1 - p) / p)
    likelihood_gain = BDeuScore(x[:, :3]).score(0, [1]) - BDeuScore(x[:, :3]).score(0, [])
    # At d = 3 the prior is flat (p = 1/2), so the difference there is the likelihood's alone.
    assert with_prior.score(0, [1]) - with_prior.score(0, []) == pytest.approx(
        likelihood_gain + penalty
    )


def test_bdeu_is_score_equivalent():
    """``x -> y`` and ``y -> x`` describe one equivalence class, so they score the same."""
    x = _codes(n=500)
    s = BDeuScore(x)
    for a, b in combinations(range(x.shape[1]), 2):
        assert s.score(a, [b]) + s.score(b, []) == pytest.approx(s.score(b, [a]) + s.score(a, []))


@pytest.mark.parametrize("d", [1, 2])
def test_bdeu_leaves_out_an_undefined_structure_prior(d):
    """With two or fewer variables, structure_prior / (d - 1) is not below 1: no prior term."""
    x = _codes()[:, :d]
    score = BDeuScore(x)
    counts = np.bincount(x[:, 0].astype(int))[None, :]
    assert score.score(0, []) == pytest.approx(-_bdeu(counts, 1))
    if d == 2:
        assert np.isfinite(score.score(0, [1]))


def test_bdeu_does_not_depend_on_the_codes():
    x = _codes()
    a, b = BDeuScore(x), BDeuScore(x * -3.0 + 2.0)
    assert b.score(1, [2, 0]) == pytest.approx(a.score(1, [0, 2]), rel=1e-12)


@pytest.mark.parametrize(("name", "value"), [("sample_prior", 0.0), ("structure_prior", -1.0)])
def test_bdeu_priors_must_be_positive(name, value):
    with pytest.raises(ValueError, match=name):
        BDeuScore(_codes(), **{name: value})


def test_configurations_number_the_observed_rows_in_code_order():
    """Many high-cardinality columns overflow a plain mixed-radix code; the configurations that
    occur are still numbered 0..m-1 in lexicographic order, at most one per row."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 50, (1000, 12)).astype(np.float64)
    x[:500, :6] = 0.0  # repeated rows, so some configurations occur more than once
    codes, cards = categorical.encode(x)
    columns = list(range(12))
    index, m = categorical.configurations(codes, cards, columns)
    _, expected = np.unique(codes[:, columns], axis=0, return_inverse=True)
    assert m == expected.max() + 1 <= len(x)
    assert np.array_equal(index, expected.reshape(-1))


def test_no_columns_is_one_configuration():
    codes, cards = categorical.encode(_codes())
    index, m = categorical.configurations(codes, cards, [])
    assert m == 1 and not index.any()
