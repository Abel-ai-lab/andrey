"""The mixed-data primitives: the degenerate and conditional Gaussian tests and scores.

Each likelihood is checked against an independent computation: SciPy's multivariate normal density
on the rows of each joint category, least-squares regressions on indicator columns written out by
hand, and the partial correlation of two residuals. The tests' p-values are checked to be uniform
on simulated data where the independence holds, a category of two rows included.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import chi2, kstest, multivariate_normal

from andrey.core.ci import CGTest, ChiSquare, DGTest, make_indep_test
from andrey.core.mixed import ConditionalGaussian, column_mask
from andrey.core.score import BICScore, CGScore, DGScore
from andrey.core.score_delta import FullScoreDelta
from andrey.data import sample_scm
from andrey.metrics import to_cpdag
from andrey.search.boss import boss
from andrey.search.exact import exact_search
from andrey.search.ges import ges
from andrey.search.grasp import grasp
from andrey.search.hc import hc


def _mixed(seed: int = 0, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Columns a (3 categories), x, b (2 categories), y: a -> x -> y <- b, b also scales y."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 3, n)
    b = rng.integers(0, 2, n)
    x = np.array([0.0, 1.5, -1.0])[a] + rng.standard_normal(n)
    y = 0.8 * x + 0.5 * b + (1.0 + b) * rng.standard_normal(n)
    return np.column_stack([a, x, b, y]).astype(np.float64), np.array([True, False, True, False])


def _cells(x: np.ndarray, columns: list[int]) -> list[np.ndarray]:
    """The row indices of each joint category of ``columns`` that occurs."""
    if not columns:
        return [np.arange(len(x))]
    _, inverse = np.unique(x[:, columns], axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    return [np.flatnonzero(inverse == c) for c in range(inverse.max() + 1)]


def _cg_loglik(x: np.ndarray, mask: np.ndarray, variables: list[int]) -> float:
    """The conditional Gaussian log-likelihood, one multivariate normal per joint category.

    Each category's normal has the category's mean and its covariance shrunk toward the covariance
    of all rows by one pseudo-row, ``(n_c S_c + R) / (n_c + 1)``.
    """
    discrete = [v for v in variables if mask[v]]
    continuous = [v for v in variables if not mask[v]]
    if continuous:
        overall = np.atleast_2d(np.cov(x[:, continuous], rowvar=False, ddof=0))
    total = 0.0
    for rows in _cells(x, discrete):
        total += len(rows) * math.log(len(rows) / len(x))
        if continuous:
            values = x[np.ix_(rows, continuous)]
            mean = values.mean(axis=0)
            own = (values - mean).T @ (values - mean)  # n_c times the cell's covariance
            cov = (own + overall) / (len(rows) + 1)
            total += multivariate_normal(mean, cov).logpdf(values).sum()
    return total


def _with_rare_category(x: np.ndarray) -> np.ndarray:
    """Give the first two rows a fourth category of column 0, too few rows for its own
    two-variable covariance."""
    x = x.copy()
    x[:2, 0] = 7.0
    return x


def _embed(x: np.ndarray, mask: np.ndarray, variables: list[int]) -> np.ndarray:
    """The indicator columns of each categorical variable but its last category; others as is."""
    columns = []
    for v in variables:
        if mask[v]:
            values = np.unique(x[:, v])
            columns += [(x[:, v] == c).astype(float) for c in values[:-1]]
        else:
            columns.append(x[:, v])
    return np.column_stack(columns) if columns else np.empty((len(x), 0))


def _residuals(target: np.ndarray, regressors: np.ndarray) -> np.ndarray:
    """Least-squares residuals of ``target`` on ``regressors`` and an intercept."""
    design = np.column_stack([np.ones(len(target)), regressors])
    return target - design @ np.linalg.lstsq(design, target, rcond=None)[0]


def _log_det_mle(residuals: np.ndarray) -> float:
    return float(np.linalg.slogdet(residuals.T @ residuals / len(residuals))[1])


@pytest.mark.parametrize("rare", [False, True], ids=["common categories", "a two-row category"])
@pytest.mark.parametrize("variables", [[0], [1], [0, 2], [1, 3], [0, 1], [0, 1, 2, 3], []])
def test_cg_loglik_is_a_gaussian_per_joint_category(variables, rare):
    x, mask = _mixed()
    x = _with_rare_category(x) if rare else x
    assert ConditionalGaussian(x, mask).loglik(variables) == pytest.approx(
        _cg_loglik(x, mask, variables), rel=1e-12, abs=1e-9
    )


@pytest.mark.parametrize("rare", [False, True], ids=["common categories", "a two-row category"])
def test_cg_sets_fit_together_equal_sets_fit_alone(rare):
    """A batch mixes sets of every size and categorical part; each value equals the lone fit."""
    x, mask = _mixed()
    x = _with_rare_category(x) if rare else x
    sets = [[0], [1], [0, 2], [1, 3], [0, 1], [0, 1, 2, 3], [], [0, 1, 3], [2, 3], [1, 2, 3]]
    together = ConditionalGaussian(x, mask).loglik_many(sets)
    assert together == [ConditionalGaussian(x, mask).loglik(s) for s in sets]
    scorer, alone = CGScore(x, mask), CGScore(x, mask)
    parent_sets = [[], [0], [2], [0, 2], [1], [0, 1, 2]]
    assert scorer.score_many(3, parent_sets) == [alone.score(3, pa) for pa in parent_sets]


def test_cg_covers_a_cell_too_small_for_its_own_covariance():
    """The two-row category's covariance of two variables is singular on its own; shrunk toward
    all rows it is not, so the likelihood is finite."""
    x, mask = _mixed()
    assert np.isfinite(ConditionalGaussian(_with_rare_category(x), mask).loglik([0, 1, 3]))


def test_cg_rejects_collinear_continuous_columns():
    x, mask = _mixed()
    x[:, 3] = 2.0 * x[:, 1] + 1.0
    with pytest.raises(ValueError, match="collinear"):
        ConditionalGaussian(x, mask).loglik([1, 3])


def test_cg_counts_parameters_for_every_joint_category():
    x, mask = _mixed()
    model = ConditionalGaussian(x, mask)
    assert model.dof([]) == 0
    assert model.dof([0, 2]) == 3 * 2 - 1  # the cell weights
    assert model.dof([1, 3]) == 2 + 3  # two means, three covariance entries
    assert model.dof([0, 1, 2]) == 6 * (1 + 1 + 1) - 1  # mean, variance, weight per category


def test_cg_score_is_the_family_likelihood_over_the_parents_likelihood():
    x, mask = _mixed()
    score = CGScore(x, mask, lambda_value=2.0)
    for node, parents in [(3, (1, 2)), (0, (1,)), (2, ()), (1, (0, 2, 3))]:
        family = [*parents, node]
        loglik = _cg_loglik(x, mask, family) - _cg_loglik(x, mask, list(parents))
        dof = score._cg.dof(family) - score._cg.dof(parents)
        expected = -2.0 * loglik + 2.0 * math.log(len(x)) * dof
        assert score.score(node, parents) == pytest.approx(expected, rel=1e-10)


def test_cg_score_of_a_conditional_gaussian_network_is_close_to_its_bic():
    """On the DAG ``a -> x -> y <- b``, where no categorical variable has a continuous parent, the
    total score is close to ``-2`` times the maximized log-likelihood of each variable's own model,
    plus the penalty: category frequencies for ``a`` and ``b``, a regression per category of ``a``
    for ``x``, and one per category of ``b`` for ``y`` on ``x``. The pseudo-row moves each
    category, of more than 100 rows here, off its maximum by little."""
    x, mask = _mixed()
    n = len(x)
    loglik, dof = 0.0, 0
    for column, k in ((0, 3), (2, 2)):
        counts = np.unique(x[:, column], return_counts=True)[1]
        loglik += float(np.sum(counts * np.log(counts / n)))
        dof += k - 1
    for target, regressors, parent in ((1, [], 0), (3, [1], 2)):
        for rows in _cells(x, [parent]):
            resid = _residuals(x[rows, target], x[np.ix_(rows, regressors)])
            variance = resid @ resid / len(rows)
            loglik += -0.5 * len(rows) * (math.log(2 * math.pi * variance) + 1)
            dof += len(regressors) + 2  # intercept, slopes, variance
    score = CGScore(x, mask)
    total = sum(score.score(v, p) for v, p in [(0, ()), (1, (0,)), (2, ()), (3, (1, 2))])
    assert total == pytest.approx(-2.0 * loglik + math.log(n) * dof, abs=0.1)


@pytest.mark.parametrize("rare", [False, True], ids=["common categories", "a two-row category"])
@pytest.mark.parametrize("make", [CGScore, DGScore])
def test_equivalent_graphs_score_the_same(make, rare):
    """``a -> x -> y`` and ``a <- x <- y`` differ by reversing covered edges."""
    x, mask = _mixed()
    score = make(_with_rare_category(x) if rare else x, mask)
    forward = score.score(0, ()) + score.score(1, (0,)) + score.score(3, (1,))
    backward = score.score(3, ()) + score.score(1, (3,)) + score.score(0, (1,))
    assert forward == pytest.approx(backward, rel=1e-12)


def test_dg_score_equals_the_bic_on_continuous_columns():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 4))
    x[:, 1] += x[:, 0]
    x[:, 3] += x[:, 1] - x[:, 2]
    bic, dg = BICScore(x, lambda_value=1.5), DGScore(x, lambda_value=1.5)
    for node, parents in [(1, ()), (1, (0,)), (3, (1, 2)), (3, (0, 1, 2)), (0, (3,))]:
        assert dg.score(node, parents) == bic.score(node, parents)


@pytest.mark.parametrize(
    ("search", "bic"),
    [
        (ges, "local_score_BIC"),
        (hc, "local_score_BIC_from_cov"),
        (boss, "local_score_BIC_from_cov"),
        (grasp, "local_score_BIC_from_cov"),
    ],
)
def test_a_dg_search_on_continuous_columns_finds_the_bic_graph(search, bic):
    x = sample_scm(d=8, n=1000, seed=0, density=2.0).data
    dg_graph, dg_score = search(x, score_func="local_score_degen_gauss")
    bic_graph, bic_score = search(x, score_func=bic)
    assert np.array_equal(dg_graph.to_numpy(), bic_graph.to_numpy())
    assert dg_score == bic_score


def test_exact_search_with_dg_on_continuous_columns_finds_the_bic_equivalence_class():
    x = sample_scm(d=5, n=1000, seed=0, density=2.0).data
    dg = to_cpdag(exact_search(x, score_func="local_score_degen_gauss"))
    assert np.array_equal(dg.to_numpy(), to_cpdag(exact_search(x)).to_numpy())


def test_full_score_batches_are_the_local_scores():
    x, mask = _mixed()
    score = CGScore(x, mask)
    moves = [("add", 1), ("add", 2), ("remove", 0)]
    expected = [score.score(3, [0, 1]), score.score(3, [0, 2]), score.score(3, [])]
    assert FullScoreDelta(score).score_many_with_base(3, [0], moves) == expected


@pytest.mark.parametrize(("node", "parents"), [(0, ()), (0, (1,)), (0, (1, 2)), (3, (0, 2))])
def test_dg_score_is_a_regression_on_indicator_columns(node, parents):
    x, mask = _mixed()
    n = len(x)
    target = _embed(x, mask, [node])
    resid = _residuals(target, _embed(x, mask, list(parents)))
    # The score reads the ddof-1 covariance, a factor n / (n - 1) on each of the node's columns.
    log_det = _log_det_mle(resid) + target.shape[1] * math.log(n / (n - 1))
    penalty = math.log(n) * target.shape[1] * _embed(x, mask, list(parents)).shape[1]
    assert DGScore(x, mask).score(node, parents) == pytest.approx(n * log_det + penalty, rel=1e-10)


@pytest.mark.parametrize(
    ("a", "b", "given"), [(0, 3, (1,)), (0, 2, ()), (1, 2, (0, 3)), (0, 2, (3,))]
)
def test_dg_test_is_the_likelihood_ratio_of_linear_regressions(a, b, given):
    x, mask = _mixed()
    conditioning = _embed(x, mask, list(given))
    ra = _residuals(_embed(x, mask, [a]), conditioning)
    rb = _residuals(_embed(x, mask, [b]), conditioning)
    statistic = len(x) * (
        _log_det_mle(ra) + _log_det_mle(rb) - _log_det_mle(np.column_stack([ra, rb]))
    )
    dof = ra.shape[1] * rb.shape[1]
    assert DGTest(x, mask)(a, b, given) == pytest.approx(chi2.sf(statistic, dof), rel=1e-9)


@pytest.mark.parametrize("make", [DGTest, CGTest])
def test_on_continuous_columns_a_test_is_the_partial_correlation_ratio(make):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((300, 4))
    x[:, 3] += 0.2 * x[:, 0] + x[:, 1] - x[:, 2]
    for a, b, given in [(0, 2, ()), (0, 3, (1,)), (0, 3, (1, 2))]:
        r = np.corrcoef(
            _residuals(x[:, a], x[:, list(given)]), _residuals(x[:, b], x[:, list(given)])
        )
        statistic = -len(x) * math.log(1 - r[0, 1] ** 2)
        assert make(x)(a, b, given) == pytest.approx(chi2.sf(statistic, 1), rel=1e-9)


def test_cg_test_on_categorical_columns_is_the_g_square_test():
    """With every joint category occupied, both count the same degrees of freedom."""
    rng = np.random.default_rng(3)
    x = rng.integers(0, 3, (3000, 3)).astype(np.float64)
    x[:, 2] = np.where(rng.random(3000) < 0.5, x[:, 0], x[:, 2])
    cg, gsq = CGTest(x, np.ones(3, bool)), ChiSquare(x, g_square=True)
    for a, b, given in [(0, 1, ()), (0, 1, (2,)), (1, 2, (0,)), (0, 2, ())]:
        assert cg(a, b, given) == pytest.approx(gsq(a, b, given), rel=1e-9, abs=1e-300)


@pytest.mark.parametrize("make", [DGTest, CGTest])
def test_a_test_is_symmetric_in_its_pair_and_its_conditioning_order(make):
    x, mask = _mixed()
    test = make(x, mask)
    assert test(0, 3, (1, 2)) == test(3, 0, (2, 1))


def test_dg_test_of_a_variable_its_conditioning_set_determines_is_one():
    x, mask = _mixed()
    x = np.column_stack([x, x[:, 0]])  # column 4 copies the categorical column 0
    test = DGTest(x, np.append(mask, True))
    assert test(4, 3, (0,)) == 1.0


def _null(case: str, rng: np.random.Generator, n: int):
    """Data where ``x`` is independent of ``y`` given ``S``, from a conditional Gaussian network.

    Returns the data, its categorical mask, and the query ``(x, y, S)``.
    """
    if case == "continuous pair given a category":
        z = rng.integers(0, 3, n)
        a = np.array([0.0, 1.5, -1.0])[z] + rng.standard_normal(n)
        b = np.array([0.5, -1.0, 1.0])[z] + np.array([1.0, 2.0, 0.5])[z] * rng.standard_normal(n)
        return np.column_stack([a, b, z]), np.array([False, False, True]), (0, 1, (2,))
    if case == "category and continuous given continuous":
        x = rng.integers(0, 3, n)
        z = np.array([0.0, 1.0, -1.0])[x] + rng.standard_normal(n)
        y = 0.8 * z + rng.standard_normal(n)
        return np.column_stack([x, y, z]), np.array([True, False, False]), (0, 1, (2,))
    if case == "two categories given continuous":
        x = rng.integers(0, 2, n)
        y = rng.integers(0, 3, n)
        z = np.array([0.0, 1.0])[x] + rng.standard_normal(n)
        return np.column_stack([x, y, z]), np.array([True, True, False]), (0, 1, (2,))
    if case == "category and continuous":
        x = rng.integers(0, 4, n)
        return np.column_stack([x, rng.standard_normal(n)]), np.array([True, False]), (0, 1, ())
    if case == "continuous pair given a category of two rows":
        z = np.r_[np.full(2, 3), rng.integers(0, 3, n - 2)]
        a = np.array([0.0, 1.5, -1.0, 0.5])[z] + rng.standard_normal(n)
        b = np.array([0.5, -1.0, 1.0, 0.0])[z] + rng.standard_normal(n)
        return np.column_stack([a, b, z]), np.array([False, False, True]), (0, 1, (2,))
    raise KeyError(case)


@pytest.mark.parametrize("make", [DGTest, CGTest])
@pytest.mark.parametrize(
    "case",
    [
        "continuous pair given a category",
        "category and continuous given continuous",
        "two categories given continuous",
        "category and continuous",
        "continuous pair given a category of two rows",
    ],
)
def test_p_values_are_uniform_when_the_independence_holds(make, case):
    """200 simulated datasets of 500 rows: about 5% of p-values fall below 0.05, and their
    distribution is close to uniform."""
    rng = np.random.default_rng(20261012)
    p_values = []
    for _ in range(200):
        x, mask, (a, b, given) = _null(case, rng, 500)
        p_values.append(make(x, mask)(a, b, given))
    p_values = np.array(p_values)
    assert 0.02 <= np.mean(p_values < 0.05) <= 0.09
    assert kstest(p_values, "uniform").statistic < 0.1


@pytest.mark.parametrize("make", [DGTest, CGTest])
def test_a_test_finds_a_dependence(make):
    x, mask = _mixed(n=1000)
    test = make(x, mask)
    assert test(0, 1) < 1e-10  # a -> x
    assert test(2, 3, (1,)) < 1e-6  # b -> y given x
    assert test(0, 3, (1,)) > 0.01  # a and y separated by x


def test_only_the_mixed_tests_take_a_categorical_mask():
    x, mask = _mixed()
    assert isinstance(make_indep_test("degen_gauss", x, categorical=mask), DGTest)
    with pytest.raises(ValueError, match="reads no column types"):
        make_indep_test("fisherz", x, categorical=mask)


def test_a_mask_marks_each_column():
    assert column_mask(None, 3).tolist() == [False, False, False]
    with pytest.raises(ValueError, match="each of the 3 columns"):
        column_mask([True, False], 3)


def test_cg_rejects_a_constant_continuous_column():
    x, mask = _mixed()
    x[:, 1] = 2.0
    with pytest.raises(ValueError, match="continuous column 1 is constant"):
        ConditionalGaussian(x, mask)
