"""BIC-score primitive tests: closed-form values, class-vs-function agreement, determinism.

These pin ``ANDREY_DEVICE=numpy``, numpy being the exact covariance oracle. The independently-known
values are derived from the linear-Gaussian BIC closed form and the ddof=1 sample covariance, never
from the implementation's internals; the inline baseline pins current outputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from andrey.core.score import BICScore, Score, local_score_bic


@pytest.fixture(autouse=True)
def force_numpy(monkeypatch):
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


@pytest.fixture
def data():
    # 60 observations of 6 variables; enough columns for a 5-parent set.
    rng = np.random.default_rng(7)
    return rng.standard_normal((60, 6))


@pytest.fixture
def cov(data):
    return np.cov(data, rowvar=False)  # ddof=1 sample covariance


# ---- independently-known values -----------------------------------------------------------------


def test_zero_parents_equals_n_log_variance(data, cov):
    # With no parents the residual variance is the target's own variance, so the score reduces to
    # ``n * log(var_i)`` with ``var_i`` the ddof=1 sample variance. Verified against a direct
    # variance, not the covariance the score consumes.
    n = data.shape[0]
    for node in range(data.shape[1]):
        var_i = data[:, node].var(ddof=1)
        assert local_score_bic(cov, n, node, []) == pytest.approx(n * np.log(var_i), abs=1e-12)


def test_zero_parents_ignores_lambda(cov):
    # The complexity term scales with the parent count, so a zero-parent score is independent of the
    # penalty discount.
    n = 60
    assert local_score_bic(cov, n, 0, [], lambda_value=1.0) == local_score_bic(
        cov, n, 0, [], lambda_value=5.0
    )


def test_single_parent_closed_form(cov):
    # One parent: residual = var_i - cov_ip^2 / var_p, score = n*log(residual) + log(n)*lambda.
    # Computed here from covariance entries by hand, independent of the branch under test.
    n = 60
    node, parent = 2, 0
    residual = cov[node, node] - cov[node, parent] ** 2 / cov[parent, parent]
    expected = n * np.log(residual) + np.log(n) * 1.0
    assert local_score_bic(cov, n, node, [parent], lambda_value=1.0) == pytest.approx(
        expected, abs=1e-9
    )


def test_two_parent_closed_form(cov):
    # Two parents: residual = var_i - cov[i,P] @ inv(cov[P,P]) @ cov[P,i], solved here directly.
    n = 60
    node, pa = 3, [0, 1]
    yX = cov[np.ix_([node], pa)]
    XX = cov[np.ix_(pa, pa)]
    residual = cov[node, node] - (yX @ np.linalg.solve(XX, yX.T)).item()
    expected = n * np.log(residual) + np.log(n) * 2 * 1.0
    assert local_score_bic(cov, n, node, pa, lambda_value=1.0) == pytest.approx(expected, abs=1e-9)


# ---- lambda / penalty term ----------------------------------------------------------------------


@pytest.mark.parametrize("k, pa", [(1, [0]), (2, [0, 1]), (3, [0, 1, 2])])
def test_lambda_scales_only_the_penalty(cov, k, pa):
    # Doubling lambda adds exactly ``log(n) * k`` (the extra penalty), leaving the fit term intact.
    n = 60
    node = 4
    s1 = local_score_bic(cov, n, node, pa, lambda_value=1.0)
    s2 = local_score_bic(cov, n, node, pa, lambda_value=2.0)
    assert s2 - s1 == pytest.approx(np.log(n) * k, abs=1e-9)


# ---- class-vs-function agreement ----------------------------------------------------------------


@pytest.mark.parametrize(
    "node, pa",
    [(0, []), (2, [1]), (3, [0, 1]), (4, [0, 1, 2]), (5, [0, 1, 2, 3, 4])],
)
def test_class_matches_function(data, cov, node, pa):
    n = data.shape[0]
    scorer = BICScore(data)
    expected = local_score_bic(cov, n, node, pa, lambda_value=1.0)
    assert scorer.score(node, pa) == pytest.approx(expected, abs=0.0, rel=0.0)


def test_class_covariance_is_ddof_one(data):
    scorer = BICScore(data)
    assert np.array_equal(scorer.cov, np.cov(data, rowvar=False))
    assert scorer.n == data.shape[0]


def test_class_respects_custom_lambda(data, cov):
    n = data.shape[0]
    scorer = BICScore(data, lambda_value=2.0)
    assert scorer.score(3, [0, 1]) == pytest.approx(
        local_score_bic(cov, n, 3, [0, 1], lambda_value=2.0), abs=0.0
    )


def test_class_is_score_protocol(data):
    assert isinstance(BICScore(data), Score)


# ---- caching contract ---------------------------------------------------------------------------


def test_cache_order_invariant(data):
    # The cache key sorts the parent tuple, so parent-set order does not change the score.
    scorer = BICScore(data)
    assert scorer.score(4, [0, 1, 2]) == scorer.score(4, [2, 0, 1])
    assert scorer.score(3, [0, 1]) == scorer.score(3, [1, 0])


def test_score_many_matches_per_set(data):
    scorer = BICScore(data)
    parent_sets = [[], [0], [1, 2], [0, 1, 2, 3]]
    assert scorer.score_many(4, parent_sets) == [scorer.score(4, pa) for pa in parent_sets]


def test_score_many_preserves_request_order(data):
    scorer = BICScore(data)
    ordered = scorer.score_many(5, [[0], [0, 1], []])
    assert ordered[0] == scorer.score(5, [0])
    assert ordered[1] == scorer.score(5, [0, 1])
    assert ordered[2] == scorer.score(5, [])


# ---- determinism --------------------------------------------------------------------------------


def test_repeated_calls_are_bit_identical(data, cov):
    n = data.shape[0]
    first = local_score_bic(cov, n, 4, [0, 1, 2], lambda_value=2.0)
    for _ in range(5):
        assert local_score_bic(cov, n, 4, [0, 1, 2], lambda_value=2.0) == first
    scorer = BICScore(data)
    cached = scorer.score(4, [0, 1, 2])
    assert BICScore(data).score(4, [0, 1, 2]) == cached


def test_precomputed_log_n_matches_default(cov):
    # Passing ``log_n`` explicitly must equal letting the function derive it from ``n``.
    n = 60
    for node, pa in [(0, []), (2, [1]), (3, [0, 1]), (5, [0, 1, 2, 3, 4])]:
        assert local_score_bic(cov, n, node, pa, log_n=float(np.log(n))) == local_score_bic(
            cov, n, node, pa
        )


# ---- fail-fast on out-of-range indices ----------------------------------------------------------


def test_out_of_range_node_raises(cov):
    with pytest.raises(IndexError):
        local_score_bic(cov, 60, 99, [])


def test_out_of_range_parent_raises(cov):
    with pytest.raises(IndexError):
        local_score_bic(cov, 60, 0, [99])


# ---- inline regression baseline ----------------------------------------------------------

# Captured from the current implementation at write time and cross-checked against the
# linear-Gaussian BIC closed form. Data: default_rng(7).standard_normal((60, 6)), lambda=1.
_BASELINE = {
    (0, ()): -10.474569977701616,
    (2, (1,)): -8.058952068406125,
    (3, (0, 1)): -0.8322322832955678,
    (4, (0, 1, 2)): -7.228572333747046,
    (5, (0, 1, 2, 3, 4)): -13.048207112823583,
}


def test_baseline_values(data, cov):
    n = data.shape[0]
    for (node, pa), expected in _BASELINE.items():
        assert local_score_bic(cov, n, node, list(pa), lambda_value=1.0) == pytest.approx(
            expected, abs=1e-12
        )
