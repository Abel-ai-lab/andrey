"""The Stein estimators: closed forms on two points, a known density, and their invariances."""

from __future__ import annotations

import math

import numpy as np
import pytest

from andrey.fcm import stein


def test_two_points_give_the_closed_form():
    """Two rows ``a`` and ``-a`` of one column, whose every quantity has a closed form.

    Their distance ``2a`` is the bandwidth, so the kernel matrix has ``k = exp(-1/2)`` off its
    diagonal. ``<grad, K>`` is ``(k / 2a, -k / 2a)``, an eigenvector of ``K + eta I`` of eigenvalue
    ``1 + eta - k``, and ``<grad^2, K>`` is ``-1 / 4a^2`` in both rows, an eigenvector of
    eigenvalue ``1 + eta + k``.
    """
    a, eta = 1.5, 0.25
    X = np.array([[a], [-a]])
    k = math.exp(-0.5)
    g = k / (2 * a * (1 + eta - k))
    np.testing.assert_allclose(stein.score(X, eta=eta), [[-g], [g]], rtol=1e-12)
    j = -(g**2) - 1 / (4 * a**2 * (1 + eta + k))
    np.testing.assert_allclose(stein.jacobian_diagonal(X, eta=eta), [[j], [j]], rtol=1e-12)


@pytest.fixture(scope="module")
def gaussian():
    """1,000 rows of two independent normal columns of standard deviations 1 and 0.5."""
    sd = np.array([1.0, 0.5])
    return np.random.default_rng(0).standard_normal((1000, 2)) * sd, sd


def test_the_score_of_a_gaussian_is_linear(gaussian):
    """A centered Gaussian has the score ``-x / sd^2``; the estimate follows it over the bulk."""
    X, sd = gaussian
    truth = -X / sd**2
    bulk = (np.abs(X / sd) < 2).all(axis=1)
    estimate = stein.score(X)
    for j in range(2):
        assert np.corrcoef(estimate[bulk, j], truth[bulk, j])[0, 1] > 0.9
        slope = np.polyfit(X[bulk, j], estimate[bulk, j], 1)[0]
        assert slope == pytest.approx(-1 / sd[j] ** 2, rel=0.15)


def test_the_jacobian_diagonal_of_a_gaussian_is_constant(gaussian):
    """A Gaussian's log density has the second derivatives ``-1 / sd^2`` at every point."""
    X, sd = gaussian
    bulk = (np.abs(X / sd) < 2).all(axis=1)
    estimate = stein.jacobian_diagonal(X)[bulk]
    np.testing.assert_allclose(np.median(estimate, axis=0), -1 / sd**2, rtol=0.2)


def test_the_estimates_follow_the_rows_and_columns(gaussian):
    """A shift changes no estimate, and reordered rows or columns reorder the estimates."""
    X = gaussian[0][:200]
    for estimator in (stein.score, stein.jacobian_diagonal):
        base = estimator(X)
        np.testing.assert_allclose(estimator(X + [3.0, -7.0]), base, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(estimator(X[:, ::-1]), base[:, ::-1], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(estimator(X[::-1]), base[::-1], rtol=1e-6, atol=1e-8)


def test_a_rescaled_sample_rescales_the_estimates(gaussian):
    """In units ``c`` times larger the score is ``1 / c`` and its Jacobian ``1 / c^2`` of before."""
    X = gaussian[0][:200]
    c = 8.0
    np.testing.assert_allclose(stein.score(c * X), stein.score(X) / c, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        stein.jacobian_diagonal(c * X), stein.jacobian_diagonal(X) / c**2, rtol=1e-6, atol=1e-8
    )


def test_equal_rows_leave_no_bandwidth():
    X = np.zeros((6, 2))
    X[0] = 1.0
    with pytest.raises(ValueError, match="score_matching: the median distance"):
        stein.jacobian_diagonal(X, method="score_matching")
