"""DYNOTEARS: the stacked loss, the two thresholds, the lags, the knowledge, and input checks."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.data import temporal
from andrey.search.continuous import least_squares
from andrey.temporal.dynotears import lagged_covariance, lagged_design

# W[cause, effect]: x0 -> x1 -> x2 within a step; A_1: every series on its own past, x1 -> x3.
W = np.zeros((4, 4))
W[0, 1], W[1, 2] = 0.9, -0.8
LAG_1 = np.diag([0.4, 0.3, 0.4, 0.3])
LAG_1[1, 3] = 0.5


@pytest.fixture(scope="module")
def series() -> np.ndarray:
    innovations = np.random.default_rng(0).standard_normal((800, 4))
    ib_inv = np.linalg.inv(np.eye(4) - W.T)
    return temporal.generate(ib_inv, LAG_1.T[None], None, innovations, burn_in=200)


def fit(data, **kwargs) -> andrey.StructureOutput:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return andrey.dynotears(data, **kwargs)


def links(out: andrey.StructureOutput) -> set[tuple[int, int, int]]:
    """Every ``(cause, effect, lag)`` link of a fit, lag 0 included."""
    s = out.structure
    return {(i, j, lag) for lag in s.lags for i, j, _ in s.lag(lag).oriented_edges()}


@pytest.fixture(scope="module")
def default_fit(series) -> andrey.StructureOutput:
    return fit(series)


def test_the_stacked_loss_is_the_mean_squared_residual_of_the_lagged_regression(series):
    """With lag blocks stacked below ``W``, the loss is ``0.5 / n * ||X - X W - Y A||**2`` on the
    centered columns, and its gradient is the derivative of that residual."""
    design = lagged_design(series, 2)
    centered = design - design.mean(axis=0)
    weights = np.random.default_rng(1).normal(0.0, 0.3, (12, 4))
    value, grad = least_squares(lagged_covariance(series, 2), weights)
    residual = centered[:, :4] - centered @ weights
    n = len(design)
    assert value == pytest.approx(0.5 / n * np.sum(residual**2), rel=1e-12)
    np.testing.assert_allclose(grad, -centered.T @ residual / n, rtol=1e-10, atol=1e-12)


def test_each_threshold_acts_on_its_own_block(series, default_fit):
    same_step = {link for link in links(default_fit) if link[2] == 0}
    lagged = links(default_fit) - same_step
    assert same_step == {(0, 1, 0), (1, 2, 0)}
    assert lagged == {(i, i, 1) for i in range(4)} | {(1, 3, 1)}
    assert links(fit(series, w_threshold=10.0)) == lagged
    assert links(fit(series, lag_threshold=10.0)) == same_step


def test_kept_weights_reach_their_threshold_and_the_rest_are_zero(series):
    out = fit(series, w_threshold=0.2, lag_threshold=0.05)
    weights = out.structure.lag_weights
    kept = np.zeros(weights.shape, dtype=bool)
    for i, j, lag in links(out):
        kept[lag, i, j] = True
    assert (np.abs(weights[0][kept[0]]) >= 0.2).all()
    assert (np.abs(weights[1:][kept[1:]]) >= 0.05).all()
    assert not weights[~kept].any()


def test_a_larger_penalty_gives_no_larger_l1_norm(series):
    """Over the same constraint, a heavier ``l1`` penalty never raises the total ``|W|_1 + |A|_1``
    of the fit; without thresholds the weights are the fit's own."""
    norms = [
        np.abs(
            fit(series, lambda1=lam, w_threshold=0.0, lag_threshold=0.0).structure.lag_weights
        ).sum()
        for lam in (0.01, 0.05, 0.2, 0.8)
    ]
    assert all(later <= earlier for earlier, later in zip(norms, norms[1:], strict=False))
    assert norms[-1] < norms[0]


def test_a_zero_lag_threshold_keeps_no_lagged_link_of_weight_zero(series):
    out = fit(series, lambda1=0.2, lag_threshold=0.0)
    weights = out.structure.lag_weights
    lagged = [(i, j, lag) for i, j, lag in links(out) if lag > 0]
    assert lagged
    assert all(weights[lag, i, j] != 0 for i, j, lag in lagged)
    assert len(lagged) < 16


def test_tau_max_sets_the_lags_and_the_weights(series):
    out = fit(series, tau_max=3)
    assert out.structure.lags == (0, 1, 2, 3)
    assert out.structure.lag(0).kind == "dag"
    assert all(out.structure.lag(k).kind == "digraph" for k in (1, 2, 3))
    assert out.structure.lag_weights.shape == (4, 4, 4)
    # The planted process has one lag, so the later lags hold no link.
    assert {link for link in links(out) if link[2] > 1} == set()


def test_too_short_a_series_for_tau_max_raises(series):
    with pytest.raises(ValueError, match="tau_max"):
        fit(series[:4], tau_max=3)


def test_a_forbidden_link_is_absent_at_its_lag_and_a_forbidden_pair_at_every_lag(series):
    out = fit(series, knowledge=andrey.Knowledge(forbidden=[(1, 3, 1), (0, 1)]))
    assert not links(out) & {(1, 3, 1), (0, 1, 0), (0, 1, 1)}
    assert out.structure.lag_weights[1, 1, 3] == 0 and out.structure.lag_weights[0, 0, 1] == 0


def test_a_required_same_step_link_is_kept_on_independent_noise():
    noise = np.random.default_rng(2).standard_normal((300, 3))
    out = fit(noise, knowledge=andrey.Knowledge(required=[(2, 0, 0)]))
    assert (2, 0, 0) in links(out)


def test_a_required_lagged_link_is_kept_on_independent_noise():
    noise = np.random.default_rng(2).standard_normal((300, 3))
    out = fit(noise, knowledge=andrey.Knowledge(required=[(2, 0, 1)]))
    assert (2, 0, 1) in links(out)


def test_a_pair_left_out_of_allowed_has_no_link_at_any_lag(series):
    """``allowed`` lists the only pairs that may be linked; a series keeps its own past."""
    pairs = {frozenset((0, 1)), frozenset((1, 2))}
    out = fit(series, knowledge=andrey.Knowledge(allowed=[(0, 1), (1, 2)]))
    across = {frozenset((i, j)) for i, j, _ in links(out) if i != j}
    assert across
    assert across <= pairs
    assert (3, 3, 1) in links(out)


def test_a_lagged_two_cycle_is_kept_as_acyclicity_reads_the_same_step_only():
    """``x0(t-1) -> x2(t)`` and ``x2(t-1) -> x0(t)`` close no cycle in time, so both stay."""
    lag_1 = np.diag([0.3, 0.3, 0.3])
    lag_1[0, 2] = lag_1[2, 0] = 0.4
    innovations = np.random.default_rng(3).standard_normal((800, 3))
    cyclic = temporal.generate(np.eye(3), lag_1.T[None], None, innovations, burn_in=200)
    assert {(0, 2, 1), (2, 0, 1)} <= links(fit(cyclic))


def test_tiers_forbid_every_link_into_an_earlier_tier_but_a_series_keeps_its_own_past(series):
    out = fit(series, knowledge=andrey.Knowledge(tiers=[[3], [0, 1, 2]]))
    assert not {(i, j, lag) for i, j, lag in links(out) if j == 3 and i != 3}
    assert (3, 3, 1) in links(out)


@pytest.mark.parametrize(
    ("required", "message"), [((0, 1), "without a lag"), ((0, 1, 2), "at lag 2")]
)
def test_a_required_link_needs_a_lag_the_fit_learns(series, required, message):
    with pytest.raises(ValueError, match=message):
        fit(series, knowledge=andrey.Knowledge(required=[required]))


def test_a_missing_value_raises_and_names_the_method(series):
    data = series.copy()
    data[10, 2] = np.nan
    with pytest.raises(ValueError, match="dynotears"):
        fit(data)


def test_column_names_become_the_labels(series):
    frame = pd.DataFrame(series, columns=["a", "b", "c", "d"])
    out = fit(frame)
    assert out.structure.labels == ("a", "b", "c", "d")
    assert ("a", "b", "directed") in out.structure.lag(0).oriented_edges()
