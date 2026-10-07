"""LongitudinalLiNGAM respects lag boundaries and coefficient orientation."""

from __future__ import annotations

import numpy as np

from andrey.temporal.longitudinal import LongitudinalLiNGAM


def _lag1_at_t1(panel: list[np.ndarray]) -> np.ndarray:
    """The lag-1 adjacency block at occasion t=1, fit with n_lags=2."""
    return LongitudinalLiNGAM(n_lags=2).fit(panel).adjacency_matrices_[1, 1].copy()


def test_early_lag_coefficients_do_not_leak_future_occasions():
    # At occasion t=1 with n_lags=2 the lag-2 predictor X(t-2) = X(-1) does not exist. It must be
    # excluded from the regression, not wrapped (negative indexing) into the last/future occasion.
    # If it leaked, the retained lag-1 coefficient -- jointly regressed with that future
    # predictor -- would depend on future data.
    rng = np.random.default_rng(0)
    panel = [rng.uniform(-1, 1, size=(400, 2)) for _ in range(4)]
    baseline = _lag1_at_t1(panel)

    perturbed = [x.copy() for x in panel]
    perturbed[-1] += rng.uniform(5, 6, size=(400, 2))  # change only the future (last) occasion
    assert np.allclose(_lag1_at_t1(perturbed), baseline)  # the t=1 lag-1 coeff is invariant to it


def test_unavailable_lag_blocks_are_nan_not_zero():
    # A lag-k block reaching before the panel start is uncomputable and must surface as NaN, not a
    # finite 0.0 that reads as a real "no effect" claim. n_lags=3 exercises t=1, where lag-2 and
    # lag-3 are both unavailable, so both must be NaN, not only the lag == t block.
    rng = np.random.default_rng(1)
    panel = [rng.uniform(-1, 1, size=(300, 2)) for _ in range(4)]
    adj = LongitudinalLiNGAM(n_lags=3).fit(panel).adjacency_matrices_
    assert np.isnan(adj[0]).all()  # occasion 0 has no predecessors
    assert np.isfinite(adj[1, 0]).all()  # instantaneous at t=1
    assert np.isfinite(adj[1, 1]).all()  # lag-1 at t=1 is available (reaches occasion 0)
    assert np.isnan(adj[1, 2]).all()  # lag-2 at t=1 reaches before the panel start
    assert np.isnan(adj[1, 3]).all()  # lag-3 at t=1 too -- must be NaN, not a finite 0.0
    assert np.isfinite(adj[2, 2]).all()  # lag-2 at t=2 is available
    assert np.isnan(adj[2, 3]).all()  # lag-3 at t=2 reaches before the panel start
    assert np.isfinite(adj[3, 3]).all()  # lag-3 at t=3 is available (reaches occasion 0)
