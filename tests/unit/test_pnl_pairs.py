"""PNL's batched pairs: the blocked HSIC equals the reference test, and rows equal single calls."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from andrey.core.independence import hsic_statistic_test

_pnl = importlib.import_module("andrey.irregular.pnl")


def _table(n: int) -> np.ndarray:
    """A cause, its post-nonlinear effect, a column with ties, and a copy of the cause."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = np.tanh(x + 0.4 * rng.laplace(size=n)) ** 3
    return np.column_stack([x, y, np.round(rng.normal(size=n)), x])


@pytest.mark.parametrize("n", [150, 600, 1300])  # each kernel width of the HSIC test
def test_blocked_hsic_equals_the_reference_test(n):
    X = _table(n)
    cause, effect = _pnl._column(X[:, 0]), _pnl._column(X[:, 1])
    disturbance = _pnl._fit_disturbance(cause.phi, effect.outer, effect.outer_deriv)
    buffers = _pnl._Buffers.for_rows(n)
    kxc = _pnl._centered_kernel(cause.standardized, cause.width, buffers.cause, buffers.block)
    blocked = _pnl._hsic(kxc, np.trace(kxc), np.sum(kxc**2), disturbance, buffers)
    assert blocked == hsic_statistic_test(_pnl._gaussianize(X[:, 0]), disturbance)


def test_pair_rows_equal_single_calls():
    X = _table(120)
    pairs = [(0, 1), (1, 0), (2, 3), (3, 3), (0, 3)]
    rows = _pnl.pnl_pairs(X, pairs)
    assert [tuple(row) for row in rows] == [_pnl.pnl(X[:, i], X[:, j]) for i, j in pairs]
    assert np.array_equal(_pnl.pnl_pairs(X, pairs, num_workers=3), rows)


def test_pairs_default_to_every_column_pair_and_are_checked():
    X = _table(50)
    rows = _pnl.pnl_pairs(X)
    assert np.array_equal(rows[1], _pnl.pnl_pairs(X, [(0, 2)])[0])
    assert rows.shape == (6, 4)
    assert _pnl.pnl_pairs(X, []).shape == (0, 4)
    with pytest.raises(ValueError, match="out of range"):
        _pnl.pnl_pairs(X, [(0, 4)])
    with pytest.raises(ValueError, match="table"):
        _pnl.pnl_pairs(X[:, 0])
