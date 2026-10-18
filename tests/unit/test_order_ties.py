"""Exact ties break the same way under every column order.

Each test fits data built so that a method meets an exact tie, refits under reordered columns,
maps the result back to the original labels, and expects the original fit. Ranking the columns by
position instead changes the result under at least one order, so the orders exercise the
tie-break.
"""

from __future__ import annotations

import importlib
from itertools import permutations

import numpy as np

from andrey.core.score import BICScore
from andrey.search.ges import ges
from andrey.search.xges import xges

_ORDERS = [np.array(order) for order in permutations(range(3))]


def _by_position(x: np.ndarray, categorical: np.ndarray | None = None) -> np.ndarray:
    return np.arange(x.shape[1])


def _swap_symmetric(seed: int, n: int = 200) -> np.ndarray:
    """Columns 0 and 1 are close copies of one signal and column 2 a noisy one.

    Every row also appears with columns 0 and 1 swapped. The values are integers, so sums over the
    rows are exact and a fit of column 2 on column 0 equals its fit on column 1.
    """
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(n)
    z = signal[:, None] + rng.standard_normal((n, 3)) * np.array([0.3, 0.3, 3.0])
    z = np.round(10 * z)
    return np.vstack([z, z[:, [1, 0, 2]]])


def _mapped(fit, x: np.ndarray, order: np.ndarray) -> np.ndarray:
    """``fit`` on the columns of ``x`` in ``order``, mapped back to the original labels."""
    back = np.argsort(order)
    return fit(x[:, order])[np.ix_(back, back)]


def _assert_order_free(fit, x: np.ndarray, module: str, monkeypatch) -> None:
    want = fit(x)
    for order in _ORDERS:
        np.testing.assert_array_equal(_mapped(fit, x, order), want)
    monkeypatch.setattr(importlib.import_module(module), "tie_ranks", _by_position)
    by_position = fit(x)
    assert any((_mapped(fit, x, order) != by_position).any() for order in _ORDERS)


def test_ges_ties(monkeypatch) -> None:
    """GES adds the 0 - 1 edge, then column 2's edges to 0 and to 1 tie; it adds one of them."""
    x = _swap_symmetric(0)
    score = BICScore(x)
    assert score.score(2, [0]) == score.score(2, [1])
    _assert_order_free(lambda data: ges(data)[0].to_numpy(), x, "andrey.search.ges", monkeypatch)


def test_xges_ties(monkeypatch) -> None:
    """XGES meets the tie GES meets: column 2's edges to 0 and to 1."""
    x = _swap_symmetric(0)
    for rescan in (False, True):
        _assert_order_free(
            lambda data: xges(data, _rescan=rescan)[0].to_numpy(),
            x,
            "andrey.search.xges",
            monkeypatch,
        )
        monkeypatch.undo()
