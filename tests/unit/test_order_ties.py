"""Ties break the same way under every column order.

Each test fits data built so that a method meets a tie, refits under every column order, maps the
result back to the original labels, and expects the original fit. Where the tie is broken by
:func:`~andrey.search.hc.tie_ranks`, ranking the columns by position instead changes the result
under at least one order, so the orders exercise the tie-break.
"""

from __future__ import annotations

import importlib
from itertools import permutations

import numpy as np

from andrey.core.ci import FisherZ
from andrey.core.score import BICScore
from andrey.search.calm import moral_graph
from andrey.search.ges import ges
from andrey.search.xges import xges


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


def _swap_symmetric_sem(seed: int, n: int = 400) -> np.ndarray:
    """A random linear model on four columns; every row also appears with columns 0 and 1 swapped.

    The values are integers, so sums over the rows are exact.
    """
    rng = np.random.default_rng(seed)
    weights = np.triu(rng.uniform(-1, 1, (4, 4)) * (rng.random((4, 4)) < 0.6), 1)
    z = rng.standard_normal((n, 4))
    for j in range(4):
        z[:, j] += z @ weights[:, j]
    z = np.round(10 * z)
    return np.vstack([z, z[:, [1, 0, 2, 3]]])


def _strong_chain(seed: int, n: int = 2000) -> np.ndarray:
    """``0 -> 1 -> 2 -> 3`` and ``0 -> 3`` at weight 5; each row also appears with 0, 3 swapped."""
    z = np.random.default_rng(seed).standard_normal((n, 4))
    z[:, 1] += 5 * z[:, 0]
    z[:, 2] += 5 * z[:, 1]
    z[:, 3] += 5 * z[:, 0] + 5 * z[:, 2]
    z = np.round(10 * z)
    return np.vstack([z, z[:, [3, 1, 2, 0]]])


def _mapped(fit, x: np.ndarray, order: np.ndarray) -> np.ndarray:
    """``fit`` on the columns of ``x`` in ``order``, mapped back to the original labels."""
    back = np.argsort(order)
    return fit(x[:, order])[np.ix_(back, back)]


def _assert_order_free(fit, x: np.ndarray, module: str, monkeypatch) -> None:
    orders = [np.array(order) for order in permutations(range(x.shape[1]))]
    want = fit(x)
    for order in orders:
        np.testing.assert_array_equal(_mapped(fit, x, order), want)
    monkeypatch.setattr(importlib.import_module(module), "tie_ranks", _by_position)
    by_position = fit(x)
    assert any((_mapped(fit, x, order) != by_position).any() for order in orders)


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


def test_calm_moral_graph_ties(monkeypatch) -> None:
    """IAMB's first choice for column 2 ties between columns 0 and 1."""
    x = _swap_symmetric_sem(181)
    test = FisherZ(x)
    assert test.statistic_pvalue(2, 0, []) == test.statistic_pvalue(2, 1, [])
    _assert_order_free(moral_graph, x, "andrey.search.calm", monkeypatch)


def test_calm_moral_graph_zero_p_values() -> None:
    """IAMB chooses by partial correlation where every candidate's p-value is zero."""
    x = _strong_chain(0)
    test = FisherZ(x)
    first = [test.statistic_pvalue(1, other, []) for other in (0, 2, 3)]
    assert {p for _, p in first} == {0.0} and len({abs(r) for r, _ in first}) == 3
    want = moral_graph(x)
    for order in permutations(range(4)):
        np.testing.assert_array_equal(_mapped(moral_graph, x, np.array(order)), want)
