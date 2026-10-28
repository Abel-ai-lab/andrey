"""Ties break the same way under every column order.

Each test fits data built so that a method meets a tie, refits under every column order, maps the
result back to the original labels, and expects the original fit. Where the tie is broken by
:func:`~andrey.search.hc.tie_ranks`, the same fits under ``tie_break="position"`` change under at
least one order, so the orders exercise the tie-break and the switch reaches the method.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

import andrey
from andrey.constraint.pc import pc
from andrey.core import TAIL
from andrey.core.ci import FisherZ
from andrey.core.independence import gamma_gram
from andrey.core.score import BICScore
from andrey.irregular.bottom_up_parce import _fisher_hsic_test, _residual
from andrey.latent.gin import _root_pvalue, gin
from andrey.search._parallel import _KEY_DP
from andrey.search.calm import moral_graph
from andrey.search.ges import ges
from andrey.search.gies import gies
from andrey.search.xges import xges
from andrey.temporal.pcmci import design_test
from tests.recovery.helpers.xges_rescan import xges_rescan


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


def _swap_symmetric_series(
    seed: int, drive: float = 0.3, memory: float = 0.5, period: int = 64, tau_max: int = 2
) -> np.ndarray:
    """Series 0 and 1 share a signal of ``period`` steps, and series 2 follows it one step later.

    Series 2 is ``memory`` times its last value plus ``drive`` times the signal's last value plus
    noise.

    Series 1 is series 0 shifted by one period, and series 2 repeats with that period, so swapping
    series 0 and 1 shifts all three by one period. The first ``2 * tau_max`` steps repeat the last
    ones, so every window of lags of the swapped series is a window of the original. The values are
    integers, so sums over the windows are exact.
    """
    rng = np.random.default_rng(seed)
    shared, driven = np.zeros(period), np.zeros(period)
    noise = rng.standard_normal(period)
    for _ in range(3):  # each pass wraps around, so the period's start follows its end
        for t in range(period):
            shared[t] = 0.5 * shared[t - 1] + noise[t]
    noise = rng.standard_normal(period)
    for _ in range(3):
        for t in range(period):
            driven[t] = memory * driven[t - 1] + drive * shared[t - 1] + noise[t]
    first = np.round(10 * (np.tile(shared, 2) + 0.5 * rng.standard_normal(2 * period)))
    x = np.column_stack([first, np.roll(first, period), np.round(10 * np.tile(driven, 2))])
    return np.vstack([x[-2 * tau_max :], x])


def _mapped(fit, x: np.ndarray, order: np.ndarray) -> np.ndarray:
    """``fit`` on the columns of ``x`` in ``order``, mapped back to the original labels."""
    back = np.argsort(order)
    return fit(x[:, order])[..., back[:, None], back]


def _assert_order_free(fit, x: np.ndarray) -> None:
    orders = [np.array(order) for order in permutations(range(x.shape[1]))]
    want = fit(x)
    for order in orders:
        np.testing.assert_array_equal(_mapped(fit, x, order), want)
    with andrey.config(tie_break="position"):
        by_position = fit(x)
        assert any((_mapped(fit, x, order) != by_position).any() for order in orders)


def test_ges_ties() -> None:
    """GES adds the 0 - 1 edge, then column 2's edges to 0 and to 1 tie; it adds one of them."""
    x = _swap_symmetric(0)
    score, n = BICScore(x), x.shape[0]
    assert round(score.score(2, [0]) / n, _KEY_DP) == round(score.score(2, [1]) / n, _KEY_DP)
    _assert_order_free(lambda data: ges(data)[0].to_numpy(), x)


def test_xges_ties() -> None:
    """XGES meets the tie GES meets: column 2's edges to 0 and to 1."""
    x = _swap_symmetric(0)
    for search in (xges, xges_rescan):
        _assert_order_free(lambda data: search(data)[0].to_numpy(), x)


def test_calm_moral_graph_ties() -> None:
    """IAMB's first choice for column 2 ties between columns 0 and 1."""
    x = _swap_symmetric_sem(181)
    test = FisherZ(x)
    (r0, _), (r1, _) = test.statistic_pvalue(2, 0, []), test.statistic_pvalue(2, 1, [])
    assert round(abs(r0), _KEY_DP) == round(abs(r1), _KEY_DP)
    _assert_order_free(moral_graph, x)


def test_calm_moral_graph_zero_p_values() -> None:
    """IAMB chooses by partial correlation where every candidate's p-value is zero."""
    x = _strong_chain(0)
    test = FisherZ(x)
    first = [test.statistic_pvalue(1, other, []) for other in (0, 2, 3)]
    assert {p for _, p in first} == {0.0} and len({abs(r) for r, _ in first}) == 3
    want = moral_graph(x)
    for order in permutations(range(4)):
        np.testing.assert_array_equal(_mapped(moral_graph, x, np.array(order)), want)


def _assert_lag_one_tie(x: np.ndarray) -> None:
    """Series 0 and 1 at lag 1 have the same statistic against series 2, with no conditions."""
    test = design_test("student_t", x, 2)
    statistics, _ = test.statistic_pvalues(np.array([3, 4]), np.array([2, 2]), np.zeros((2, 0)))
    assert statistics[0] == statistics[1]


def test_pcmciplus_ties() -> None:
    """PC1 meets equal strengths for the lagged links of series 0 and 1 into series 2."""
    x = _swap_symmetric_series(0)
    _assert_lag_one_tie(x)

    def fit(data: np.ndarray) -> np.ndarray:
        graph = andrey.pcmciplus(data, tau_max=2, alpha=0.2).structure
        return np.stack([graph.lag(k).to_numpy() for k in graph.lags])

    _assert_order_free(fit, x)


def test_pcmci_parents_ties() -> None:
    """PC1's choice of parents for series 2 follows the series when they are reordered."""
    x = _swap_symmetric_series(0)

    def parents(order: np.ndarray) -> dict[int, list[tuple[int, int]]]:
        found = andrey.pcmci(x[:, order], tau_max=2, pc_alpha=0.2).metadata["parents"]
        return {
            int(order[j]): [(int(order[i]), lag) for i, lag in links]
            for j, links in enumerate(found)
        }

    orders = [np.array(order) for order in permutations(range(3))]
    want = parents(np.arange(3))
    assert all(parents(order) == want for order in orders)
    with andrey.config(tie_break="position"):
        by_position = parents(np.arange(3))
        assert any(parents(order) != by_position for order in orders)


def test_lpcmci_ties() -> None:
    """The separating-set searches for series 2 meet equal statistics of series 0 and 1."""
    x = _swap_symmetric_series(0, drive=1.0, memory=0.0)
    _assert_lag_one_tie(x)

    def fit(data: np.ndarray) -> np.ndarray:
        graph = andrey.lpcmci(data, tau_max=2, alpha=0.2).structure
        return np.stack([graph.lag(k).to_numpy() for k in graph.lags])

    _assert_order_free(fit, x)


def test_bottom_up_parce_lingam_ties() -> None:
    """Columns 0 and 1 are children of column 2 and tie as the first sink.

    Every row is followed by a copy with columns 0 and 1 swapped, so each leading block of rows,
    which the HSIC bandwidth reads, is symmetric too.
    """
    rng = np.random.default_rng(0)
    z = rng.uniform(-1, 1, (300, 3))
    z[:, :2] += z[:, 2:]
    z = np.round(10 * z)
    x = np.empty((600, 3))
    x[0::2], x[1::2] = z, z[:, [1, 0, 2]]
    centered = x - x.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    grams = [gamma_gram(centered[:, [j]]) for j in range(3)]
    stats = []
    for j in (0, 1):
        residual = _residual(centered, cov, np.array([1 - j, 2]), j)
        stats.append(_fisher_hsic_test([grams[1 - j], grams[2]], residual, 600, np.inf)[1])
    assert round(stats[0], _KEY_DP) == round(stats[1], _KEY_DP)

    def order(data: np.ndarray, columns: np.ndarray) -> list[int]:
        found = andrey.bottom_up_parce_lingam(data[:, columns]).metadata["causal_order"]
        return [int(columns[v]) for v in found]

    orders = [np.array(columns) for columns in permutations(range(3))]
    want = order(x, np.arange(3))
    assert all(order(x, columns) == want for columns in orders)
    with andrey.config(tie_break="position"):
        by_position = order(x, np.arange(3))
        assert any(order(x, columns) != by_position for columns in orders)


def test_gin_ties() -> None:
    """Two clusters of two indicators each pass as the root with the same lowest p-value.

    Every row is followed by a copy with the clusters swapped, so each leading block of rows, which
    the HSIC bandwidth reads, is symmetric too.
    """
    rng = np.random.default_rng(15)
    latent = rng.uniform(-1, 1, (400, 2)) ** 3
    noise = rng.uniform(-1, 1, (400, 4)) ** 3
    z = np.round(20 * (latent[:, [0, 0, 1, 1]] * np.array([1, 2, 1, 2]) + 0.3 * noise))
    x = np.empty((800, 4))
    x[0::2], x[1::2] = z, z[:, [2, 3, 0, 1]]
    cov = np.cov(x, rowvar=False)
    grams = [gamma_gram(x[:, j]) for j in range(4)]
    lowest = [_root_pvalue(x, cov, grams, [a, a + 1], [b, b + 1], []) for a, b in ((0, 2), (2, 0))]
    assert round(lowest[0], _KEY_DP) == round(lowest[1], _KEY_DP)
    orders = [np.arange(4), np.array([2, 3, 0, 1])]

    def causal_order(columns: np.ndarray) -> list[list[int]]:
        return [sorted(int(columns[v]) for v in c) for c in gin(x[:, columns]).causal_order]

    want = causal_order(orders[0])
    assert len(want) == 2
    assert all(causal_order(columns) == want for columns in orders)
    with andrey.config(tie_break="position"):
        by_position = causal_order(orders[0])
        assert any(causal_order(columns) != by_position for columns in orders)


def test_pc_collider_ties() -> None:
    """Colliders at 1 and at 2 dispute the edge 1 - 2, and their ranks agree to the last bits.

    The data follow ``0 -> 1 <- L -> 2 <- 3`` with ``L`` hidden, and every row is followed by a
    copy with the columns reversed. The two ranks are p-values computed from different rows, which
    can differ in their last bits by how the data lie in memory, so both colliders are oriented
    together and the disputed edge stays undirected under every column order.
    """
    rng = np.random.default_rng(0)
    e = rng.standard_normal((300, 5))
    z = np.column_stack(
        [e[:, 0], e[:, 0] + e[:, 4] + 0.5 * e[:, 1], e[:, 4] + e[:, 3] + 0.5 * e[:, 2], e[:, 3]]
    )
    x = np.empty((600, 4))
    x[0::2], x[1::2] = z, z[:, ::-1]
    trace: list[dict] = []
    want = pc(x, _trace=trace).to_numpy()
    ranks = [step["p"] for step in trace if step["step"] == "collider"]
    assert len(ranks) == 2 and round(ranks[0], _KEY_DP) == round(ranks[1], _KEY_DP)
    for order in permutations(range(4)):
        found = _mapped(lambda data: pc(data).to_numpy(), x, np.array(order))
        np.testing.assert_array_equal(found, want)
    assert want[1, 2] == want[2, 1] == TAIL


def test_gies_ties() -> None:
    """GIES meets the tie GES meets, with column 2 set by an intervention in some rows."""
    x = _swap_symmetric(0)
    mask = np.zeros(x.shape, dtype=bool)
    mask[np.arange(len(x)) % 5 == 0, 2] = True  # the rows of both halves alike

    def fit(data: np.ndarray, columns: np.ndarray) -> np.ndarray:
        return gies(data, mask[:, columns])[0].to_numpy()

    orders = [np.array(order) for order in permutations(range(3))]
    want = fit(x, np.arange(3))
    for order in orders:
        back = np.argsort(order)
        np.testing.assert_array_equal(fit(x[:, order], order)[np.ix_(back, back)], want)
    with andrey.config(tie_break="position"):
        by_position = fit(x, np.arange(3))
        assert any(
            (fit(x[:, o], o)[np.ix_(np.argsort(o), np.argsort(o))] != by_position).any()
            for o in orders
        )


def test_bottom_up_parce_lingam_weights() -> None:
    """Column 2 is regressed on the unordered block of columns 0 and 1 in an order-free order.

    Column 2 depends on columns 0 and 1, and every row is followed by a copy with columns 0 and 1
    swapped, so the two are left unordered and the adaptive-Lasso fit of column 2 reads both.
    """
    rng = np.random.default_rng(4)
    z = rng.uniform(-1, 1, (300, 3))
    z[:, 2] += z[:, 0] + z[:, 1]
    z = np.round(10 * z)
    x = np.empty((600, 3))
    x[0::2], x[1::2] = z, z[:, [1, 0, 2]]
    out = andrey.bottom_up_parce_lingam(x)
    assert out.metadata["causal_order"] == [[0, 1], 2]
    want = out.structure.to_numpy()
    for order in permutations(range(3)):
        found = _mapped(
            lambda data: andrey.bottom_up_parce_lingam(data).structure.to_numpy(),
            x,
            np.array(order),
        )
        np.testing.assert_array_equal(found, want)


def test_calm_fit_order() -> None:
    """CALM's initial weights and gate noise follow the columns, so its graph follows them too."""
    pytest.importorskip("torch")
    z = np.random.default_rng(0).standard_normal((200, 4))
    z[:, 1] += z[:, 0]
    z[:, 2] += 0.8 * z[:, 1]
    z[:, 3] += z[:, 0] - z[:, 2]

    def fit(data: np.ndarray) -> np.ndarray:
        return andrey.calm(data, seed=0, max_iter=4, subproblem_iter=300).structure.to_numpy()

    want = fit(z)
    for order in ([3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]):
        np.testing.assert_array_equal(_mapped(fit, z, np.array(order)), want)
