"""RESIT's contract: knowledge, units, ties, the p-values it compares, and the data it refuses."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
from andrey.core.independence import gamma_gram, hsic_gamma_from_grams, hsic_gamma_log_p
from andrey.core.units import unit_factors
from andrey.fcm.gam import BasisCache
from andrey.fcm.resit import _gram, _residual, causal_order
from andrey.search.hc import tie_ranks

N = 400


def _fit(X, **kwargs) -> andrey.StructureOutput:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        return andrey.resit(X, **kwargs)


def _edges(out: andrey.StructureOutput) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j, _ in out.structure.oriented_edges()}


def _chain(seed: int = 0) -> np.ndarray:
    """``x0 -> x1 -> x2`` with nonlinear effects and uniform noise."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2, 2, N)
    x1 = np.sin(2 * x0) + x0 + rng.uniform(-0.5, 0.5, N)
    x2 = np.tanh(2 * x1) + rng.uniform(-0.5, 0.5, N)
    return np.column_stack([x0, x1, x2])


@pytest.fixture(scope="module")
def chain_fit():
    return _fit(_chain())


def test_the_chain_is_found(chain_fit):
    assert _edges(chain_fit) == {(0, 1), (1, 2)}
    assert chain_fit.ordering == (0, 1, 2)


def test_rescaling_a_column_changes_neither_the_graph_nor_the_order(chain_fit):
    out = _fit(_chain() * np.array([1e-4, 3.7, 2e3]))
    assert out.structure == chain_fit.structure
    assert out.ordering == chain_fit.ordering


def _mirrored(seed: int = 0) -> np.ndarray:
    """Linear Gaussian ``x`` and ``y`` with each row's swap also a row.

    Swapping ``x`` and ``y`` only reorders the rows, so either one as the sink has the same
    p-value up to rounding.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(N // 2)
    b = 0.8 * a + 0.6 * rng.standard_normal(N // 2)
    return np.column_stack([np.concatenate([a, b]), np.concatenate([b, a])])


def _named(out: andrey.StructureOutput) -> set[tuple[str, str]]:
    return {(str(i), str(j)) for i, j, _ in out.structure.oriented_edges()}


def test_tied_p_values_make_the_earlier_column_the_sink():
    X = _mirrored()
    allowed = ~np.eye(2, dtype=bool)
    for columns in (X, X[:, ::-1]):
        unit = columns * unit_factors(columns, standardize=True)
        assert causal_order(BasisCache(columns), unit, allowed).tolist() == [1, 0]


def test_a_tied_sink_is_the_lower_ranked_column_in_any_column_order():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_mirrored(), columns=list("xy"))
    ranks = dict(zip("xy", tie_ranks(frame.to_numpy()).tolist(), strict=True))
    sink, cause = sorted("xy", key=ranks.get)
    for columns in (["x", "y"], ["y", "x"]):
        assert _named(_fit(frame[columns])) == {(cause, sink)}
    with andrey.config(tie_break="position"):
        assert _named(_fit(frame[["y", "x"]])) == {("x", "y")}


def test_a_forbidden_edge_is_never_in_the_graph():
    out = _fit(_chain(), knowledge=andrey.Knowledge(forbidden=[(0, 1)]))
    assert (0, 1) not in _edges(out)


def test_a_required_edge_stays_on_independent_columns():
    X = np.random.default_rng(1).standard_normal((N, 3))
    out = _fit(X, knowledge=andrey.Knowledge(required=[(2, 0)]))
    assert _edges(out) == {(2, 0)}
    assert out.ordering.index(2) < out.ordering.index(0)


def test_tiers_order_the_variables_and_forbid_edges_into_an_earlier_tier():
    # x2 first: no edge may enter it, and no order may place x0 or x1 before it.
    out = _fit(_chain(), knowledge=andrey.Knowledge(tiers=[[2], [0, 1]]))
    assert all(j != 2 for _, j in _edges(out))
    assert out.ordering[0] == 2


def test_p_values_far_below_double_precision_stay_positive_and_ordered():
    rng = np.random.default_rng(3)
    x = rng.uniform(-2, 2, 200)
    noise = rng.standard_normal(200)
    gram = gamma_gram(x, median_rows=None)
    p = [
        hsic_gamma_from_grams(
            gram, gamma_gram(np.sin(2 * x) + s * noise, median_rows=None), 200, variance="product"
        )[1]
        for s in (0.1, 0.3)
    ]
    assert 0 < p[0] < p[1] < 1e-16


def _arc(seed: int = 2, n: int = 1600) -> np.ndarray:
    """Points near three quarters of a circle: neither coordinate is a function of the other, so
    both residuals stay dependent, and at this size both p-values lie below the smallest double."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 1.5 * np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t)]) + 0.01 * rng.standard_normal((n, 2))


def test_the_sink_follows_p_values_below_the_smallest_double():
    X = _arc()
    n = X.shape[0]
    unit = X * unit_factors(X, standardize=True)
    bases = BasisCache(X)
    p, log_p = {}, {}
    for sink, cause in ((1, 0), (0, 1)):
        grams = _gram(unit[:, [cause]]), _gram(_residual(bases, sink, [cause]))
        p[sink] = hsic_gamma_from_grams(*grams, n, variance="product")[1]
        log_p[sink] = hsic_gamma_log_p(*grams, n, variance="product")
    sink = max(log_p, key=log_p.get)
    # Both p-values are 0, and the ranks, which break a tie, would make the other column the sink.
    assert p == {0: 0.0, 1: 0.0}
    assert abs(log_p[0] - log_p[1]) > 1
    assert int(np.argmin(tie_ranks(X))) != sink
    assert _fit(X).ordering[-1] == sink


@pytest.mark.filterwarnings("ignore::andrey.DataTypeWarning")
def test_a_binary_column_is_named():
    X = _chain()
    X[:, 1] = np.arange(N) % 2
    with pytest.raises(ValueError, match="column 1 has fewer than 3 distinct values; resit"):
        _fit(X)


def test_fewer_rows_than_twice_the_columns_are_refused():
    X = np.random.default_rng(4).standard_normal((15, 8))
    with pytest.raises(ValueError, match=r"needs at least 16 rows \(samples\) over 8 variables"):
        _fit(X)
    assert _fit(X[:, :7]).structure.n_nodes == 7
