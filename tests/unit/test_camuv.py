"""CAM-UV's contract: knowledge, units, offsets, column order, ties, cycles, and the data it
refuses."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pytest

import andrey
from andrey.core import ARROW, CIRCLE
from andrey.data import sample_scm
from andrey.fcm.camuv import drop_cycles
from andrey.search.hc import tie_ranks

N = 600


def _fit(X, **kwargs) -> andrey.StructureOutput:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        return andrey.camuv(X, **kwargs)


def _edges(out: andrey.StructureOutput, kind: str = "directed") -> set[tuple]:
    return {(i, j) for i, j, k in out.structure.oriented_edges() if k == kind}


def _fork(seed: int = 0) -> np.ndarray:
    """``x0 <- h -> x1``, ``x1 -> x2``, ``x0 -> x3`` with ``h`` hidden, nonlinear effects, and
    uniform noise."""
    rng = np.random.default_rng(seed)
    h, *noise = rng.uniform(-1, 1, (5, N))
    x0 = np.sin(3 * h) + 0.3 * noise[0]
    x1 = np.cos(3 * h) + 0.3 * noise[1]
    return np.column_stack(
        [x0, x1, np.tanh(2 * x1) + 0.3 * noise[2], np.tanh(2 * x0) + 0.3 * noise[3]]
    )


@pytest.fixture(scope="module")
def fork_fit():
    return _fit(_fork())


def test_the_fork_is_found(fork_fit):
    assert _edges(fork_fit) == {(0, 3), (1, 2)}
    assert _edges(fork_fit, "circle") == {(0, 1)}


def test_rescaling_a_column_changes_nothing(fork_fit):
    out = _fit(_fork() * np.array([1e-4, 3.7, 2e3, 0.02]))
    assert out.structure == fork_fit.structure
    assert out.metadata == fork_fit.metadata


def test_shifting_a_column_changes_nothing(fork_fit):
    out = _fit(_fork() + np.array([1e3, -50.0, 7.0, 1e4]))
    assert out.structure == fork_fit.structure
    assert out.metadata == fork_fit.metadata


def _additive(seed: int = 0) -> np.ndarray:
    case = sample_scm(
        functional="additive_noise", d=6, n=400, seed=seed, density=2, scale="rescale"
    )
    return np.asarray(case.data, dtype=float)


def test_a_new_row_order_changes_nothing():
    X = _additive()
    rows = np.random.default_rng(100).permutation(X.shape[0])
    assert _fit(X[rows]).structure == _fit(X).structure


def test_flipping_a_column_sign_changes_nothing():
    X = _additive()
    assert _fit(X * np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])).structure == _fit(X).structure


def test_a_power_of_two_unit_changes_nothing():
    X = np.arange(60.0)[:, None] + np.random.default_rng(0).uniform(-0.4, 0.4, (60, 4))
    assert _fit(X * np.array([4.0, 1.0, 1.0, 1.0])).structure == _fit(X).structure


def test_a_new_column_order_relabels_the_result():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_fork(), columns=list("abcd"))
    fits = [_fit(frame[list(columns)]) for columns in ("abcd", "dbca", "cadb")]
    named = [(_edges(out), _edges(out, "circle")) for out in fits]
    assert named[0] == ({("a", "d"), ("b", "c")}, {("a", "b")})
    for edges, circles in named[1:]:
        assert edges == named[0][0]
        assert {frozenset(pair) for pair in circles} == {frozenset(("a", "b"))}


def _mirrored(seed: int = 0) -> np.ndarray:
    """Linear Gaussian ``x`` and ``y`` with each row's swap also a row.

    Swapping ``x`` and ``y`` only reorders the rows, so either one as the sink of the pair has the
    same p-value up to rounding.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(N // 2)
    b = 0.8 * a + 0.6 * rng.standard_normal(N // 2)
    return np.column_stack([np.concatenate([a, b]), np.concatenate([b, a])])


def test_a_tied_sink_is_the_lower_ranked_column_in_any_column_order():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_mirrored(), columns=list("xy"))
    ranks = dict(zip("xy", tie_ranks(frame.to_numpy()).tolist(), strict=True))
    sink, cause = sorted("xy", key=ranks.get)
    for columns in (["x", "y"], ["y", "x"]):
        assert _edges(_fit(frame[columns])) == {(cause, sink)}
    with andrey.config(tie_break="position"):
        assert _edges(_fit(frame[["y", "x"]])) == {("x", "y")}


def test_every_edge_on_a_directed_cycle_goes_but_a_required_one():
    required = np.zeros((4, 4), dtype=bool)
    cycle = [{2}, {0}, {1}, {0}]  # 0 -> 1 -> 2 -> 0, and 0 -> 3 off the cycle
    parents = [set(p) for p in cycle]
    dropped = drop_cycles(parents, required)
    assert parents == [set(), set(), set(), {0}]
    assert {(int(i), int(j)) for i, j in zip(*np.nonzero(dropped))} == {(0, 1), (1, 2), (2, 0)}
    required[0, 1] = True
    parents = [set(p) for p in cycle]
    drop_cycles(parents, required)
    assert parents == [set(), {0}, set(), {0}]


def _has_directed_cycle(edges: set[tuple[int, int]], d: int) -> bool:
    reach = np.zeros((d, d), dtype=bool)
    for i, j in edges:
        reach[i, j] = True
    for k in range(d):
        reach |= np.outer(reach[:, k], reach[k, :])
    return bool(reach.diagonal().any())


def test_a_fit_whose_parents_close_cycles_drops_and_records_their_edges(caplog):
    case = sample_scm(functional="additive_noise", d=12, n=500, seed=14, density=2)
    with caplog.at_level(logging.INFO, logger="andrey.api.fcm"):
        out = _fit(np.asarray(case.data, dtype=float))
    dropped = {tuple(e) for e in out.metadata["cycle_edges"]}
    assert dropped
    assert not _has_directed_cycle(_edges(out), 12)
    assert not dropped & _edges(out)
    assert "directed cycles" in caplog.text


def test_a_forbidden_edge_is_never_in_the_graph():
    out = _fit(_fork(), knowledge=andrey.Knowledge(forbidden=[(1, 2)]))
    assert (1, 2) not in _edges(out)


def test_a_required_edge_stays_on_independent_columns():
    X = np.random.default_rng(1).standard_normal((N, 3))
    out = _fit(X, knowledge=andrey.Knowledge(required=[(2, 0)]))
    assert _edges(out) == {(2, 0)}
    assert out.metadata["latent_pairs"] == []


def test_a_pair_forbidden_both_ways_is_never_marked():
    out = _fit(_fork(), knowledge=andrey.Knowledge(forbidden=[(0, 1), (1, 0)]))
    assert _edges(out, "circle") == set()
    assert _edges(out) == {(0, 3), (1, 2)}


def test_tiers_forbid_edges_into_an_earlier_tier():
    # x2 and x3 first: no edge may enter them from x0 or x1, so neither true edge is found.
    out = _fit(_fork(), knowledge=andrey.Knowledge(tiers=[[2, 3], [0, 1]]))
    assert all(i in (2, 3) for i, j in _edges(out) if j in (2, 3))


def _tiered(seed: int = 0) -> np.ndarray:
    """Ten variables: the fork of ``_fork`` with ``h`` hidden, and the chains x4 -> x5 -> x6 and
    x7 -> x8 -> x9, with nonlinear effects and uniform noise."""
    rng = np.random.default_rng(seed)
    X = _fork(seed)
    noise = 0.3 * rng.uniform(-1, 1, (N, 6))
    x4, x7 = rng.uniform(-1, 1, (2, N))
    x5 = np.sin(3 * x4) + noise[:, 0]
    x8 = np.tanh(3 * x7) + noise[:, 1]
    rest = [x4, x5, x5**2 + noise[:, 2], x7, x8, np.abs(x8) + noise[:, 3]]
    return np.column_stack([X, *rest])


def test_tiers_and_edges_hold_among_ten_variables():
    early, late = [0, 4, 7], [1, 2, 3, 5, 6, 8, 9]
    knowledge = andrey.Knowledge(tiers=[early, late], required=[(7, 9)], forbidden=[(5, 6)])
    out = _fit(_tiered(), knowledge=knowledge)
    edges = _edges(out)
    assert (7, 9) in edges and (5, 6) not in edges
    assert not any(i in late and j in early for i, j in edges)
    across = [(i, j) for i, j in out.metadata["latent_pairs"] if (i in early) != (j in early)]
    assert across  # x0 o-o x1, from the hidden common cause, crosses the tiers
    for i, j in across:
        first, later = (i, j) if i in early else (j, i)
        assert out.structure.endpoints(later, first)[0] == ARROW
        assert out.structure.endpoints(first, later)[0] == CIRCLE


@pytest.mark.filterwarnings("ignore::andrey.DataTypeWarning")
def test_a_binary_column_is_named():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_fork(), columns=list("abcd"))
    frame["b"] = np.arange(N) % 2
    with pytest.raises(ValueError, match="'b'"):
        _fit(frame)


def test_a_column_that_totals_others_is_named():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_fork(), columns=list("abcd"))
    frame["t"] = frame["a"] + frame["c"]
    with pytest.raises(ValueError, match="'t'"):
        _fit(frame)


def test_fewer_rows_than_twice_the_columns_are_refused():
    X = np.random.default_rng(4).standard_normal((15, 8))
    with pytest.raises(ValueError, match="16"):
        _fit(X)
    assert _fit(X[:, :7]).structure.n_nodes == 7


def test_a_balanced_grid_of_independent_columns_fits():
    """Each value of one column meets each of the other once, so the HSIC statistic is 0 in exact
    arithmetic and slightly negative after rounding."""
    X = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    out = _fit(X)
    assert out.structure.n_nodes == 2
    assert out.structure.oriented_edges() == []
