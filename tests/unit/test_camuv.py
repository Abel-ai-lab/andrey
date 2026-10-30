"""CAM-UV's contract: knowledge, units, column order, ties, cycles, and the data it refuses."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
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
    assert drop_cycles([set(p) for p in cycle], required) == [set(), set(), set(), {0}]
    required[0, 1] = True
    assert drop_cycles([set(p) for p in cycle], required) == [set(), {0}, set(), {0}]


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


@pytest.mark.filterwarnings("ignore::andrey.DataTypeWarning")
def test_a_binary_column_is_named():
    X = _fork()
    X[:, 1] = np.arange(N) % 2
    with pytest.raises(ValueError, match="column 1 has fewer than 3 distinct values; camuv"):
        _fit(X)


def test_fewer_rows_than_twice_the_columns_are_refused():
    X = np.random.default_rng(4).standard_normal((15, 8))
    with pytest.raises(ValueError, match=r"needs at least 16 rows \(samples\) over 8 variables"):
        _fit(X)
    assert _fit(X[:, :7]).structure.n_nodes == 7
