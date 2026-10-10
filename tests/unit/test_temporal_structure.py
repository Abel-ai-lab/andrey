"""Constructor / accessor / round-trip tests for ``TemporalStructure.from_lag_graphs``
(VAR + VARMA)."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.api._marks import directed_marks
from andrey.core import ARROW, TAIL, GraphStructure, StructureOutput, TemporalStructure


def _lag_graph(B: np.ndarray, *, kind: str) -> GraphStructure:
    """A per-lag ``GraphStructure`` from an AR slice (``B[i, j] != 0`` is edge ``j -> i``)."""
    return GraphStructure.from_numpy(directed_marks(B), kind=kind)


def _var_stack() -> tuple[tuple[GraphStructure, GraphStructure], np.ndarray]:
    n = 3
    ar = np.zeros((2, n, n))
    ar[0, 1, 0] = 0.6  # lag-0 (instantaneous) edge 0 -> 1
    ar[1, 2, 1] = -0.4  # lag-1 edge 1 -> 2
    ar[1, 0, 0] = 0.5  # lag-1 autoregressive self-loop (dropped from topology)
    lags = (_lag_graph(ar[0], kind="dag"), _lag_graph(ar[1], kind="dag"))
    return lags, ar


def test_from_lag_graphs_single_stack_var():
    lags, ar = _var_stack()
    t = TemporalStructure.from_lag_graphs(lags, lag_weights=ar)
    assert t.n_lags == 2
    assert t.type == "temporal"
    assert t.n_nodes == 3
    # lag-0 topology: only 0 -> 1.
    assert t.lag(0).endpoints(0, 1) == (TAIL, ARROW)
    assert t.lag(0) == lags[0]
    # lag-1 topology: 1 -> 2, and the autoregressive diagonal is dropped (no self-loop).
    assert t.lag(1).endpoints(1, 2) == (TAIL, ARROW)
    assert np.array_equal(np.diagonal(t.lag(1).to_numpy()), np.zeros(3))
    # AR stack is the lossless source of truth; MA absent.
    assert np.array_equal(t.lag_weights, ar)
    assert t.lag_weights_ma is None


def test_from_lag_graphs_dual_stack_varma():
    lags, ar = _var_stack()
    ma = np.zeros((1, 3, 3))  # MA order q=1, independent of the AR order
    ma[0, 2, 0] = 0.3
    t = TemporalStructure.from_lag_graphs(lags, lag_weights=ar, lag_weights_ma=ma)
    assert np.array_equal(t.lag_weights, ar)
    assert np.array_equal(t.lag_weights_ma, ma)
    # A different MA payload makes an unequal structure; identical rebuild is equal.
    assert t == TemporalStructure.from_lag_graphs(lags, lag_weights=ar, lag_weights_ma=ma)
    assert t != TemporalStructure.from_lag_graphs(lags, lag_weights=ar)


def test_stored_stacks_are_read_only():
    lags, ar = _var_stack()
    t = TemporalStructure.from_lag_graphs(lags, lag_weights=ar)
    with pytest.raises(ValueError):
        t.lag_weights[0, 0, 0] = 9.0


def test_from_lag_graphs_validation():
    lags, ar = _var_stack()
    with pytest.raises(ValueError, match="at least one lag"):
        TemporalStructure.from_lag_graphs([], lag_weights=None)
    with pytest.raises(ValueError, match="shape"):
        TemporalStructure.from_lag_graphs(lags, lag_weights=np.zeros((2, 4, 4)))  # wrong node dim
    with pytest.raises(ValueError, match="lags"):
        TemporalStructure.from_lag_graphs(
            lags, lag_weights=np.zeros((3, 3, 3))
        )  # AR count != n_lags
    with pytest.raises(ValueError, match="shape"):
        TemporalStructure.from_lag_graphs(
            lags, lag_weights_ma=np.zeros((1, 2, 2))
        )  # wrong MA node dim


def test_json_roundtrip_single_stack(tmp_path):
    lags, ar = _var_stack()
    out = StructureOutput.new(
        TemporalStructure.from_lag_graphs(lags, lag_weights=ar), metadata={"algorithm": "varlingam"}
    )
    p = tmp_path / "var.json"
    out.save(p)
    assert StructureOutput.load(p) == out


def test_json_roundtrip_dual_stack(tmp_path):
    lags, ar = _var_stack()
    ma = np.zeros((2, 3, 3))
    ma[0, 1, 2] = 0.2
    ma[1, 0, 2] = -0.1
    t = TemporalStructure.from_lag_graphs(lags, lag_weights=ar, lag_weights_ma=ma)
    out = StructureOutput.new(t, metadata={"algorithm": "varma"})
    p = tmp_path / "varma.json"
    out.save(p)
    back = StructureOutput.load(p)
    assert back == out
    assert np.array_equal(back.structure.lag_weights_ma, ma)  # type: ignore[union-attr]


def test_npz_unsupported_for_temporal(tmp_path):
    lags, ar = _var_stack()
    out = StructureOutput.new(TemporalStructure.from_lag_graphs(lags, lag_weights=ar))
    with pytest.raises(NotImplementedError):
        out.save(tmp_path / "t.npz", fmt="npz")


def test_explicit_uneven_lag_indices(tmp_path):
    """``lag(k)`` is a lookup by lag value; the axis may be uneven and persists in json."""
    lags, ar = _var_stack()
    t = TemporalStructure.from_lag_graphs(lags, lags=[0, 12], lag_weights=ar)
    assert t.lags == (0, 12)
    assert t.n_lags == 2
    assert t.lag(0) == lags[0]
    assert t.lag(12) == lags[1]  # by value, not position
    with pytest.raises(KeyError):
        t.lag(1)  # no such lag
    # the default axis is contiguous 0..L-1
    assert TemporalStructure.from_lag_graphs(lags, lag_weights=ar).lags == (0, 1)
    # length mismatch / duplicate lags are rejected (incl. an explicit empty lags=[])
    with pytest.raises(ValueError, match="lags length"):
        TemporalStructure.from_lag_graphs(lags, lags=[0])
    with pytest.raises(ValueError, match="lags length"):
        TemporalStructure.from_lag_graphs(lags, lags=[])
    with pytest.raises(ValueError, match="unique"):
        TemporalStructure.from_lag_graphs(lags, lags=[5, 5])
    # the uneven axis round-trips through json
    out = StructureOutput.new(t)
    p = tmp_path / "uneven.json"
    out.save(p)
    assert StructureOutput.load(p) == out


# ---- (time, lag) axis (LongitudinalLiNGAM) ------------------------------------------------------


def _tl_grid() -> tuple[list[list[GraphStructure]], np.ndarray]:
    """A (T=3, L=2) grid over occasions 1, 2, 3; occasion 1's lag-1 block is uncomputable (NaN)."""
    n = 2
    inst = _lag_graph(np.array([[0.0, 0.0], [0.6, 0.0]]), kind="dag")  # instantaneous 0 -> 1
    lag1 = _lag_graph(np.array([[0.0, 0.4], [0.0, 0.0]]), kind="digraph")  # lag-1 edge 1 -> 0
    empty = GraphStructure.from_numpy(np.zeros((n, n), dtype=np.int8), kind="digraph")
    grid = [[inst, empty], [inst, lag1], [inst, lag1]]  # occasion 1's lag-1 is uncomputable
    tw = np.full((3, 2, n, n), np.nan)  # (n_times, n_lags, n, n)
    for ti in range(3):
        tw[ti, 0] = np.array([[0.0, 0.6], [0.0, 0.0]])  # W[0->1] each occasion
    tw[1, 1] = tw[2, 1] = np.array([[0.0, 0.0], [0.4, 0.0]])  # W[1->0] at occasions 2, 3
    return grid, tw


def test_from_time_lag_graphs_axis_and_projection():
    grid, tw = _tl_grid()
    t = TemporalStructure.from_time_lag_graphs(grid, times=[1, 2, 3], time_weights=tw)
    assert t.n_times == 3
    assert t.times == (1, 2, 3)
    assert t.n_lags == 2
    # at() looks up by (time value, lag value); the full grid is reachable.
    assert t.at(2, 0) is grid[1][0]
    assert t.at(3, 1) is grid[2][1]
    with pytest.raises(KeyError):
        t.at(0, 0)  # no occasion 0
    with pytest.raises(KeyError):
        t.at(2, 5)  # no lag 5
    # the lag-only surface is a VIEW of the final occasion.
    assert t.lag(0) == t.at(t.times[-1], 0)
    assert t.lag(1) == t.at(t.times[-1], 1)
    assert np.array_equal(t.lag_weights, np.nan_to_num(tw[-1]))
    # the lossless per-occasion tensor keeps the uncomputable NaN block.
    assert t.time_weights.shape == (3, 2, 2, 2)
    assert np.isnan(t.time_weights[0, 1]).all()


def test_from_time_lag_graphs_defaults_and_validation():
    grid, tw = _tl_grid()
    assert TemporalStructure.from_time_lag_graphs(grid).times == (0, 1, 2)  # default 0..T-1
    with pytest.raises(ValueError, match="non-empty"):
        TemporalStructure.from_time_lag_graphs([])
    with pytest.raises(ValueError, match="times length|occasions"):
        TemporalStructure.from_time_lag_graphs(grid, times=[1, 2])  # 2 != 3 rows
    with pytest.raises(ValueError, match="times length|occasions"):
        TemporalStructure.from_time_lag_graphs(grid, times=[])  # explicit empty != 3 rows
    with pytest.raises(ValueError, match="strictly increasing"):
        TemporalStructure.from_time_lag_graphs(grid, times=[3, 1, 2])
    with pytest.raises(ValueError, match="shape"):
        TemporalStructure.from_time_lag_graphs(grid, time_weights=np.zeros((3, 2, 3, 3)))
    with pytest.raises(ValueError, match="time_graphs row 1 has 1 graphs, but row 0 has 2"):
        TemporalStructure.from_time_lag_graphs([grid[0], grid[1][:1], grid[2]])


def test_time_weights_read_only_and_equality():
    grid, tw = _tl_grid()
    t = TemporalStructure.from_time_lag_graphs(grid, times=[1, 2, 3], time_weights=tw)
    with pytest.raises(ValueError):
        t.time_weights[0, 0, 0, 0] = 9.0  # frozen store
    # NaN-bearing weights compare equal to an identical rebuild (equal_nan).
    assert t == TemporalStructure.from_time_lag_graphs(grid, times=[1, 2, 3], time_weights=tw)
    # a different time axis makes it unequal.
    assert t != TemporalStructure.from_time_lag_graphs(grid, times=[1, 2, 4], time_weights=tw)


def test_time_axis_json_roundtrip_and_format_version(tmp_path):
    import json

    grid, tw = _tl_grid()
    out = StructureOutput.new(
        TemporalStructure.from_time_lag_graphs(grid, times=[1, 2, 3], time_weights=tw),
        metadata={"algorithm": "longitudinallingam"},
    )
    p = tmp_path / "long.json"
    out.save(p)
    doc = json.loads(p.read_text())
    assert doc["format_version"] == 1  # the (time, lag) axis bumps the reader version
    assert StructureOutput.load(p) == out  # NaN in time_weights round-trips (equal_nan)
    # a lag-only structure still serializes at version 0 (old files stay byte-stable).
    lags, ar = _var_stack()
    p0 = tmp_path / "var.json"
    StructureOutput.new(TemporalStructure.from_lag_graphs(lags, lag_weights=ar)).save(p0)
    assert json.loads(p0.read_text())["format_version"] == 0


def test_time_axis_view_invariant_is_enforced():
    # the lag-only surface must equal the final occasion; a construction (for example, a corrupted /
    # hand-edited doc) that violates it is rejected, so lag(k) can never disagree with
    # at(times[-1], k).
    grid, tw = _tl_grid()
    grid_t = tuple(tuple(r) for r in grid)
    with pytest.raises(ValueError, match="final"):  # _lags is the FIRST occasion, not the final
        TemporalStructure(
            _n_nodes=2,
            _lags=grid_t[0],
            _lag_indices=(0, 1),
            _lag_weights=np.nan_to_num(tw[-1]),
            _time_graphs=grid_t,
            _time_indices=(1, 2, 3),
            _time_weights=tw,
        )
    with pytest.raises(ValueError, match="zero-filled"):  # right graphs, desynced lag_weights
        TemporalStructure(
            _n_nodes=2,
            _lags=grid_t[-1],
            _lag_indices=(0, 1),
            _lag_weights=np.zeros((2, 2, 2)),
            _time_graphs=grid_t,
            _time_indices=(1, 2, 3),
            _time_weights=tw,
        )


def test_loaded_time_weights_are_read_only(tmp_path):
    # a loaded (time, lag) structure's weights are immutable, like a constructor-built one -- the
    # deserialization path must freeze them, not just from_time_lag_graphs.
    grid, tw = _tl_grid()
    out = StructureOutput.new(
        TemporalStructure.from_time_lag_graphs(grid, times=[1, 2, 3], time_weights=tw)
    )
    p = tmp_path / "t.json"
    out.save(p)
    loaded = StructureOutput.load(p).structure
    with pytest.raises(ValueError):
        loaded.time_weights[0, 0, 0, 0] = 9.0  # frozen after load, not just after construction


def test_temporal_scoring_of_a_time_carrying_structure_fails_loud():
    # the (time, lag) axis has no metric yet; scoring must fail loud rather than silently compare
    # only the final occasion (which would report perfect for structures differing in earlier ones).
    from andrey.metrics import score, temporal_scores

    grid, tw = _tl_grid()
    t = TemporalStructure.from_time_lag_graphs(grid, times=[1, 2, 3], time_weights=tw)
    for scorer in (temporal_scores, score):
        with pytest.raises(NotImplementedError, match="time"):
            scorer(t, t)
