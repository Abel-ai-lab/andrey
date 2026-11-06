"""Adapter tests for ``andrey.api._temporal`` (VARMA-LiNGAM, Granger, Longitudinal LiNGAM)."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.api._temporal import adapt_granger, adapt_longitudinal_lingam, adapt_varma_lingam
from andrey.core import ARROW, TAIL, StructureOutput, TemporalStructure


def _varma_fit() -> tuple[list[int], np.ndarray, np.ndarray]:
    """A VARMA-LiNGAM fit: the causal order, ``psis``, and ``omegas``."""
    n = 3
    psis = np.zeros((2, n, n))  # AR order p=1 (index 0 instantaneous, index 1 lag-1)
    psis[0, 1, 0] = 0.6  # instantaneous edge 0 -> 1
    psis[1, 2, 1] = -0.4  # lag-1 edge 1 -> 2
    omegas = np.zeros((2, n, n))  # MA order q=2, independent of the AR order
    omegas[0, 1, 2] = 0.2
    omegas[1, 0, 2] = -0.1
    return [2, 0, 1], psis, omegas


def test_varma_lingam_dual_stack():
    order, psis, omegas = _varma_fit()
    out = adapt_varma_lingam(order, psis, omegas)
    t = out.structure
    assert isinstance(t, TemporalStructure)
    # The lag topology comes from psis (AR) only; both stacks are present and lossless.
    assert t.n_lags == psis.shape[0]
    assert np.array_equal(t.lag_weights, psis.transpose(0, 2, 1))  # canonical B.T
    assert np.array_equal(t.lag_weights_ma, omegas.transpose(0, 2, 1))
    assert t.lag(0).endpoints(0, 1) == (TAIL, ARROW)
    assert t.lag(1).endpoints(1, 2) == (TAIL, ARROW)
    assert out.ordering == (2, 0, 1)


def test_varma_lingam_json_roundtrip(tmp_path):
    order, psis, omegas = _varma_fit()
    out = adapt_varma_lingam(order, psis, omegas)
    p = tmp_path / "varma.json"
    out.save(p)
    back = StructureOutput.load(p)
    assert back == out
    assert np.array_equal(back.structure.lag_weights_ma, omegas.transpose(0, 2, 1))  # type: ignore[union-attr]


def _granger_coeff() -> np.ndarray:
    """Granger-lasso coeff (d=3, maxlag=2): block A_k[i, j] is influence j -> i at lag k."""
    d, maxlag = 3, 2
    coeff = np.zeros((d, d * maxlag))
    coeff[1, 0] = 0.7  # lag-1 block: edge 0 -> 1
    coeff[0, 0] = 0.4  # lag-1 block: autoregressive self-loop 0 -> 0 (kept in topology)
    coeff[1, d + 2] = -0.5  # lag-2 block: edge 2 -> 1
    return coeff


def test_granger_lag_topology_and_count():
    coeff = _granger_coeff()
    out = adapt_granger(coeff, maxlag=2)
    t = out.structure
    assert isinstance(t, TemporalStructure)
    assert t.n_lags == 2  # no lag-0; one graph per lag 1..maxlag
    assert t.lags == (1, 2)  # explicit lag axis, no lag-0
    # every lag view (Granger has no contemporaneous block) is the directed-cyclic kind.
    assert t.lag(1).kind == "digraph"
    assert t.lag(2).kind == "digraph"
    assert t.lag(1).endpoints(0, 1) == (TAIL, ARROW)  # lag-1 edge 0 -> 1 (lookup by lag value)
    assert t.lag(1).endpoints(0, 0) == (ARROW, ARROW)  # lag-1 autoregressive self-loop 0 -> 0
    assert t.lag(2).endpoints(2, 1) == (TAIL, ARROW)  # lag-2 edge 2 -> 1 (tail at cause 2)
    # Re-stacked coefficients stored losslessly in canonical (B.T) orientation; no causal order.
    ar = out.structure.lag_weights
    assert ar.shape == (2, 3, 3)
    assert ar[0, 0, 1] == 0.7  # canonical W[0->1] at lag 1 (raw block[1, 0])
    assert ar[0, 0, 0] == 0.4  # canonical self-loop weight W[0->0] at lag 1 (raw block[0, 0])
    assert ar[1, 2, 1] == -0.5  # canonical W[1->2] at lag 2 (raw block[2, 1])
    assert out.ordering is None


def test_granger_json_roundtrip(tmp_path):
    out = adapt_granger(_granger_coeff(), maxlag=2)
    p = tmp_path / "granger.json"
    out.save(p)
    assert StructureOutput.load(p) == out


@pytest.mark.parametrize("shape", [(3, 5), (6,), (3, 6, 1)])
def test_granger_rejects_bad_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        adapt_granger(np.zeros(shape), maxlag=2)


def test_orientation_invariant_weights_match_edges():
    """Canonical ``lag_weights[pos][i, j] != 0`` <=> an arrowhead into j on edge i-j.

    Holds for directed (i -> j), a 2-cycle (both ways), and lag>=1 self-loops (i -> i) alike, so
    the stored weights and the derived graph share one orientation.
    """
    n = 3
    psis = np.zeros((3, n, n))
    psis[0, 1, 0] = 0.6  # lag-0 (instantaneous): edge 0 -> 1
    psis[1, 2, 1] = -0.4  # lag-1: edge 1 -> 2
    psis[1, 0, 0] = 0.5  # lag-1: autoregressive self-loop
    psis[2, 1, 0] = 0.3  # lag-2: edge 0 -> 1 ...
    psis[2, 0, 1] = 0.2  # lag-2: ... and edge 1 -> 0, a 2-cycle
    t = adapt_varma_lingam([0, 1, 2], psis, np.zeros((0, n, n))).structure
    assert isinstance(t, TemporalStructure)
    weights = t.lag_weights
    for pos, lag_value in enumerate(t.lags):
        g = t.lag(lag_value)
        for i in range(t.n_nodes):
            for j in range(t.n_nodes):
                has_weight = weights[pos, i, j] != 0.0
                arrowhead_at_j = g.endpoints(i, j)[1] == ARROW
                assert has_weight == arrowhead_at_j, (pos, i, j)


def _longitudinal_fit() -> tuple[np.ndarray, list[list[int]]]:
    """A Longitudinal LiNGAM fit: adjacencies of shape (T, 1+n_lags, p, p) with NaN blocks, and
    the per-time causal orders."""
    T, L, p = 4, 2, 3
    adj = np.full((T, 1 + L, p, p), np.nan)
    for t in range(1, T):
        adj[t, 0] = np.zeros((p, p))
        adj[t, 0, 1, 0] = 0.6  # instantaneous edge 0 -> 1
        for lag in range(L):
            if t - lag == 0:
                continue  # reaches before the panel start -> stays NaN
            adj[t, lag + 1] = np.zeros((p, p))
            adj[t, lag + 1, 2, 1] = 0.3  # lag edge 1 -> 2
    orders = [[]] + [[0, 1, 2] for _ in range(T - 1)]  # occasion 0 is an empty placeholder
    return adj, orders


def test_longitudinal_adapter_stores_the_time_axis_losslessly():
    out = adapt_longitudinal_lingam(*_longitudinal_fit())
    t = out.structure
    assert isinstance(t, TemporalStructure)
    # occasion 0 (all-NaN) is dropped; occasions 1..3 become the explicit time axis.
    assert t.n_times == 3
    assert t.times == (1, 2, 3)
    # the per-occasion tensor is lossless: NaN survives where a lag reaches before the panel start.
    assert t.time_weights.shape == (3, 1 + 2, 3, 3)
    assert np.isnan(t.time_weights[0, 2]).any()  # occasion 1, lag-2 block is uncomputable
    # the lag-only view projects the final occasion (bit-identical to a VAR collapse).
    assert t.lag(0) == t.at(t.times[-1], 0)
    assert t.lag(0).endpoints(0, 1) == (TAIL, ARROW)  # instantaneous 0 -> 1
    # per-time causal orders (with the t=0 placeholder) ride in metadata; no envelope ordering.
    assert out.metadata["causal_orders"][0] == []
    assert out.ordering is None


def test_longitudinal_adapter_json_roundtrip(tmp_path):
    out = adapt_longitudinal_lingam(*_longitudinal_fit())
    p = tmp_path / "longitudinal.json"
    out.save(p)
    assert StructureOutput.load(p) == out  # (time, lag) axis + NaN weights round-trip


def test_longitudinal_adapter_gives_an_uncomputable_block_no_link_and_zero_fills_the_lag_view():
    # A lag block before the panel start (k > t) is uncomputable, and the fit leaves it NaN; the
    # lag-only view zero-fills it. With T=2 and n_lags=2, the last occasion's lag-2 block is
    # uncomputable.
    p = 2
    adj = np.full((2, 1 + 2, p, p), np.nan)
    adj[1, 0] = np.zeros((p, p))
    adj[1, 0, 1, 0] = 0.5  # instantaneous 0 -> 1
    adj[1, 1] = np.zeros((p, p))
    adj[1, 1, 1, 0] = 0.3  # lag-1 (reaches occasion 0, valid)
    t = adapt_longitudinal_lingam(adj, [[], [0, 1]]).structure
    assert t.n_times == 1 and t.times == (1,)
    assert np.isnan(t.time_weights[0, 2]).all()  # the uncomputable block is kept NaN (lossless)
    assert int(t.at(1, 2).to_numpy().sum()) == 0  # its graph has no link
    # the lag-only view zero-fills the NaN block and stays finite.
    assert np.isfinite(t.lag_weights).all()
    assert np.array_equal(t.lag_weights, np.nan_to_num(t.time_weights[0]))
