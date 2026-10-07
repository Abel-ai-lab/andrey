"""Adapter tests for ``andrey.api._temporal`` (VAR / VARMA / Granger).

The fitted models are duck-typed: a ``types.SimpleNamespace`` carrying the same attributes
(``adjacency_matrices_`` / ``causal_order_``) exercises the adapters.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from andrey.api._temporal import (
    _adapt_granger,
    _adapt_longitudinal_lingam,
    _adapt_var_lingam,
    _adapt_varma_lingam,
)
from andrey.core import ARROW, TAIL, StructureOutput, TemporalStructure


def _var_model() -> tuple[SimpleNamespace, np.ndarray]:
    """A VARLiNGAM stand-in with a known AR stack (1 instantaneous + 2 lag blocks, n=3)."""
    n = 3
    ar = np.zeros((3, n, n))
    ar[0, 1, 0] = 0.6  # lag-0 (instantaneous): edge 0 -> 1
    ar[1, 2, 1] = -0.4  # lag-1: edge 1 -> 2
    ar[1, 0, 0] = 0.5  # lag-1: autoregressive self-loop (kept in topology)
    ar[2, 1, 0] = 0.3  # lag-2: edge 0 -> 1 ...
    ar[2, 0, 1] = 0.2  # lag-2: ... and edge 1 -> 0 => both directions => a 2-cycle (digraph)
    model = SimpleNamespace(adjacency_matrices_=ar, causal_order_=[0, 1, 2])
    return model, ar


def test_var_lingam_recovers_lag_topology():
    model, ar = _var_model()
    out = _adapt_var_lingam(model)
    assert isinstance(out, StructureOutput)
    t = out.structure
    assert isinstance(t, TemporalStructure)
    assert t.n_lags == 3
    assert t.n_nodes == 3
    # lag-0 is the instantaneous DAG view.
    assert t.lag(0).kind == "dag"
    assert t.lag(0).endpoints(0, 1) == (TAIL, ARROW)
    # every lag>=1 window is a directed graph allowing 2-cycles and self-loops (digraph), even a
    # lag whose only extra mark is a self-loop -- the kind is a property of the producer, not data.
    assert t.lag(1).kind == "digraph"
    # lag-1 keeps the directed edge AND the autoregressive self-loop.
    assert t.lag(1).endpoints(1, 2) == (TAIL, ARROW)
    assert t.lag(1).endpoints(0, 0) == (ARROW, ARROW)  # self-loop 0 -> 0 survives
    assert np.array_equal(np.diagonal(t.lag(1).to_numpy()), np.array([ARROW, 0, 0], dtype=np.int8))
    # lag-2 carries both directions of 0-1 => a 2-cycle, not a PAG's bidirected (confounded) edge.
    assert t.lag(2).kind == "digraph"
    assert t.lag(2).endpoints(0, 1) == (ARROW, ARROW)


def test_var_lingam_stack_and_ordering():
    model, ar = _var_model()
    out = _adapt_var_lingam(model)
    # lag_weights is the canonical (B.T) AR stack; MA absent; ordering = causal_order_.
    assert np.array_equal(out.structure.lag_weights, ar.transpose(0, 2, 1))
    assert out.structure.lag_weights_ma is None
    assert out.ordering == (0, 1, 2)
    assert out.metadata["algorithm"] == "varlingam"


def test_var_lingam_json_roundtrip(tmp_path):
    model, _ = _var_model()
    out = _adapt_var_lingam(model)
    p = tmp_path / "var.json"
    out.save(p)
    assert StructureOutput.load(p) == out


def _varma_model() -> tuple[SimpleNamespace, np.ndarray, np.ndarray]:
    """A VARMALiNGAM stand-in: ``adjacency_matrices_`` is a (psis, omegas) tuple."""
    n = 3
    psis = np.zeros((2, n, n))  # AR order p=1 (index 0 instantaneous, index 1 lag-1)
    psis[0, 1, 0] = 0.6  # instantaneous edge 0 -> 1
    psis[1, 2, 1] = -0.4  # lag-1 edge 1 -> 2
    omegas = np.zeros((2, n, n))  # MA order q=2, independent of the AR order
    omegas[0, 1, 2] = 0.2
    omegas[1, 0, 2] = -0.1
    model = SimpleNamespace(adjacency_matrices_=(psis, omegas), causal_order_=[2, 0, 1])
    return model, psis, omegas


def test_varma_lingam_dual_stack():
    model, psis, omegas = _varma_model()
    out = _adapt_varma_lingam(model)
    t = out.structure
    assert isinstance(t, TemporalStructure)
    # The lag topology comes from psis (AR) only; both stacks are present and lossless.
    assert t.n_lags == psis.shape[0]
    assert np.array_equal(t.lag_weights, psis.transpose(0, 2, 1))  # canonical B.T
    assert np.array_equal(t.lag_weights_ma, omegas.transpose(0, 2, 1))
    assert t.lag(0).endpoints(0, 1) == (TAIL, ARROW)
    assert t.lag(1).endpoints(1, 2) == (TAIL, ARROW)
    assert out.ordering == (2, 0, 1)
    assert out.metadata["algorithm"] == "varmalingam"


def test_varma_lingam_json_roundtrip(tmp_path):
    model, _, omegas = _varma_model()
    out = _adapt_varma_lingam(model)
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
    out = _adapt_granger(coeff, n_vars=3, maxlag=2)
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
    assert out.metadata["algorithm"] == "granger"


def test_granger_json_roundtrip(tmp_path):
    out = _adapt_granger(_granger_coeff(), n_vars=3, maxlag=2)
    p = tmp_path / "granger.json"
    out.save(p)
    assert StructureOutput.load(p) == out


def test_granger_rejects_bad_shape():
    with pytest.raises(ValueError, match="shape"):
        _adapt_granger(np.zeros((3, 5)), n_vars=3, maxlag=2)  # 5 != d*maxlag = 6


def test_orientation_invariant_weights_match_edges():
    """Canonical ``lag_weights[pos][i, j] != 0`` <=> an arrowhead into j on edge i-j.

    Holds for directed (i -> j), a 2-cycle (both ways), and lag>=1 self-loops (i -> i) alike, so
    the stored weights and the derived graph share one orientation.
    """
    model, _ = _var_model()
    t = _adapt_var_lingam(model).structure
    assert isinstance(t, TemporalStructure)
    weights = t.lag_weights
    for pos, lag_value in enumerate(t.lags):
        g = t.lag(lag_value)
        for i in range(t.n_nodes):
            for j in range(t.n_nodes):
                has_weight = weights[pos, i, j] != 0.0
                arrowhead_at_j = g.endpoints(i, j)[1] == ARROW
                assert has_weight == arrowhead_at_j, (pos, i, j)


def _longitudinal_model() -> SimpleNamespace:
    """A LongitudinalLiNGAM stand-in with ``adjacency_matrices_`` of shape (T, 1+n_lags, p, p)
    and NaN blocks."""
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
    return SimpleNamespace(adjacency_matrices_=adj, causal_orders_=orders)


def test_longitudinal_adapter_stores_the_time_axis_losslessly():
    out = _adapt_longitudinal_lingam(_longitudinal_model())
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
    assert out.metadata["algorithm"] == "longitudinallingam"


def test_longitudinal_adapter_json_roundtrip(tmp_path):
    out = _adapt_longitudinal_lingam(_longitudinal_model())
    p = tmp_path / "longitudinal.json"
    out.save(p)
    assert StructureOutput.load(p) == out  # (time, lag) axis + NaN weights round-trip


def test_longitudinal_adapter_masks_wraparound_and_zero_fills_the_lag_view():
    # A lag block before the panel start (k > t) is uncomputable. The adapter masks any
    # coefficients there with NaN; the lag-only view zero-fills them. With T=2 and n_lags=2,
    # the last occasion's lag-2 block is uncomputable.
    p = 2
    adj = np.full((2, 1 + 2, p, p), np.nan)
    adj[1, 0] = np.zeros((p, p))
    adj[1, 0, 1, 0] = 0.5  # instantaneous 0 -> 1
    adj[1, 1] = np.zeros((p, p))
    adj[1, 1, 1, 0] = 0.3  # lag-1 (reaches occasion 0, valid)
    adj[1, 2] = np.full((p, p), 0.9)  # lag-2 reaches t=-1: wrap-around garbage
    t = _adapt_longitudinal_lingam(
        SimpleNamespace(adjacency_matrices_=adj, causal_orders_=[[], [0, 1]])
    ).structure
    assert t.n_times == 1 and t.times == (1,)
    assert np.isnan(t.time_weights[0, 2]).all()  # the garbage block is masked to NaN (lossless)
    assert int(t.at(1, 2).to_numpy().sum()) == 0  # its graph is empty, not a dense garbage digraph
    # the lag-only view zero-fills the NaN block and stays finite.
    assert np.isfinite(t.lag_weights).all()
    assert np.array_equal(t.lag_weights, np.nan_to_num(t.time_weights[0]))
