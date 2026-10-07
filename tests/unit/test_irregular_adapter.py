"""Adapter tests for the irregular / latent-confounder family (``andrey.api._irregular``).

The fitted models are duck-typed, so ``SimpleNamespace`` / raw numpy stand-ins drive
the same path. Every metadata payload is proven JSON-safe by round-tripping the whole
``StructureOutput`` through ``save`` / ``load``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from andrey.api._irregular import (
    _adapt_anm,
    _adapt_bottom_up_parce,
    _adapt_camuv,
    _adapt_pnl,
    _adapt_rcd,
)
from andrey.core import ARROW, CIRCLE, NULL, TAIL, GraphStructure, StructureOutput


def _roundtrips(out: StructureOutput, tmp_path, name: str) -> StructureOutput:
    """Save + reload the envelope (fails if any metadata value is not JSON-safe); return it back."""
    p = tmp_path / f"{name}.json"
    out.save(p)
    back = StructureOutput.load(p)
    assert back == out
    return back


# ---- ANM / PNL ----------------------------------------------------------------------------------


# (pval_forward, pval_backward, alpha) -> the edge, as (source, target); None for no edge.
DECISIONS = [
    (0.60, 0.30, None, (0, 1)),  # the larger p-value wins
    (0.30, 0.60, None, (1, 0)),
    (0.40, 0.40, None, None),  # a tie decides nothing
    (0.23, 0.02, 0.05, (0, 1)),  # forward passes, backward fails
    (0.02, 0.23, 0.05, (1, 0)),
    (0.60, 0.30, 0.05, None),  # both pass: either direction fits
    (0.01, 0.02, 0.05, None),  # neither passes: the model does not fit
    (0.05, 0.01, 0.05, None),  # a p-value at alpha fails its test
]


@pytest.mark.parametrize(("forward", "backward", "alpha", "edge"), DECISIONS)
def test_pairwise_decision_rules(forward, backward, alpha, edge):
    out = _adapt_pnl(forward, backward, alpha=alpha)
    expected = [] if edge is None else [(*edge, "directed")]
    assert out.structure.oriented_edges() == expected
    assert _adapt_anm(forward, backward, alpha=alpha).structure == out.structure


def test_anm_decided_graph_and_float_metadata(tmp_path):
    out = _adapt_anm(0.02, 0.81)
    g = out.structure
    assert isinstance(g, GraphStructure)
    assert g.n_nodes == 2
    assert g.kind == "dag"
    assert g.oriented_edges() == [(1, 0, "directed")]  # the backward model fits better
    assert out.weighted_adjacency is None
    md = out.metadata
    assert md["algorithm"] == "ANM"
    assert md["pval_forward"] == 0.02
    assert md["pval_backward"] == 0.81
    assert isinstance(md["pval_forward"], float) and isinstance(md["pval_backward"], float)
    _roundtrips(out, tmp_path, "anm")


def test_pnl_coerces_shape_one_arrays(tmp_path):
    # PNL returns each p-value as a shape-(1,) numpy array; the adapter flattens to Python float.
    out = _adapt_pnl(np.array([0.013]), np.array([0.47]))
    md = out.metadata
    assert md["pval_forward"] == 0.013
    assert md["pval_backward"] == 0.47
    assert type(md["pval_forward"]) is float  # not np.float64 (JSON-safe)
    assert out.structure.n_nodes == 2
    assert out.structure.oriented_edges() == [(1, 0, "directed")]
    assert md["algorithm"] == "PNL"
    _roundtrips(out, tmp_path, "pnl")


# ---- CAMUV --------------------------------------------------------------------------------------


def test_camuv_directed_and_bidirected(tmp_path):
    # var 1 has parent 0 (edge 0 -> 1); vars 1 and 2 share an unobserved confounder.
    P = [set(), {0}, set()]
    U = [{2, 1}]  # unsorted on purpose -> normalized to [1, 2]
    out = _adapt_camuv(P, U, n_vars=3)
    g = out.structure
    assert g.kind == "pag"
    assert g.endpoints(0, 1) == (TAIL, ARROW)  # directed 0 -> 1
    assert g.endpoints(1, 2) == (ARROW, ARROW)  # bidirected confounder
    assert g.endpoints(0, 2) == (NULL, NULL)
    assert out.weighted_adjacency is None
    assert out.metadata["confounded_pairs"] == [[1, 2]]
    _roundtrips(out, tmp_path, "camuv")


def test_camuv_no_confounders_is_pure_directed():
    P = [set(), {0}, {1}]  # chain 0 -> 1 -> 2, no latent confounder
    out = _adapt_camuv(P, U=[], n_vars=3)
    g = out.structure
    assert g.endpoints(0, 1) == (TAIL, ARROW)
    assert g.endpoints(1, 2) == (TAIL, ARROW)
    assert out.metadata["confounded_pairs"] == []


# ---- RCD ----------------------------------------------------------------------------------------


def test_rcd_nan_becomes_bidirected(tmp_path):
    B = np.zeros((3, 3))
    B[1, 0] = 0.9  # finite: edge 0 -> 1
    B[0, 2] = np.nan  # NaN sentinel: 0 and 2 share a latent confounder
    B[2, 0] = np.nan
    out = _adapt_rcd(SimpleNamespace(adjacency_matrix_=B))
    g = out.structure
    assert g.kind == "pag"
    assert g.endpoints(0, 1) == (TAIL, ARROW)
    assert g.endpoints(0, 2) == (ARROW, ARROW)  # bidirected confounder from NaN
    W = out.weighted_adjacency
    assert W[0, 1] == 0.9  # canonical (row-is-source): edge 0 -> 1, matching endpoints(0, 1)
    assert W[0, 2] == 0.0 and W[2, 0] == 0.0  # NaN replaced by 0
    assert not np.isnan(W).any()
    _roundtrips(out, tmp_path, "rcd")


def test_rcd_asymmetric_single_nan_cell():
    # RCD may write the sentinel in only one cell of the pair; still one bidirected edge.
    B = np.zeros((2, 2))
    B[0, 1] = np.nan
    out = _adapt_rcd(SimpleNamespace(adjacency_matrix_=B))
    assert out.structure.endpoints(0, 1) == (ARROW, ARROW)


# ---- BottomUpParceLiNGAM ------------------------------------------------------------------------


def test_bottom_up_nan_becomes_circle_and_nested_order(tmp_path):
    B = np.zeros((3, 3))
    B[1, 0] = 0.5  # finite: edge 0 -> 1
    B[1, 2] = np.nan  # symmetric NaN block: order between 1 and 2 unknown
    B[2, 1] = np.nan
    model = SimpleNamespace(adjacency_matrix_=B, causal_order_=[0, [1, 2]])
    out = _adapt_bottom_up_parce(model)
    g = out.structure
    assert g.kind == "pag"
    assert g.endpoints(0, 1) == (TAIL, ARROW)
    assert g.endpoints(1, 2) == (CIRCLE, CIRCLE)  # order-unknown circle
    W = out.weighted_adjacency
    assert W[0, 1] == 0.5  # canonical B.T: edge 0 -> 1
    assert W[1, 2] == 0.0 and W[2, 1] == 0.0
    # Nested causal order cannot be a flat ordering permutation -> metadata only.
    assert out.ordering is None
    assert out.metadata["causal_order"] == [0, [1, 2]]
    _roundtrips(out, tmp_path, "bottom_up")


def test_bottom_up_numpy_int_order_is_json_safe(tmp_path):
    # causal_order_ often holds numpy ints / a plain fully-ordered list; both must serialize.
    B = np.zeros((2, 2))
    B[1, 0] = 1.2
    model = SimpleNamespace(adjacency_matrix_=B, causal_order_=[np.int64(0), np.int64(1)])
    out = _adapt_bottom_up_parce(model)
    assert out.metadata["causal_order"] == [0, 1]
    assert all(type(x) is int for x in out.metadata["causal_order"])
    _roundtrips(out, tmp_path, "bottom_up_flat")
