"""Round-trip recorded outputs through the fixture helpers without running an engine."""

from __future__ import annotations

import types

import numpy as np
import pytest

from andrey import GraphStructure
from andrey.api import _adapt
from andrey.api._temporal import _adapt_var_lingam, _adapt_varma_lingam
from andrey.api.lingam import _canonical_weights
from tests.recovery.helpers import harness
from tests.recovery.helpers.slice_fixture import assert_recorded_output, project_output


def _graph_out(baseline_graph: object, kind: str, **meta: object):
    """A native-shaped StructureOutput from a baseline unsigned endpoint matrix (+ metadata)."""
    gs = GraphStructure.from_numpy(np.asarray(baseline_graph, dtype=np.int8), kind=kind)
    return _adapt.structure_output(gs, metadata=meta or None)


def test_pc_roundtrip():
    doc = harness.load_baseline("PC", "gauss_5v")
    out = _graph_out(doc["output"]["graph"], "cpdag")
    assert_recorded_output(out, "PC", case="gauss_5v")  # must not raise


def test_ges_roundtrip_graph_and_score():
    doc = harness.load_baseline("GES", "gauss_5v")
    out = _graph_out(doc["output"]["graph"], "cpdag", score=doc["output"]["score"])
    assert_recorded_output(out, "GES", case="gauss_5v")


def test_direct_lingam_roundtrip():
    doc = harness.load_baseline("DirectLiNGAM", "lingam_5v_uniform")
    baseline_b = np.asarray(doc["output"]["weighted_adjacency"], dtype=float)  # B[i, j] = j -> i
    native_w = _canonical_weights(baseline_b)  # W[i, j] = i -> j
    gs = _adapt.dag_from_adjacency(native_w)
    out = _adapt.structure_output(
        gs, ordering=doc["output"]["causal_order"], weighted_adjacency=native_w
    )
    assert_recorded_output(out, "DirectLiNGAM", case="lingam_5v_uniform")


# --- temporal / multi-group / pairwise: build a native output through the PRODUCTION encode path
# (the duck-typed temporal adapters, the _adapt LiNGAM helpers) from each baseline, then round-trip.
# Covers the projection families the graph/LiNGAM tests above do not.


def test_var_lingam_roundtrip():
    doc = harness.load_baseline("VARLiNGAM", "var_4v_stable")
    model = types.SimpleNamespace(
        adjacency_matrices_=np.asarray(doc["output"]["adjacency_matrices"], dtype=float),
        causal_order_=doc["output"]["causal_order"],
    )
    out = _adapt_var_lingam(model)
    assert_recorded_output(out, "VARLiNGAM", case="var_4v_stable")


def test_varma_lingam_roundtrip():
    doc = harness.load_baseline("VARMALiNGAM", "varma_3v")
    model = types.SimpleNamespace(
        adjacency_matrices_=(
            np.asarray(doc["output"]["psis"], dtype=float),
            np.asarray(doc["output"]["omegas"], dtype=float),
        ),
        causal_order_=doc["output"]["causal_order"],
    )
    out = _adapt_varma_lingam(model)
    assert_recorded_output(out, "VARMALiNGAM", case="varma_3v")


def test_multi_group_roundtrip():
    doc = harness.load_baseline("MultiGroupDirectLiNGAM", "mgdl_2groups")
    order = doc["output"]["causal_order"]
    outs = []
    for baseline_b in np.asarray(doc["output"]["group_weighted_adjacency"], dtype=float):
        native_w = _canonical_weights(baseline_b)
        gs = _adapt.dag_from_adjacency(native_w)
        outs.append(_adapt.structure_output(gs, ordering=order, weighted_adjacency=native_w))
    assert_recorded_output(outs, "MultiGroupDirectLiNGAM", case="mgdl_2groups")


def test_score_mismatch_fails():
    doc = harness.load_baseline("GES", "gauss_5v")
    bad = _graph_out(doc["output"]["graph"], "cpdag", score=float(doc["output"]["score"]) + 1.0)
    with pytest.raises(AssertionError):
        assert_recorded_output(bad, "GES", case="gauss_5v")


def test_structural_mismatch_fails():
    # Remove both endpoints of an edge so the CPDAG remains valid but differs from the baseline.
    doc = harness.load_baseline("PC", "gauss_5v")
    g = np.asarray(doc["output"]["graph"], dtype=np.int8).copy()
    i, j = (int(v) for v in np.argwhere(g != 0)[0])
    g[i, j] = 0
    g[j, i] = 0
    out = _graph_out(g, "cpdag")
    with pytest.raises(AssertionError):
        assert_recorded_output(out, "PC", case="gauss_5v")


def test_unregistered_algorithm_raises():
    with pytest.raises(KeyError):
        project_output(object(), "Granger")
