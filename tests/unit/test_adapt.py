"""Shared adapter tests for adjacency, metadata coercion, and result envelopes."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.api import _adapt
from andrey.core import ARROW, CIRCLE, TAIL, GraphStructure, StructureOutput

# --- json_safe -----------------------------------------------------------------


def test_json_safe_coerces_numpy_scalars_and_arrays():
    assert isinstance(_adapt.json_safe(np.int64(3)), int)
    assert _adapt.json_safe(np.int64(3)) == 3
    assert isinstance(_adapt.json_safe(np.float64(1.5)), float)
    assert _adapt.json_safe(np.bool_(True)) is True
    assert _adapt.json_safe(np.array([1, 2])) == [1, 2]
    assert _adapt.json_safe({"a": np.float32(2.0), "b": [np.int8(1)]}) == {"a": 2.0, "b": [1]}


def test_json_safe_rejects_unserializable():
    with pytest.raises(TypeError):
        _adapt.json_safe(object())
    with pytest.raises(TypeError):
        _adapt.json_safe({"s": {1, 2}})  # a set has no JSON-safe form


# --- structure_output ----------------------------------------------------------


def test_structure_output_graph_only():
    out = _adapt.structure_output(
        GraphStructure.from_numpy(np.array([[0, TAIL], [ARROW, 0]]), kind="cpdag")
    )
    assert isinstance(out, StructureOutput)
    assert out.ordering is None and out.weighted_adjacency is None


def test_structure_output_lingam_shape():
    # DAG + causal order + weighted B (j->i), the LiNGAM adapter's envelope.
    g = GraphStructure.from_numpy(np.array([[0, TAIL], [ARROW, 0]]), kind="dag")  # 0 -> 1
    B = np.array([[0.0, 0.0], [0.7, 0.0]])  # x1 = 0.7 x0  -> weight of edge 0->1 at B[1, 0]
    out = _adapt.structure_output(
        g, ordering=np.array([0, 1]), weighted_adjacency=B, metadata={"score": np.float64(1.5)}
    )
    assert out.ordering == (0, 1)
    assert np.allclose(out.weighted_adjacency, B)
    assert isinstance(out.metadata["score"], float)
    assert out.metadata["score"] == 1.5


def test_structure_output_metadata_coerced_json_safe():
    g = GraphStructure.from_numpy(np.array([[0, TAIL], [ARROW, 0]]), kind="dag")
    out = _adapt.structure_output(g, metadata={"score": np.float64(2.0), "p": np.array([0.1, 0.2])})
    # numpy leaking into metadata would raise on save; coercion makes the envelope round-trip.
    assert out.metadata == {"score": 2.0, "p": [0.1, 0.2]}


def test_structure_output_rejects_inconsistent_kind():
    # Circle marks tagged dag: envelope validation (StructureOutput.new) rejects it.
    S = np.array([[0, CIRCLE], [ARROW, 0]])  # 0 o-> 1 (circle at 0), requires pag
    with pytest.raises(ValueError):
        _adapt.structure_output(GraphStructure.from_numpy(S, kind="dag"))


def test_canonical_weights_transposes_coefficients():
    """``_canonical_weights`` maps coefficient ``B[i,j]=j->i`` to Andrey ``W[i,j]=i->j`` (W = B.T).

    A standalone pin (W = B.T) so the LiNGAM round-trips cannot be fooled by a matched-pair
    convention flip -- an adapter and the fixture decode changed together -- since this fixes the
    orientation independently of both.
    """
    from andrey.api.lingam import _canonical_weights

    B = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, -3.0, 0.0]])  # B[1,0]: 0->1; B[2,1]: 1->2
    W = _canonical_weights(B)
    assert np.array_equal(W, B.T)
    assert W[0, 1] == 2.0 and W[1, 2] == -3.0  # Andrey W[i, j] = edge i -> j
