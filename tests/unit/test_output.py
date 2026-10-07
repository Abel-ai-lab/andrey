"""Round-trip and contract tests for the ``StructureOutput`` run envelope."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.core import GraphStructure, StructureOutput
from andrey.core.structure import ARROW, CIRCLE, TAIL


def dag(pairs, n, labels=None):
    M = np.zeros((n, n), dtype=np.int8)
    for i, j in pairs:
        M[i, j], M[j, i] = TAIL, ARROW
    return GraphStructure.from_numpy(M, kind="dag", labels=labels)


def lingam_output():
    g = dag([(0, 1), (1, 2)], 3)
    W = np.zeros((3, 3))
    W[1, 0], W[2, 1] = 0.8, -1.3  # edge j -> i convention
    return StructureOutput.new(
        g, ordering=[0, 1, 2], weighted_adjacency=W, metadata={"algorithm": "direct_lingam"}
    )


def test_weighted_adjacency_roundtrips_dense():
    W = np.zeros((3, 3))
    W[1, 0], W[2, 1] = 0.8, -1.3
    out = lingam_output()
    assert np.array_equal(out.weighted_adjacency, W)
    assert out.weighted_adjacency.dtype == np.float64
    assert np.array_equal(out.to_scipy_sparse().toarray(), W)
    assert out.ordering == (0, 1, 2)


def test_no_weights_paths():
    out = StructureOutput.new(dag([(0, 1)], 2), metadata={"score": -42.0})
    assert out.weighted_adjacency is None
    assert out.ordering is None
    with pytest.raises(ValueError):
        out.to_scipy_sparse()


def test_weighted_shape_validated():
    with pytest.raises(ValueError):
        StructureOutput.new(dag([(0, 1)], 2), weighted_adjacency=np.zeros((3, 3)))


def test_new_rejects_non_permutation_ordering():
    # new() validates up front, so it never writes an envelope load() would reject.
    with pytest.raises(ValueError, match="permutation"):
        StructureOutput.new(dag([(0, 1)], 2), ordering=[0, 0])


def test_to_scipy_sparse_weights_false_raises():
    with pytest.raises(ValueError):
        lingam_output().to_scipy_sparse(weights=False)


def test_nan_weights_are_reflexive_and_roundtrip(tmp_path):
    W = np.zeros((3, 3))
    W[1, 0] = np.nan
    out = StructureOutput.new(dag([(0, 1), (1, 2)], 3), weighted_adjacency=W)
    assert out == out  # equal_nan: a NaN-bearing result stays equal to itself
    p = tmp_path / "nan.json"
    out.save(p)
    assert StructureOutput.load(p) == out


def test_metadata_is_readonly_view():
    out = StructureOutput.new(dag([(0, 1)], 2), metadata={"score": 1.0})
    meta = out.metadata
    with pytest.raises(TypeError):
        meta["score"] = 2.0  # MappingProxyType is read-only


@pytest.mark.parametrize(
    "bad",
    [
        {"x": np.int64(3)},
        {"x": np.float32(3.0)},
        {"x": np.zeros(3)},
        {"x": {1, 2}},
        {"x": (1, 2)},
        {1: "non-str-key"},
    ],
)
def test_metadata_json_safety_rejected(bad):
    with pytest.raises(TypeError):
        StructureOutput.new(dag([(0, 1)], 2), metadata=bad)


def test_metadata_json_safe_values_accepted():
    ok = {"s": "a", "b": True, "i": 3, "f": 1.5, "n": None, "lst": [1, 2.0, "x"], "d": {"k": 1}}
    out = StructureOutput.new(dag([(0, 1)], 2), metadata=ok)
    assert dict(out.metadata) == ok


@pytest.mark.parametrize("fmt", ["json", "npz"])
def test_save_load_roundtrip_graph(tmp_path, fmt):
    out = lingam_output()
    p = tmp_path / f"out.{fmt}"
    out.save(p, fmt=fmt)
    assert StructureOutput.load(p) == out


@pytest.mark.parametrize("fmt", ["json", "npz"])
def test_save_load_roundtrip_pag_labeled(tmp_path, fmt):
    M = np.zeros((3, 3), dtype=np.int8)
    M[0, 1], M[1, 0] = CIRCLE, ARROW
    M[1, 2], M[2, 1] = ARROW, ARROW
    g = GraphStructure.from_numpy(M, kind="pag", labels=("a", "b", "c"))
    out = StructureOutput.new(g, metadata={"score": 0.0})
    p = tmp_path / f"pag.{fmt}"
    out.save(p, fmt=fmt)
    back = StructureOutput.load(p)
    assert back == out
    assert back.structure.labels == ("a", "b", "c")


def test_npz_requires_npz_suffix(tmp_path):
    # np.savez would silently append .npz; save() rejects a mismatched suffix up front.
    with pytest.raises(ValueError, match="npz"):
        lingam_output().save(tmp_path / "out.bin", fmt="npz")


def test_load_rejects_weighted_shape_mismatch(tmp_path):
    import json

    p = tmp_path / "w.json"
    lingam_output().save(p)
    doc = json.loads(p.read_text())
    doc["weighted"]["indptr"] = doc["weighted"]["indptr"][:-1]  # corrupt the CSR indptr length
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        StructureOutput.load(p)


def test_load_rejects_unknown_format_version(tmp_path):
    import json

    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"format_version": 999, "structure_type": "graph"}))
    with pytest.raises(ValueError, match="format_version"):
        StructureOutput.load(p)


def test_load_validates_ordering_permutation(tmp_path):
    out = lingam_output()
    p = tmp_path / "o.json"
    out.save(p)
    import json

    doc = json.loads(p.read_text())
    doc["ordering"] = [0, 1, 1]  # not a permutation
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="permutation"):
        StructureOutput.load(p)


def test_npz_load_is_allow_pickle_free(tmp_path):
    # A crafted object-array npz must raise on load, never execute a pickle.
    p = tmp_path / "evil.npz"
    np.savez(p, _json=np.array({"payload": 1}, dtype=object))
    with pytest.raises((ValueError, KeyError)):
        StructureOutput.load(p)


def test_equality():
    assert lingam_output() == lingam_output()
    a = StructureOutput.new(dag([(0, 1)], 2))
    b = StructureOutput.new(dag([(1, 0)], 2))
    assert a != b
    assert a.__hash__ is None


def test_save_rejects_bad_fmt(tmp_path):
    with pytest.raises(ValueError):
        lingam_output().save(tmp_path / "x.bin", fmt="bin")  # type: ignore[arg-type]


def test_repr_heads_with_the_algorithm_and_adds_order_weights_and_metadata():
    M = np.zeros((3, 3), dtype=np.int8)
    M[0, 1], M[1, 0] = TAIL, ARROW  # a -> b
    M[1, 2], M[2, 1] = TAIL, ARROW  # b -> c
    g = GraphStructure.from_numpy(M, kind="dag", labels=("a", "b", "c"))
    W = np.zeros((3, 3))
    W[0, 1], W[1, 2] = 0.8, -1.25
    out = StructureOutput.new(
        g,
        ordering=(0, 1, 2),
        weighted_adjacency=W,
        metadata={"algorithm": "DirectLiNGAM", "score": 1.23456},
    )
    assert repr(out) == (
        "DirectLiNGAM  dag  |  3 nodes  |  2 edges (2 directed)\n"
        "\n"
        "  a -> b  (+0.8)\n"
        "  b -> c  (-1.25)\n"
        "\n"
        "  order: a -> b -> c\n"
        "\n"
        "  score: 1.235"
    )
    assert str(out) == repr(out)


def test_repr_two_cycles_use_each_direction_weight():
    M = np.array([[0, ARROW], [ARROW, 0]], dtype=np.int8)
    g = GraphStructure.from_numpy(M, kind="digraph", labels=("a", "b"))
    out = StructureOutput.new(g, weighted_adjacency=np.array([[0, 0.8], [-1.25, 0]]))
    assert repr(out).splitlines() == [
        "digraph  |  2 nodes  |  2 edges (2 directed)",
        "",
        "  a -> b  (+0.8)",
        "  b -> a  (-1.25)",
    ]
