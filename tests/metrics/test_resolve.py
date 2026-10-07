"""score(): capability resolution, the flat result schema, and loud validation."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.core import ARROW, LATENT, OBSERVED, GraphStructure, StructureOutput, TemporalStructure
from andrey.metrics import score

CHAIN = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]  # 0 -> 1 -> 2
FORK = [[0, 0, 0], [1, 0, 1], [0, 0, 0]]  # MEC-equivalent to CHAIN


def test_dag_family_reports_directed_and_mec(dag_of):
    result = score(dag_of(CHAIN), dag_of(FORK))
    assert result["family"] == "dag"
    assert result["shd"] == 1  # directed graphs differ by one reversal
    assert result["mec_shd"] == 0  # but they share a Markov equivalence class
    assert result["skeleton_f1"] == 1.0
    assert result["arrowhead_f1"] < 1.0
    assert result["mec_arrowhead_f1"] == 1.0


def test_cpdag_family_canonicalizes(dag_of):
    from andrey.metrics import to_cpdag

    # a DAG scored against its own CPDAG lands in the cpdag family and matches exactly.
    result = score(dag_of(CHAIN), to_cpdag(CHAIN))
    assert result["family"] == "cpdag"
    assert result["shd"] == 0
    assert result["skeleton_f1"] == 1.0


def test_pag_family_has_endpoint_and_confounder_keys():
    pag = GraphStructure.from_numpy(np.array([[0, 2], [2, 0]], dtype=np.int8), kind="pag")
    result = score(pag, pag)
    assert result["family"] == "pag"
    assert "shd_endpoint" in result
    assert result["confounder_f1"] == 1.0


def test_digraph_family_scores_base_trio_only():
    # a directed-cyclic graph (2-cycle 0 <=> 1) scores on skeleton / arrowhead / shd alone: it has
    # no essential graph (no mec pair) and makes no confounder claim (no pag extras).
    marks = np.array([[0, 2], [2, 0]], dtype=np.int8)  # ARROW/ARROW read as a 2-cycle under digraph
    g = GraphStructure.from_numpy(marks, kind="digraph")
    result = score(g, g)
    assert result["family"] == "digraph"
    assert result["skeleton_f1"] == 1.0
    assert result["arrowhead_f1"] == 1.0
    assert result["shd"] == 0
    assert "mec_shd" not in result  # a directed cycle has no essential graph
    assert "confounder_f1" not in result  # nor a latent-confounder claim


def test_digraph_family_from_truth_side_and_scores_self_loops():
    # precedence holds from the truth side too: a dag estimate vs a cyclic (digraph) truth still
    # routes to the digraph family (k_true == "digraph"), with no MEC pair.
    dag = GraphStructure.from_numpy(np.array([[0, 1], [2, 0]], dtype=np.int8), kind="dag")  # 0 -> 1
    two_cycle = GraphStructure.from_numpy(np.array([[0, 2], [2, 0]], dtype=np.int8), kind="digraph")
    result = score(dag, two_cycle)
    assert result["family"] == "digraph"
    assert "mec_shd" not in result

    # two digraphs differing only by a self-loop must not score shd == 0: the digraph family
    # counts the diagonal (self-loops are real structure), unlike the off-diagonal-only trio.
    with_loop = np.array([[ARROW, 2], [2, 0]], dtype=np.int8)  # 2-cycle 0 <=> 1 plus self-loop 0->0
    g_loop = GraphStructure.from_numpy(with_loop, kind="digraph", allow_self_loops=True)
    g_bare = GraphStructure.from_numpy(np.array([[0, 2], [2, 0]], dtype=np.int8), kind="digraph")
    assert g_loop != g_bare
    assert score(g_loop, g_bare)["shd"] == 1  # the self-loop insertion is one structural edit


def test_ordering_and_weights_are_capability_gated(dag_of):
    plain = dag_of(CHAIN)  # a bare graph: no ordering, no weights
    result = score(plain, plain)
    assert "order_accuracy" not in result
    assert "coefficient_mae" not in result

    w = np.array([[0, 1.0, 0], [0, 0, 2.0], [0, 0, 0]])
    rich = StructureOutput.new(dag_of(CHAIN), ordering=[0, 1, 2], weighted_adjacency=w)
    result = score(rich, rich)
    assert result["order_accuracy"] == 1.0
    assert result["coefficient_mae"] == 0.0


def test_temporal_family():
    g = GraphStructure.from_numpy(np.array([[0, 1], [2, 0]], dtype=np.int8), kind="dag")
    t = TemporalStructure.from_lag_graphs([g], lags=[0])
    result = score(t, t)
    assert result["family"] == "temporal"
    assert result["summary"]["f1"] == 1.0


def test_latent_family_preempts_graph_metrics():
    marks = np.array([[0, 0, 2], [0, 0, 2], [1, 1, 0]], dtype=np.int8)  # latent 2 -> {0, 1}
    types = np.array([OBSERVED, OBSERVED, LATENT], dtype=np.int8)
    g = GraphStructure.from_numpy(marks, kind="dag", node_types=types)
    result = score(g, g)
    assert result["family"] == "latent"
    assert result["latent_cluster_ari"] == 1.0
    assert result["n_latents_true"] == 1
    assert "shd" not in result  # graph metrics are not attempted for a latent structure


def test_latent_single_observed_scores_are_json_safe():
    # 2-node latent graph (1 observed, 1 latent): ARI over one node must be 1.0, not nan.
    import json

    marks = np.array([[0, 2], [1, 0]], dtype=np.int8)  # latent 1 -> observed 0
    types = np.array([OBSERVED, LATENT], dtype=np.int8)
    g = GraphStructure.from_numpy(marks, kind="dag", node_types=types)
    result = score(g, g)
    assert result["latent_cluster_ari"] == 1.0
    json.dumps(result)  # nan would break strict JSON


def test_cyclic_input_keeps_directed_scores_and_omits_mec():
    # a thresholded cyclic estimate has no CPDAG: score() still returns the directed metrics.
    cycle = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # 0 -> 1 -> 2 -> 0
    result = score(cycle, cycle)
    assert result["family"] == "dag"
    assert result["shd"] == 0
    assert result["skeleton_f1"] == 1.0
    assert "mec_shd" not in result  # canonicalization skipped, not crashed


def test_result_is_json_safe(dag_of):
    import json

    w = np.array([[0, 1.0], [0, 0]])
    marks = np.array([[0, 1], [2, 0]], dtype=np.int8)
    rich = StructureOutput.new(
        GraphStructure.from_numpy(marks, kind="dag"), ordering=[0, 1], weighted_adjacency=w
    )
    json.dumps(score(rich, rich))  # raises if any value is a numpy scalar / non-serializable


# ---- validation ---------------------------------------------------------------------------------


def test_node_count_mismatch_raises():
    with pytest.raises(ValueError, match="node-count mismatch"):
        score(np.zeros((3, 3), dtype=int), np.zeros((2, 2), dtype=int))


def test_temporal_vs_graph_mismatch_raises():
    g = GraphStructure.from_numpy(np.array([[0, 1], [2, 0]], dtype=np.int8), kind="dag")
    t = TemporalStructure.from_lag_graphs([g], lags=[0])
    with pytest.raises(ValueError, match="temporal"):
        score(t, g)


def test_digraph_vs_pag_raises():
    # a digraph 2-cycle and a PAG bidirected edge share ARROW/ARROW marks but mean opposite things
    # (a directed 2-cycle vs a latent confounder); the pair has no meaningful comparison family.
    marks = np.array([[0, ARROW], [ARROW, 0]], dtype=np.int8)
    digraph = GraphStructure.from_numpy(marks, kind="digraph")
    pag = GraphStructure.from_numpy(marks, kind="pag")
    with pytest.raises(ValueError, match="digraph to a PAG"):
        score(digraph, pag)
    with pytest.raises(ValueError, match="digraph to a PAG"):
        score(pag, digraph)  # rejected from either side


def test_label_mismatch_raises():
    a = GraphStructure.from_numpy(
        np.array([[0, 1], [2, 0]], dtype=np.int8), kind="dag", labels=("x", "y")
    )
    b = GraphStructure.from_numpy(
        np.array([[0, 1], [2, 0]], dtype=np.int8), kind="dag", labels=("y", "x")
    )
    with pytest.raises(ValueError, match="label mismatch"):
        score(a, b)


def test_raw_array_with_mark_codes_raises():
    with pytest.raises(ValueError, match="endpoint-mark"):
        score(np.array([[0, 2], [0, 0]]), np.array([[0, 1], [0, 0]]))
