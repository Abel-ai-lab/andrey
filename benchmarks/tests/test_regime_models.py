"""The domain, group, and measurement regimes, their adapters, and the scores read on them.

Each regime's data must have the layout its adapters split, and its truth the graph the method can
identify. Each reference adapter must hand back the graph its package returned in the encoding
Andrey's adapter uses, checked with a stand-in for the package where it is not installed here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from andrey import GraphStructure
from andrey.core import LATENT, OBSERVED
from andrey_bench import datasets, pairing, regime_models, rundir
from andrey_bench.adapters.andrey_cdnod import AndreyCDNOD
from andrey_bench.adapters.andrey_gin import AndreyGIN
from andrey_bench.adapters.andrey_multi_group import AndreyMultiGroupDirectLiNGAM
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.causal_learn_cdnod import CausalLearnCDNOD
from andrey_bench.adapters.gin_convert import encode_clusters, latent_dag
from andrey_bench.adapters.lingam_multi_group import LingamMultiGroupDirectLiNGAM
from andrey_bench.contracts import Status
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.scoring import measurement_scores, score_result

# Andrey endpoint marks.
_TAIL, _ARROW = 1, 2


def _key(regime: str, d: int, n: int, seed: int = 0) -> DatasetKey:
    topology, functional, noise, standardize = datasets.REGIMES[regime]
    density = 3.0 if functional == "measurement" else datasets.CANONICAL_DENSITY
    return DatasetKey(
        topology=topology,
        functional=functional,
        noise=noise,
        standardize=standardize,
        density=density,
        d=d,
        n=n,
        seed=seed,
        latents=datasets.latents_for(regime, d),
    )


def _directed(marks: np.ndarray) -> set[tuple[int, int]]:
    m = np.asarray(marks)
    return {(int(i), int(j)) for i, j in zip(*np.nonzero((m == _TAIL) & (m.T == _ARROW)))}


def _undirected(marks: np.ndarray) -> set[tuple[int, int]]:
    m = np.asarray(marks)
    return {(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu((m == _TAIL) & (m.T == _TAIL))))}


# ==================================================================================================
# The regimes.
# ==================================================================================================


def test_each_regime_has_the_layout_its_adapters_split() -> None:
    x, marks, kind, _ = datasets.generate_one(_key("domain_shift_er", 6, 400))
    assert x.shape == (400, 6) and marks.shape == (6, 6) and kind == "cpdag"

    groups = regime_models.N_GROUPS
    x, marks, kind, _ = datasets.generate_one(_key("multi_group_er", 5, 300))
    assert x.shape == (300, 5 * groups) and kind == "dag"
    blocks = [marks[g * 5 : (g + 1) * 5, g * 5 : (g + 1) * 5] for g in range(groups)]
    assert all(np.array_equal(b, blocks[0]) for b in blocks), "one DAG in every group"
    assert np.count_nonzero(marks) == groups * np.count_nonzero(blocks[0]), "no edge across groups"

    key = _key("latent_measurement", 12, 300)
    x, marks, kind, _ = datasets.generate_one(key)
    assert key.latents == 3 and x.shape == (300, 9) and kind == "dag"
    parents = (marks == _TAIL) & (marks.T == _ARROW)  # parents[i, j]: i -> j
    assert (parents[:, :9].sum(axis=0) == 1).all(), "each indicator has one parent"
    assert parents[9:, :9].sum(axis=1).tolist() == [3, 3, 3], "each latent has three indicators"
    assert not parents[:9].any(), "indicators cause nothing"


def test_a_measurement_model_needs_whole_latents() -> None:
    with pytest.raises(ValueError, match="m >= 2"):
        datasets.latents_for("latent_measurement", 10)


def test_the_domain_index_orients_what_it_reaches() -> None:
    chain = np.zeros((3, 3), dtype=np.int8)
    for i, j in ((0, 1), (1, 2)):
        chain[i, j], chain[j, i] = _TAIL, _ARROW
    # A shift at the root orients the whole chain; PC's CPDAG leaves it undirected.
    root = regime_models.domain_cpdag(chain, np.array([0]))
    assert _directed(root) == {(0, 1), (1, 2)}
    # A shift at the end makes 1 a parent of 2 by the collider at 2; 0 - 1 stays open.
    end = regime_models.domain_cpdag(chain, np.array([2]))
    assert _directed(end) == {(1, 2)} and _undirected(end) == {(0, 1)}


# ==================================================================================================
# The adapters.
# ==================================================================================================


def test_causal_learn_cdnod_drops_the_index_and_passes_its_settings(monkeypatch) -> None:
    pytest.importorskip("causallearn")
    from causallearn.search.ConstraintBased import CDNOD

    calls = []
    # causal-learn's endpoints over x0, x1, and the index: index -> x0 -> x1.
    graph = np.array([[0, -1, 1], [1, 0, 0], [-1, 0, 0]])

    def cdnod(data, domain_index, **kwargs):
        calls.append((data.shape, domain_index[:, 0].tolist(), kwargs))
        return SimpleNamespace(G=SimpleNamespace(graph=graph))

    monkeypatch.setattr(CDNOD, "cdnod", cdnod)
    adapter = CausalLearnCDNOD()
    native = adapter.fit(np.zeros((8, 2)), {**adapter.params(), "n_domains": 4})
    assert _directed(adapter.to_structure(native).to_numpy()) == {(0, 1)}
    shape, index, settings = calls[0]
    assert shape == (8, 2) and index == [0, 0, 1, 1, 2, 2, 3, 3]
    assert settings == {
        "alpha": 0.05,
        "indep_test": "fisherz",
        "stable": True,
        "uc_rule": 0,
        "uc_priority": 2,
        "show_progress": False,
    }


def test_gin_clusters_and_order_build_one_graph_whichever_package_listed_them() -> None:
    # Clusters {1, 3} before {0, 4}; 2 in none; {5, 6} left unordered.
    native = encode_clusters([[1, 3], [4, 0]], [[6, 5]], 7)
    assert native.tolist() == [[1, 0, -1, 0, 1, 2, 2], [1, 0, -1, 0, 1, -1, -1]]
    graph = latent_dag(native)
    assert graph.n_nodes == 10 and list(graph.node_types[7:]) == [LATENT] * 3
    assert _directed(graph.to_numpy()) == {
        (7, 1), (7, 3), (8, 0), (8, 4), (9, 5), (9, 6),  # each latent into its cluster
        (7, 8), (7, 9), (8, 9),  # ordered latents into later and unordered ones
    }  # fmt: skip


def test_causal_learn_gin_reads_the_clusters_it_left_unordered() -> None:
    pytest.importorskip("causallearn")
    from causallearn.graph.GeneralGraph import GeneralGraph
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.graph.NodeType import NodeType

    from andrey_bench.adapters.causal_learn_gin import clusters_of

    nodes = {name: GraphNode(name) for name in ("X3", "X1", "X2", "L1", "L2")}
    for name in ("L1", "L2"):
        nodes[name].set_node_type(NodeType.LATENT)
    graph = GeneralGraph(list(nodes.values()))
    graph.add_directed_edge(nodes["L1"], nodes["L2"])
    graph.add_directed_edge(nodes["L1"], nodes["X3"])
    graph.add_directed_edge(nodes["L1"], nodes["X1"])
    graph.add_directed_edge(nodes["L2"], nodes["X2"])
    assert clusters_of(graph) == [[0, 2], [1]]


def _cl_gin_case(case: int) -> tuple[np.ndarray, list[list[int]]]:
    """Data and true clusters of causal-learn's TestGIN case `case` (1 or 2)."""
    n, u = 500, np.random.uniform
    np.random.seed(42)
    l1 = u(-1, 1, n)
    l2 = u(1.2, 1.8) * l1 + u(-1, 1, n)
    if case == 1:
        latents, clusters = (l1, l1, l2, l2), [[0, 1], [2, 3]]
    else:
        l3 = u(1.2, 1.8) * l1 + u(1.2, 1.8) * l2 + u(-1, 1, n)
        latents, clusters = (l1,) * 3 + (l2,) * 3 + (l3,) * 3, [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    x = np.array([u(1.2, 1.8) * lat + 0.2 * u(-1, 1, n) for lat in latents]).T
    return (x - x.mean(0)) / x.std(0), clusters


def _patched_gin() -> type:
    """Return the causal-learn GIN adapter, skipping where the reference lacks its patch."""
    pytest.importorskip("causallearn")
    from andrey_bench.adapters.causal_learn_gin import CausalLearnGIN, _gin_fixed

    if not _gin_fixed():
        pytest.skip(
            "causal-learn GIN lacks the column-index fix: run benchmarks/patch_causal_learn.sh"
        )
    return CausalLearnGIN


@pytest.mark.parametrize("case", [1, 2])
def test_causal_learn_gin_recovers_its_own_test_cases(case: int) -> None:
    adapter = _patched_gin()()
    adapter.setup()
    x, clusters = _cl_gin_case(case)
    truth = latent_dag(encode_clusters(clusters, [], x.shape[1]))
    scores = measurement_scores(adapter.to_structure(adapter.fit(x, adapter.params())), truth)
    assert scores["measurement_cluster_ari"] == 1.0
    assert scores["measurement_order_accuracy"] == 1.0
    assert adapter.package_version.endswith("with the GIN column-index fix")


def test_causal_learn_gin_recovers_a_lane_dataset() -> None:
    gin = _patched_gin()
    key = _key("latent_measurement", 8, 1000)
    x, marks, kind, _ = datasets.generate_one(key)
    truth = GraphStructure.from_numpy(
        marks, kind=kind, node_types=datasets.latent_node_types(len(marks), key.latents)
    )
    adapter = gin()
    adapter.setup()
    scores = measurement_scores(adapter.to_structure(adapter.fit(x, adapter.params())), truth)
    assert scores["measurement_cluster_ari"] == 1.0
    assert scores["measurement_clusters_found"] == 1.0


def test_tetrad_gin_reads_ordered_and_unordered_clusters(monkeypatch) -> None:
    from andrey_bench.adapters import tetrad_gin

    def run(command, **kwargs):
        Path(command[-1]).write_text(
            "Search seconds: 1.5\nOrdered clusters: 3 4 5\nUnordered clusters: 0 1;2 6\n"
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(tetrad_gin, "_java", lambda: "java")
    monkeypatch.setattr(tetrad_gin, "_jar", lambda: Path("tetrad-lib.jar"))
    monkeypatch.setattr(tetrad_gin.subprocess, "run", run)
    adapter = tetrad_gin.TetradGIN()
    native = adapter.fit(np.zeros((5, 7)), adapter.params())
    assert adapter.search_s == 1.5
    assert native.tolist() == encode_clusters([[3, 4, 5]], [[0, 1], [2, 6]], 7).tolist()


@pytest.mark.skipif(
    not os.environ.get("ANDREY_BENCH_TETRAD_GIN_JAR"), reason="no tetrad-lib jar for GIN"
)
@pytest.mark.parametrize("case", [1, 2])
def test_tetrad_gin_recovers_causal_learns_test_cases(case: int) -> None:
    from andrey_bench.adapters.tetrad_gin import TetradGIN

    adapter = TetradGIN()
    adapter.setup()
    x, clusters = _cl_gin_case(case)
    truth = latent_dag(encode_clusters(clusters, [], x.shape[1]))
    scores = measurement_scores(adapter.to_structure(adapter.fit(x, adapter.params())), truth)
    assert scores["measurement_cluster_ari"] == 1.0
    assert scores["measurement_order_accuracy"] == 1.0
    assert adapter.package_version.startswith("tetrad-lib 7.6.10")


def test_lingam_multi_group_transposes_each_group(monkeypatch) -> None:
    # lingam's B[i, j] weights j -> i: in group 0, 0 -> 1; in group 1, 1 -> 0.
    fitted = []

    class MultiGroupDirectLiNGAM:
        def fit(self, groups):
            fitted.append([g.shape for g in groups])
            self.adjacency_matrices_ = [np.array([[0, 0], [0.7, 0]]), np.array([[0, -0.4], [0, 0]])]
            return self

    monkeypatch.setitem(
        sys.modules, "lingam", SimpleNamespace(MultiGroupDirectLiNGAM=MultiGroupDirectLiNGAM)
    )
    adapter = LingamMultiGroupDirectLiNGAM()
    native = adapter.fit(np.zeros((10, 4)), {"n_groups": 2})
    assert fitted == [[(10, 2), (10, 2)]]
    assert _directed(adapter.to_structure(native).to_numpy()) == {(0, 1), (3, 2)}


# ==================================================================================================
# Scores.
# ==================================================================================================


def _measurement_truth() -> GraphStructure:
    """Latents 4 -> 5, measured by {0, 1} and {2, 3}."""
    marks = np.zeros((6, 6), dtype=np.int8)
    for i, j in ((4, 0), (4, 1), (5, 2), (5, 3), (4, 5)):
        marks[i, j], marks[j, i] = _TAIL, _ARROW
    types = np.array([OBSERVED] * 4 + [LATENT] * 2, dtype=np.int8)
    return GraphStructure.from_numpy(marks, kind="dag", node_types=types)


def test_measurement_scores_read_the_clusters_and_their_order() -> None:
    truth = _measurement_truth()
    right = latent_dag(encode_clusters([[0, 1], [2, 3]], [], 4))
    reversed_ = latent_dag(encode_clusters([[2, 3], [0, 1]], [], 4))
    one_found = latent_dag(encode_clusters([[0, 1]], [], 4))
    empty = GraphStructure.from_numpy(np.zeros((4, 4), dtype=np.int8), kind="cpdag")

    assert measurement_scores(right, truth) == {
        "measurement_cluster_ari": 1.0,
        "measurement_n_latents": 2,
        "measurement_clusters_found": 1.0,
        "measurement_order_accuracy": 1.0,
    }
    assert measurement_scores(reversed_, truth)["measurement_order_accuracy"] == 0.0
    partial = measurement_scores(one_found, truth)
    assert partial["measurement_clusters_found"] == 0.5
    assert np.isnan(partial["measurement_order_accuracy"]), "no pair of found latents"
    nothing = measurement_scores(empty, truth)
    assert nothing["measurement_cluster_ari"] == 0.0 and nothing["measurement_n_latents"] == 0


def test_a_stored_latent_estimate_keeps_its_latents(tmp_path) -> None:
    graph = latent_dag(encode_clusters([[0, 1], [2, 3]], [], 4))
    rundir.write_structure(tmp_path, "run", graph)
    back = rundir.read_structure(tmp_path, "run")
    assert back == graph
    assert measurement_scores(back, _measurement_truth())["measurement_order_accuracy"] == 1.0


def test_a_latent_estimate_is_scored_on_its_clustering() -> None:
    truth = _measurement_truth()
    scores = score_result(latent_dag(encode_clusters([[0, 1], [2, 3]], [], 4)), truth)
    assert scores["family"] == "latent" and scores["latent_cluster_ari"] == 1.0


def test_same_graph_compares_the_datasets_both_finished() -> None:
    a = pd.DataFrame(
        {"dataset_id": ["x", "y", "z"], "status": ["ok"] * 3, "structure_hash": ["1", "2", "3"]}
    )
    b = pd.DataFrame(
        {
            "dataset_id": ["x", "y", "z"],
            "status": ["ok", "ok", "timeout"],
            "structure_hash": ["1", "9", None],
        }
    )
    assert pairing.same_graph(a, b) == (1, 2)


# ==================================================================================================
# One tiny dataset per regime, end to end.
# ==================================================================================================


@pytest.mark.parametrize(
    "regime, d, n, adapter",
    [
        ("domain_shift_er", 6, 400, AndreyCDNOD()),
        ("multi_group_er", 4, 300, AndreyMultiGroupDirectLiNGAM()),
        ("latent_measurement", 8, 300, AndreyGIN()),
    ],
)
def test_andrey_and_the_empty_graph_fit_and_score_each_regime(tmp_path, regime, d, n, adapter):
    result = run_batch(
        tmp_path, [adapter, BaselineEmpty()], [_key(regime, d, n)], machine_id="tiny-cpu",
        cap_wall_s=240.0,
    )  # fmt: skip
    assert not result.errors, result.errors
    for record in result.records:
        assert record.status == Status.OK.value, f"{record.name}: {record.error_text}"
        assert record.metrics, record.name
