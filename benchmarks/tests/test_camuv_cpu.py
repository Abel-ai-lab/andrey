"""The CAM-UV lane: the paper's simulation, CAM-UV's targets, the adapters, and the scoring pass.

Facts about the simulations, the scores, and Andrey's adapter are checked in bench-env. Facts about
the reference boundary (a lingam or causal-learn fit runs in ``.venv-lingam`` and comes back as marks
this process scores) are checked by running one, and skipped when that environment has not been
built.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import camuv_cpu
import numpy as np
import pandas as pd
import pytest

import andrey
from andrey import GraphStructure
from andrey.core import LATENT, OBSERVED
from andrey_bench import datasets, regime_models, rundir, runner, scoring
from andrey_bench.adapters.andrey_camuv import AndreyCAMUV
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.causal_learn_camuv import CausalLearnCAMUV, causal_learn_camuv_adapter
from andrey_bench.adapters.lingam_camuv import LingamCAMUV, lingam_camuv_adapter
from andrey_bench.adapters.lingam_env import PYTHON_ENV_VAR
from andrey_bench.integration import run_batch

# Andrey endpoint marks.
_TAIL, _ARROW, _CIRCLE = 1, 2, 3

LINGAM_PYTHON = Path(runner.BENCH_ROOT) / ".venv-lingam" / "bin" / "python"
needs_lingam_env = pytest.mark.skipif(
    not LINGAM_PYTHON.exists(), reason="lingam env not built - run benchmarks/start_lingam.sh"
)


def _paper(seed: int, n: int = 200, d: int = 14) -> tuple[np.ndarray, np.ndarray]:
    data, marks, kind, _ = regime_models.sample(
        "camuv2021", d=d, n=n, seed=seed, density=camuv_cpu.PAPER_DENSITY, noise="uniform"
    )
    assert kind == "dag"
    return data, marks


def _directed(marks: np.ndarray) -> set[tuple[int, int]]:
    m = np.asarray(marks)
    return {(int(i), int(j)) for i, j in zip(*np.nonzero((m == _TAIL) & (m.T == _ARROW)))}


def _dag(edges: list[tuple[int, int]], d: int, hidden: int) -> GraphStructure:
    marks = np.zeros((d, d), dtype=np.int8)
    for i, j in edges:
        marks[i, j], marks[j, i] = _TAIL, _ARROW
    types = np.full(d, OBSERVED, dtype=np.int8)
    types[d - hidden :] = LATENT
    return GraphStructure.from_numpy(marks, kind="dag", node_types=types)


def _pag(directed: list[tuple[int, int]], pairs: list[tuple[int, int]], mark: int, d: int):
    marks = np.zeros((d, d), dtype=np.int8)
    for i, j in directed:
        marks[i, j], marks[j, i] = _TAIL, _ARROW
    for i, j in pairs:
        marks[i, j] = marks[j, i] = mark
    return GraphStructure.from_numpy(marks, kind="pag")


# --- the data --------------------------------------------------------------------------------------


def test_the_latent_regimes_hide_one_node_in_five_and_one_in_ten() -> None:
    hidden = {
        regime: [datasets.latents_for(regime, d) for d in camuv_cpu.SETTINGS[regime]["ladder"]]
        for regime in ("latent_anm_er", "latent_anm_er_tenth", "camuv_paper")
    }
    assert hidden == {
        "latent_anm_er": [1, 2, 3],
        "latent_anm_er_tenth": [1, 2, 5],
        "camuv_paper": [4],
    }


def test_the_paper_simulation_has_two_hidden_causes_and_two_hidden_intermediates() -> None:
    for seed in range(10):
        data, marks = _paper(seed)
        assert data.shape == (200, 10) and marks.shape == (14, 14)
        np.testing.assert_allclose(data.std(axis=0), 1.0)
        edges = _directed(marks)
        for cause in (10, 11):  # roots, each into two observed variables
            assert not any(c == cause for _, c in edges)
            children = [c for p, c in edges if p == cause]
            assert len(children) == 2 and max(children) < 10
        for y in (12, 13):  # one observed parent, one observed child, and no edge between them
            (parent,) = [p for p, c in edges if c == y]
            (child,) = [c for p, c in edges if p == y]
            assert max(parent, child) < 10 and (parent, child) not in edges


def test_the_observed_graph_has_the_papers_edge_probability() -> None:
    """Each pair of the 10 observed variables is joined with probability 0.3 before the two hidden
    intermediates take the place of two of the edges."""
    drawn = []
    for seed in range(200):
        _, marks = _paper(seed, n=20)
        observed = [(p, c) for p, c in _directed(marks) if max(p, c) < 10]
        drawn.append(len(observed) + 2)
    assert 0.28 < np.mean(drawn) / 45 < 0.32


def test_the_paper_simulation_draws_its_own_noise_only() -> None:
    with pytest.raises(ValueError, match="uniform"):
        regime_models.sample("camuv2021", d=14, n=50, seed=0, density=2.7, noise="gaussian")


#: SHA-256 of the data and truth of the paper's simulation at seeds 0 to 2, 50 rows. A change to any
#: drawn value changes it, and with it the datasets a campaign has stored.
PAPER_DRAWS = "4f054bc3d4f7c9b4f6ca0e8687b548e70e9054ea9e36853dbbc5d2ad04e9477a"


def test_the_paper_simulation_keeps_its_draws() -> None:
    digest = hashlib.sha256()
    for seed in range(3):
        data, marks = _paper(seed, n=50)
        digest.update(np.ascontiguousarray(data).tobytes() + marks.tobytes())
    assert digest.hexdigest() == PAPER_DRAWS


def test_a_seed_that_cannot_hide_its_nodes_is_skipped() -> None:
    """At 10 nodes, seed 31 has fewer than two nodes with two children."""
    keys = camuv_cpu.regime_keys("latent_anm_er", 10, 100, seeds=2, seed_start=31)
    assert [k.seed for k in keys] == [32, 33]
    with pytest.raises(ValueError, match="cannot hide"):
        datasets.generate_one(dataclasses.replace(keys[0], seed=31))


# --- the targets and the scores --------------------------------------------------------------------

#: Observed 0 to 3, hidden 4 and 5: 4 causes 0 and 1, and 2 -> 5 -> 3 puts 5 between 2 and 3.
HAND = _dag([(4, 0), (4, 1), (0, 2), (2, 5), (5, 3), (1, 3)], 6, 2)


def test_a_hidden_cause_or_intermediate_leaves_its_pairs_unresolved() -> None:
    """(0, 1) share the hidden 4; 2 and 0 reach 3 through the hidden 5; 1 <- 4 -> 0 -> 2 -> 5 -> 3
    is a backdoor path entering both 1 and 3 from hidden parents. Only 0 -> 2 has no such path."""
    assert scoring.unobserved_path_pairs(HAND) == {(0, 1), (0, 3), (1, 3), (2, 3)}
    assert scoring.direct_edges(HAND) == {(0, 2)}
    assert scoring.true_confounded_pairs(HAND, "hidden_cause") == {(0, 1)}


def test_the_direct_target_scores_edges_and_marks_apart() -> None:
    estimate = _pag([(0, 2), (1, 3)], [(0, 1)], _CIRCLE, 4)
    directed = scoring.directed_edge_scores(estimate, HAND)
    assert directed["directed_direct_n_true"] == 1
    assert directed["directed_direct_errors"] == 1  # 1 -> 3 lies on an unobserved path
    pairs = scoring.confounded_pair_scores(estimate, HAND, "circle")
    assert pairs["confounded_unobserved_path_precision"] == 1.0
    assert pairs["confounded_unobserved_path_recall"] == 0.25


def test_the_redraw_turns_the_marked_pairs_bidirected_and_keeps_the_graph() -> None:
    ours = _pag([(0, 2)], [(0, 1)], _CIRCLE, 4)
    theirs = _pag([(0, 2)], [(0, 1)], _ARROW, 4)
    redrawn = camuv_cpu.as_bidirected(ours)
    np.testing.assert_array_equal(redrawn.to_numpy(), theirs.to_numpy())
    np.testing.assert_array_equal(camuv_cpu.as_bidirected(theirs).to_numpy(), theirs.to_numpy())
    assert camuv_cpu.graph_key(ours, "circle") == camuv_cpu.graph_key(theirs, "bidirected")
    assert camuv_cpu.graph_key(ours, "circle") != camuv_cpu.graph_key(
        _pag([(0, 2)], [], _ARROW, 4), "bidirected"
    )


# --- the adapters ----------------------------------------------------------------------------------


def test_every_package_runs_andreys_defaults() -> None:
    defaults = inspect.signature(andrey.camuv).parameters
    ours = AndreyCAMUV().params()
    assert ours == {name: defaults[name].default for name in ours}
    assert LingamCAMUV().params() == CausalLearnCAMUV().params() == ours
    for theirs in (LingamCAMUV(), CausalLearnCAMUV()):
        assert (theirs.algorithm, theirs.output_type) == ("camuv", "pag")
        assert theirs.python_env == PYTHON_ENV_VAR
    assert AndreyCAMUV.confounded_mark == "circle"
    assert LingamCAMUV.confounded_mark == CausalLearnCAMUV.confounded_mark == "bidirected"


def test_the_references_take_the_largest_set_as_max_depth_plus_one(monkeypatch) -> None:
    """lingam's ``num_explanatory_vals`` and causal-learn's third argument count the variable
    itself; a ``NaN`` pair of lingam and a pair of causal-learn become ``<->``."""
    calls = []
    adjacency = np.array([[0.0, 0.0, np.nan], [1.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])

    class FakeCAMUV:
        def __init__(self, **settings):
            calls.append(settings)

        def fit(self, X):
            return SimpleNamespace(adjacency_matrix_=adjacency)

    lingam = ModuleType("lingam")
    lingam.CAMUV = FakeCAMUV
    monkeypatch.setitem(sys.modules, "lingam", lingam)
    reference = SimpleNamespace(
        execute=lambda X, alpha, k: calls.append((alpha, k)) or ([set(), {0}, set()], [{0, 2}])
    )
    for name in ("causallearn", "causallearn.search", "causallearn.search.FCMBased"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    module = ModuleType("causallearn.search.FCMBased.lingam")
    module.CAMUV = reference
    monkeypatch.setitem(sys.modules, "causallearn.search.FCMBased.lingam", module)

    X = np.zeros((10, 3))
    marks = [a.fit(X, a.params()) for a in (LingamCAMUV(), CausalLearnCAMUV())]
    assert calls == [{"alpha": 0.01, "num_explanatory_vals": 3}, (0.01, 3)]
    expected = _pag([(0, 1)], [(0, 2)], _ARROW, 3).to_numpy()
    for m in marks:
        np.testing.assert_array_equal(m, expected)


def test_the_regimes_of_camuvs_model_class_are_in_claim() -> None:
    assert [camuv_cpu.claim(r) for r in camuv_cpu.REGIMES] == ["in", "in", "in", "out", "out"]


def test_the_scoring_pass_adds_every_column(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(datasets.DATA_STORE_ENV, raising=False)
    keys = camuv_cpu.regime_keys("latent_anm_er", 5, 300, seeds=1)
    records_dir = tmp_path / rundir.RECORDS_DIRNAME
    records_dir.mkdir()
    result = run_batch(
        datasets.store_dir(tmp_path),
        [AndreyCAMUV(), BaselineEmpty()],
        keys,
        machine_id="test",
        cap_wall_s=300.0,
        cores=1,
        records_dir=records_dir,
        run_dir=tmp_path,
    )
    assert not result.errors
    camuv_cpu.add_scores(tmp_path, records_dir, {k.digest: k for k in keys})
    table = rundir.assemble(records_dir).set_index("name")
    assert set(camuv_cpu.SCORE_COLUMNS) <= set(table.columns)
    empty = table.loc["baseline.empty"]
    assert empty["directed_direct_errors"] == empty["directed_direct_n_true"]
    assert empty["shd_endpoint_bidirected"] == empty["shd_endpoint"]
    assert empty["confounded_n_est"] == 0
    ours = table.loc["andrey.camuv.numpy"]
    if ours["confounded_n_est"] == 0:
        assert ours["shd_endpoint_bidirected"] == ours["shd_endpoint"]
    assert isinstance(ours["graph_key"], str) and len(ours["graph_key"]) == 40
    assert pd.api.types.is_float_dtype(table["shd_endpoint_bidirected"])


@needs_lingam_env
def test_both_references_run_there_and_return_one_graph(monkeypatch, tmp_path: Path) -> None:
    """The two references implement one algorithm; on a small dataset they return one graph."""
    monkeypatch.setenv(PYTHON_ENV_VAR, str(LINGAM_PYTHON))
    (key,) = camuv_cpu.regime_keys("latent_anm_er", 5, 500, seeds=1)
    datasets.materialize([key], tmp_path)
    graphs = []
    for adapter, version in (
        (lingam_camuv_adapter(), "1.13.0"),
        (causal_learn_camuv_adapter(), "0.1.4.8"),
    ):
        assert adapter.package_version == version
        (measures,) = runner.run_task(
            adapter,
            datasets.data_path(tmp_path, key),
            adapter.params(),
            cap_wall_s=600.0,
            repeats=1,
            warmup=0,
        )
        assert measures["status"] == "ok", measures.get("stderr")
        graphs.append(scoring.structure_hash(adapter.to_structure(measures["adj"])))
    assert graphs[0] == graphs[1]
