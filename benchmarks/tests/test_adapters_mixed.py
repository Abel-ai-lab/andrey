"""The mixed-data lane: its regime, its adapters, and the driver's solution list.

Every solution reads the categorical columns by one rule, at most ten distinct whole numbers, so
the regime must give its categorical columns few codes and its continuous columns many values; the
pairs must run one method with one setting; Tetrad must read exactly the generated categorical
columns; and the Tetrad adapter must read causal-cmd's graph text exactly. Tetrad's fits run where
``ANDREY_BENCH_TETRAD_JAR`` names the jar and bnlearn's where R is loaded.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import sys

import mixed_cpu
import numpy as np
import pytest
from conftest import needs_r

import andrey
from andrey.data import sample_mixed
from andrey_bench import datasets
from andrey_bench.adapters import tetrad_cmd
from andrey_bench.adapters.andrey_mixed import MAX_CATEGORIES, andrey_mixed_adapters
from andrey_bench.adapters.bnlearn_mixed import BnlearnHCCondGauss, BnlearnPCStableCondGauss
from andrey_bench.contracts import SolutionAdapter
from andrey_bench.datasets import DatasetKey

ADAPTERS = [a for a in mixed_cpu.SOLUTIONS if a.name != "baseline.empty"]
needs_tetrad = pytest.mark.skipif(
    not os.environ.get(tetrad_cmd.JAR_ENV)
    or not (os.environ.get(tetrad_cmd.JAVA_ENV) or shutil.which("java")),
    reason=f"no causal-cmd jar (${tetrad_cmd.JAR_ENV}) or no java",
)


def _key(d: int = 8, n: int = 600, seed: int = 0, share: float = 0.5, categories=(2, 3)):
    return DatasetKey(**mixed_cpu.regime_slice(share, categories), d=d, n=n, seed=seed)


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name)
def test_an_adapter_meets_the_harness_contract(adapter: SolutionAdapter) -> None:
    assert isinstance(adapter, SolutionAdapter)
    json.dumps(adapter.params(), sort_keys=True)
    clone = pickle.loads(pickle.dumps(adapter))
    assert (clone.name, clone.params()) == (adapter.name, adapter.params())
    assert adapter.params()["max_categories"] == MAX_CATEGORIES


def test_every_solution_has_its_own_name() -> None:
    names = [a.name for a in mixed_cpu.SOLUTIONS]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("pair", list(mixed_cpu.PAIRS))
def test_both_packages_run_one_algorithm_with_one_setting(pair: str) -> None:
    mine, theirs = mixed_cpu.PAIRS[pair]
    assert (mine.algorithm, mine.mode) == (theirs.algorithm, theirs.mode)
    assert theirs.output_type in {mine.output_type, "dag"}  # a DAG is scored through its CPDAG
    shared = mine.params().keys() & theirs.params().keys()
    differ = {k for k in shared if mine.params()[k] != theirs.params()[k]}
    # bnlearn's hill climbing has the conditional linear Gaussian BIC only.
    assert differ == ({"score_func"} if pair == "hc degen_gauss bnlearn" else set())
    assert mine.params().keys() - theirs.params().keys() <= {"lambda_value"}


def test_every_comparator_solution_is_paired() -> None:
    paired = {a.name for pair in mixed_cpu.PAIRS.values() for a in pair}
    assert paired == {a.name for a in mixed_cpu.SOLUTIONS} - {"baseline.empty"}


# --- the regime ------------------------------------------------------------------------------------


def test_other_keys_keep_their_label_and_generator_call() -> None:
    """The mixed fields join neither the label nor the call of any other regime."""
    plain = DatasetKey("er", "linear", "gaussian", 2.0, 10, 100, 0, "standardized")
    assert plain.digest == "314b39c589d4"
    discrete = DatasetKey("er", "categorical", "dirichlet", 2.0, 10, 100, 0, "raw")
    shifted = DatasetKey(
        "er", "categorical", "dirichlet", 2.0, 10, 100, 0, "raw", categorical_share=0.9
    )
    assert (discrete.digest, discrete.label) == (shifted.digest, shifted.label)
    assert "categorical_share" not in datasets.generation_kwargs(discrete)


def test_the_share_and_the_categories_name_the_dataset() -> None:
    keys = [_key(), _key(share=0.3), _key(categories=(3, 5))]
    assert len({k.digest for k in keys}) == len({k.label for k in keys}) == 3
    call = datasets.generation_kwargs(keys[2])
    assert call["generator"] == "sample_mixed" and call["categories"] == [3, 5]


def test_the_regime_rejects_rescaled_data() -> None:
    key = DatasetKey(**{**mixed_cpu.regime_slice(0.5, (2, 3)), "standardize": "standardized"},
                     d=5, n=50, seed=0)  # fmt: skip
    with pytest.raises(ValueError, match="raw codes"):
        datasets.generation_kwargs(key)


@pytest.mark.parametrize("categories", [(2, 3), (3, 5)])
def test_the_lane_rule_reads_exactly_the_generated_categorical_columns(categories) -> None:
    key = _key(categories=categories)
    data, marks, kind = datasets.generate_one(key)
    truth = sample_mixed(d=8, n=600, seed=0, categories=categories).provenance["config"]
    with pytest.warns(andrey.DataTypeWarning):
        out = andrey.pc(data, indep_test="degen_gauss", max_categories=MAX_CATEGORIES)
    assert out.metadata["categorical"] == truth["categorical"]
    assert kind == "dag" and marks.shape == (8, 8)


# --- fits ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", andrey_mixed_adapters(), ids=lambda a: a.name)
def test_an_andrey_fit_gives_a_cpdag(adapter) -> None:
    data, _, _ = datasets.generate_one(_key())
    adapter.setup()
    structure = adapter.to_structure(np.asarray(adapter.fit(data, adapter.params())))
    assert structure.kind == "cpdag" and structure.n_nodes == 8


def test_setup_imports_the_searches_fit_defers() -> None:
    searches = ("andrey.constraint.pc", "andrey.search.ges", "andrey.search.hc")
    for module in searches:
        sys.modules.pop(module, None)
    andrey_mixed_adapters()[0].setup()
    assert set(searches) <= set(sys.modules)


def test_tetrad_graph_text_reads_as_a_cpdag_adjacency() -> None:
    text = (
        "Graph Nodes:\nx0;x1;x2\n\nGraph Edges:\n1. x0 --> x2\n2. x1 --- x2\n3. x2 <-- x1\n\n"
        "Graph Attributes:\nBIC: 1.0\n"
    )
    expected = np.zeros((3, 3), dtype=np.int8)
    expected[0, 2] = expected[1, 2] = expected[2, 1] = 1
    assert np.array_equal(tetrad_cmd.read_graph(text, 3), expected)
    with pytest.raises(ValueError, match="bidirected"):
        tetrad_cmd.read_graph("Graph Edges:\n1. x0 <-> x1\n", 2)
    with pytest.raises(ValueError, match="unexpected"):
        tetrad_cmd.read_graph("Graph Edges:\n1. x0 o-> x1\n", 2)


def test_the_tetrad_driver_names_the_columns_it_read_as_categorical() -> None:
    text = "Categorical columns: x0;x2\n\nGraph Nodes:\nx0;x1;x2\n\nGraph Edges:\n"
    assert tetrad_cmd.read_categorical(text) == ["x0", "x2"]
    assert tetrad_cmd.read_categorical("Categorical columns: \n\nGraph Edges:\n") == []
    with pytest.raises(ValueError, match="categorical-column line"):
        tetrad_cmd.read_categorical("Graph Edges:\n")


@needs_tetrad
@pytest.mark.parametrize("categories", [(2, 3), (3, 5)])
def test_tetrad_reads_exactly_the_generated_categorical_columns(categories) -> None:
    data, _, _ = datasets.generate_one(_key(categories=categories))
    truth = sample_mixed(d=8, n=600, seed=0, categories=categories).provenance["config"]
    adapter = tetrad_cmd.TetradPC("cond_gauss")
    adapter.setup()
    declared = tetrad_cmd.read_categorical(adapter.run(data, adapter.params()))
    assert declared == [f"x{j}" for j in truth["categorical"]]


@needs_tetrad
@pytest.mark.parametrize("adapter", tetrad_cmd.tetrad_adapters(), ids=lambda a: a.name)
def test_a_tetrad_fit_gives_a_cpdag(adapter) -> None:
    data, _, _ = datasets.generate_one(_key())
    adapter.setup()
    structure = adapter.to_structure(adapter.fit(data, adapter.params()))
    assert structure.kind == "cpdag" and structure.n_nodes == 8


@needs_r()
def test_bnlearn_gives_no_categorical_node_a_continuous_parent() -> None:
    data, _, _ = datasets.generate_one(_key(d=10, n=1000))
    categorical = sample_mixed(d=10, n=1000, seed=0).provenance["config"]["categorical"]
    hc = BnlearnHCCondGauss()
    hc.setup()
    dag = hc.fit(data, hc.params())
    continuous = [j for j in range(10) if j not in categorical]
    assert not dag[np.ix_(continuous, categorical)].any()
    pc = BnlearnPCStableCondGauss()
    pc.setup()
    assert pc.to_structure(pc.fit(data, pc.params())).kind == "cpdag"


def test_an_unknown_solution_name_stops_the_driver(tmp_path) -> None:
    with pytest.raises(SystemExit, match="unknown solutions"):
        mixed_cpu.main(["--out", str(tmp_path), "--solutions", "andrey.pc.fisherz"])
