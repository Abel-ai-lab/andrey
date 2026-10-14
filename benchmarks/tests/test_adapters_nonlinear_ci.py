"""The adapters and the driver of the nonlinear CI lane: PC with KCI, RCoT, CMIknn, and GPDC.

Every solution runs at Andrey's defaults under the package's own names, and every fit returns a
CPDAG the harness scores the same way. tigramite fits only in its own environment, so its fit
runs when ``ANDREY_BENCH_TIGRAMITE_PYTHON`` names one; its graph conversion is checked here.
"""

from __future__ import annotations

import inspect
import os
import subprocess
from pathlib import Path

import nonlinear_ci_cpu
import numpy as np
import pytest

from andrey import GraphStructure
from andrey_bench import datasets
from andrey_bench.adapters.andrey_pc import AndreyPC
from andrey_bench.adapters.causal_learn_pc import CI_TESTS, CausalLearnPC
from andrey_bench.adapters.tigramite_pc import (
    PYTHON_ENV_VAR,
    TigramitePC,
    graph_to_adjacency,
    tigramite_pc_adapter,
)

NONLINEAR = ("kci", "rcot", "cmiknn", "gpdc")


def _nonlinear_data(seed: int = 0, n: int = 200) -> np.ndarray:
    """``x0 -> x1`` through ``|x0|``, and an independent ``x2``."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    return np.column_stack([x0, np.abs(x0) + 0.2 * rng.standard_normal(n), rng.standard_normal(n)])


def test_every_solution_has_its_own_name_and_test() -> None:
    field = nonlinear_ci_cpu.solutions()
    names = [a.name for a in field]
    assert len(set(names)) == len(names)
    tested = {(a.package, a.params().get("indep_test")) for a in field if a.package != "baseline"}
    assert {("andrey", t) for t in ("fisherz", *NONLINEAR)} <= tested
    assert {("causal-learn", "kci"), ("causal-learn", "rcot")} <= tested
    assert {("tigramite", "cmiknn"), ("tigramite", "gpdc")} <= tested


def _defaults(cls) -> dict:
    return {k: v.default for k, v in inspect.signature(cls).parameters.items() if k != "data"}


def test_references_run_at_andreys_defaults() -> None:
    """The settings each reference is pinned to are Andrey's defaults for the same test."""
    from andrey.core.ci_cmiknn import CMIknn
    from andrey.core.ci_kernel import RCoT

    name, settings = CI_TESTS["rcot"]
    rcot = _defaults(RCoT)
    assert (name, settings["rcit"], settings["approx"]) == ("rcit", False, "hbe")
    assert (settings["num_f2"], settings["num_f"]) == (rcot["n_features"], rcot["n_features_cond"])
    mine, cmiknn = TigramitePC("cmiknn").params(), _defaults(CMIknn)
    assert all(mine[k] == cmiknn[k] for k in ("k_cmi", "k_perm", "n_permutations"))


def test_the_regime_is_additive_nonlinear_noise() -> None:
    assert datasets.REGIMES[nonlinear_ci_cpu.REGIME] == ("er", "anm", "gaussian", "standardized")
    keys = nonlinear_ci_cpu.ladder_keys([5], [200, 500], seeds=2)
    assert sorted({(k.d, k.n) for k in keys}) == [(5, 200), (5, 500)]
    assert len({k.digest for k in keys}) == 4


@pytest.mark.parametrize("test", NONLINEAR)
def test_andrey_pc_runs_each_nonlinear_test(test: str) -> None:
    adapter = AndreyPC(test)
    assert adapter.name == f"andrey.pc.{test}.numpy"
    adapter.setup()
    native = adapter.fit(_nonlinear_data(), adapter.params())
    structure = adapter.to_structure(native)
    assert isinstance(structure, GraphStructure) and structure.kind == "cpdag"
    assert {(i, j) for i, j, _ in structure.oriented_edges()} == {(0, 1)}


@pytest.mark.parametrize("test", ["kci", "rcot"])
def test_causal_learn_pc_runs_each_reference_test(test: str) -> None:
    """Plumbing only: causal-learn's RCoT draws unseeded features, so its graph is not fixed."""
    pytest.importorskip("causallearn")
    adapter = CausalLearnPC(test)
    assert adapter.name == f"causal-learn.pc.{test}"
    native = np.asarray(adapter.fit(_nonlinear_data(), adapter.params()))
    assert native.shape == (3, 3) and set(np.unique(native).tolist()) <= {0, 1}
    assert adapter.to_structure(native).kind == "cpdag"


@pytest.mark.parametrize("cls", [AndreyPC, CausalLearnPC, TigramitePC])
def test_an_unknown_test_is_rejected(cls) -> None:
    with pytest.raises(ValueError, match="indep_test"):
        cls("no_such_test")


def test_tigramite_links_convert_to_a_cpdag_adjacency() -> None:
    """``-->`` and ``<--`` orient an edge; ``o-o`` and ``x-x`` leave it undirected."""
    graph = np.full((4, 4, 1), "", dtype="<U3")
    graph[0, 1, 0], graph[1, 0, 0] = "-->", "<--"
    graph[1, 2, 0], graph[2, 1, 0] = "o-o", "o-o"
    graph[2, 3, 0], graph[3, 2, 0] = "x-x", "x-x"
    expected = np.zeros((4, 4), dtype=np.int8)
    expected[0, 1] = expected[1, 2] = expected[2, 1] = expected[2, 3] = expected[3, 2] = 1
    np.testing.assert_array_equal(graph_to_adjacency(graph), expected)
    with pytest.raises(ValueError, match="unexpected link"):
        graph_to_adjacency(np.full((2, 2, 1), "+->", dtype="<U3"))


@pytest.mark.skipif(not os.environ.get(PYTHON_ENV_VAR), reason=f"${PYTHON_ENV_VAR} is unset")
@pytest.mark.parametrize("test", ["cmiknn", "gpdc"])
def test_tigramite_pc_runs_in_its_environment(test: str, tmp_path: Path) -> None:
    """The fit, in tigramite's interpreter, finds the one dependent pair."""
    np.save(tmp_path / "x.npy", _nonlinear_data())
    script = (
        "import sys, numpy as np\n"
        f"sys.path.insert(0, {str(Path(nonlinear_ci_cpu.__file__).parent)!r})\n"
        "from andrey_bench.adapters.tigramite_pc import TigramitePC\n"
        f"a = TigramitePC({test!r})\n"
        f"np.save({str(tmp_path / 'adj.npy')!r}, a.fit(np.load({str(tmp_path / 'x.npy')!r}), "
        "a.params()))\n"
    )
    subprocess.run([os.environ[PYTHON_ENV_VAR], "-c", script], check=True, timeout=600)
    native = np.load(tmp_path / "adj.npy")
    adjacent = {(i, j) for i in range(3) for j in range(i + 1, 3) if native[i, j] or native[j, i]}
    assert adjacent == {(0, 1)}
    assert TigramitePC(test).to_structure(native).kind == "cpdag"
    assert tigramite_pc_adapter(test).package_version not in ("", "unknown")
