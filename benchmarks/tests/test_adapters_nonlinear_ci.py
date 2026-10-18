"""The adapters and drivers of the nonlinear CI lane: PC and FCI with KCI, RCoT, CMIknn, and GPDC,
and the per-query comparison.

Every solution runs at Andrey's defaults under the package's own names, and every fit returns a
graph the harness scores the same way. tigramite fits only in its own environment, so its fit
runs when ``ANDREY_BENCH_TIGRAMITE_PYTHON`` names one; its graph conversion is checked here.
"""

from __future__ import annotations

import inspect
import os
import subprocess
from pathlib import Path

import nonlinear_ci_cpu
import nonlinear_ci_queries
import numpy as np
import pytest

import andrey
from andrey import GraphStructure
from andrey_bench import datasets
from andrey_bench.adapters.andrey_fci import AndreyFCI
from andrey_bench.adapters.andrey_pc import AndreyPC
from andrey_bench.adapters.causal_learn_fci import CausalLearnFCI
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


@pytest.mark.parametrize("method", ["pc", "fci"])
def test_every_solution_has_its_own_name_and_test(method: str) -> None:
    field = nonlinear_ci_cpu.solutions(method)
    names = [a.name for a in field]
    assert len(set(names)) == len(names)
    assert {a.algorithm for a in field if a.package != "baseline"} == {method}
    tested = {(a.package, a.params().get("indep_test")) for a in field if a.package != "baseline"}
    assert {("andrey", t) for t in ("fisherz", *NONLINEAR)} <= tested
    assert {("causal-learn", "kci"), ("causal-learn", "rcot")} <= tested
    if method == "pc":
        assert {("tigramite", "cmiknn"), ("tigramite", "gpdc"), ("pcalg+RCIT", "rcot")} <= tested


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


def test_the_nonmonotone_regime_hides_its_edges_from_partial_correlation() -> None:
    """On ``nonmonotone_er`` Fisher-Z misses edges RCoT finds: the regime's purpose."""
    (key,) = nonlinear_ci_cpu.ladder_keys([8], [1000], seeds=1, regime="nonmonotone_er")
    data, marks, *_ = datasets.generate_one(key)
    truth = np.asarray(marks) != 0
    errors = {}
    for test in ("fisherz", "rcot"):
        found = np.asarray(andrey.pc(data, indep_test=test).structure.to_numpy()) != 0
        errors[test] = int(np.triu((found | found.T) ^ (truth | truth.T), 1).sum())
    assert errors["rcot"] <= 1 < 3 <= errors["fisherz"]


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


@pytest.mark.parametrize("cls", [AndreyPC, CausalLearnPC, TigramitePC, CausalLearnFCI])
def test_an_unknown_test_is_rejected(cls) -> None:
    with pytest.raises(ValueError, match="indep_test"):
        cls("no_such_test")


@pytest.mark.parametrize("test", ["rcot", "kci"])
def test_fci_runs_each_test(test: str) -> None:
    """Andrey's FCI variant finds the one dependent pair; causal-learn's returns a PAG."""
    adapter = AndreyFCI(indep_test=test)
    assert adapter.name == f"andrey.fci.{test}.numpy"
    adapter.setup()
    structure = adapter.to_structure(adapter.fit(_nonlinear_data(), adapter.params()))
    adjacent = np.asarray(structure.to_numpy()) != 0
    assert structure.kind == "pag" and set(zip(*np.nonzero(np.triu(adjacent)))) == {(0, 1)}
    pytest.importorskip("causallearn")
    reference = CausalLearnFCI(test)
    assert reference.name == f"causal-learn.fci.{test}"
    native = reference.fit(_nonlinear_data(), reference.params())
    assert reference.to_structure(native).kind == "pag"


def test_query_summary_reads_level_power_and_agreement() -> None:
    """Level and power are rejection shares at alpha; agreement compares each reference's
    decisions with Andrey's test of the same kind on the same draws; a timeout is not counted."""
    records = []
    for name, ps in [
        ("andrey.kci", {("null", 0): 0.01, ("null", 1): 0.5, ("alt", 0): 0.001}),
        ("causal-learn.kci", {("null", 0): 0.2, ("null", 1): 0.6, ("alt", 0): 0.002}),
        ("rcit.kcit", {("null", 0): 0.03, ("null", 1): None, ("alt", 0): 0.004}),
    ]:
        for (kind, seed), p in ps.items():
            answer = {"status": "timeout"} if p is None else {"status": "ok", "p": p, "seconds": 1}
            records.append(
                {"contestant": name, "n": 50, "k": 1, "kind": kind, "seed": seed, **answer}
            )
    rows = {r["contestant"]: r for r in nonlinear_ci_queries.summary(records, alpha=0.05)}
    assert (rows["andrey.kci"]["level"], rows["andrey.kci"]["power"]) == (0.5, 1.0)
    assert rows["andrey.kci"]["agreement"] is None
    assert rows["causal-learn.kci"]["agreement"] == pytest.approx(2 / 3)
    assert (rows["rcit.kcit"]["answered"], rows["rcit.kcit"]["agreement"]) == (2, 1.0)


def test_query_driver_records_every_draw(tmp_path: Path) -> None:
    """One record per contestant and draw, and a summary row per cell."""
    out = tmp_path / "q"
    args = ["--out", str(out), "--contestants", "andrey.rcot,andrey.kci", "--rows", "60"]
    assert nonlinear_ci_queries.main([*args, "--k", "0", "1", "--null", "2", "--alt", "1"]) == 0
    lines = (out / "records.jsonl").read_text().splitlines()
    assert len(lines) == 2 * 2 * 3
    assert len((out / "summary.csv").read_text().splitlines()) == 1 + 2 * 2


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
