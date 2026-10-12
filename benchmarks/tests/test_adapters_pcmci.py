"""The PCMCI adapters: Andrey's two tests, tigramite in its own environment, and the empty stack.

Two layers, as for XGES. Everything that is a fact about an adapter - metadata, names, the
orientation of a link stack, which interpreter a solution is routed to - is checked in bench-env
with no tigramite import. Everything that is a fact about the environment boundary - that a
tigramite fit runs there, comes back as a stack this process can score, and finds the graph
Andrey's Student-t PCMCI finds - is checked by running one, and skipped when ``.venv-tigramite``
has not been built.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import andrey
from andrey_bench import datasets, runner, scoring
from andrey_bench.adapters.andrey_pcmci import INDEP_TESTS, andrey_pcmci_adapters
from andrey_bench.adapters.baseline_empty import baseline_empty_temporal
from andrey_bench.adapters.tigramite_pcmci import (
    PYTHON_ENV_VAR,
    TigramitePCMCI,
    links_from_graph,
    tigramite_adapter,
)
from andrey_bench.contracts import SolutionAdapter, links_from_temporal
from andrey_bench.integration import _truth

TIGRAMITE_PYTHON = Path(runner.BENCH_ROOT) / ".venv-tigramite" / "bin" / "python"

needs_tigramite_env = pytest.mark.skipif(
    not TIGRAMITE_PYTHON.exists(),
    reason="tigramite-env not built - run benchmarks/start_tigramite.sh",
)


@pytest.fixture
def tigramite_env(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``$ANDREY_BENCH_TIGRAMITE_PYTHON`` at the built venv for one test."""
    monkeypatch.setenv(PYTHON_ENV_VAR, str(TIGRAMITE_PYTHON))
    return TIGRAMITE_PYTHON


def _series(seed: int = 0, d: int = 4, steps: int = 400) -> np.ndarray:
    """A lagged series with the planted link X0(t-1) -> X1(t) and every series' own past."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((steps, d))
    for t in range(1, steps):
        x[t] += 0.4 * x[t - 1]
        x[t, 1] += 0.7 * x[t - 1, 0]
    return x


def _lagged_dataset(tmp_path: Path, d: int, steps: int, seed: int) -> datasets.DatasetKey:
    topology, functional, noise, standardize = datasets.REGIMES["var_lagged"]
    key = datasets.DatasetKey(
        topology=topology,
        functional=functional,
        noise=noise,
        standardize=standardize,
        density=datasets.TEMPORAL_REGIMES["var_lagged"],
        d=d,
        n=steps,
        seed=seed,
    )
    datasets.materialize([key], tmp_path)
    return key


# --- metadata (no fit, no tigramite) -------------------------------------------------------------


def test_each_test_and_window_is_a_solution_of_its_own() -> None:
    """``params`` are not in a ``run_id``, so every setting that changes a fit is in the name."""
    field = [*andrey_pcmci_adapters(3), tigramite_adapter(3), baseline_empty_temporal(3)]
    assert [a.name for a in field] == [
        "andrey.pcmci.student_t.tau3",
        "andrey.pcmci.fisherz.tau3",
        "tigramite.pcmci.parcorr.tau3",
        "baseline.empty.temporal.tau3",
    ]
    assert [a.params()["indep_test"] for a in field[:2]] == list(INDEP_TESTS)
    for adapter in field:
        assert isinstance(adapter, SolutionAdapter)
        assert adapter.output_type == "temporal"
        assert adapter.params()["tau_max"] == 3


def test_both_packages_run_the_same_settings() -> None:
    andrey_params = andrey_pcmci_adapters(2)[0].params()
    tigramite_params = TigramitePCMCI().params()
    for name in ("tau_max", "pc_alpha", "alpha_level"):
        assert andrey_params[name] == tigramite_params[name]
    # Lag 0 is PCMCI+'s scope: tigramite tests it at its own tau_min=0.
    assert tigramite_params["tau_min"] == 1
    assert tigramite_params["fdr_method"] == "none"
    assert tigramite_params["significance"] == "analytic"


def test_tigramite_fits_are_routed_to_their_own_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = tigramite_adapter(2)
    assert adapter.python_env == PYTHON_ENV_VAR
    monkeypatch.delenv(PYTHON_ENV_VAR, raising=False)
    with pytest.raises(FileNotFoundError, match=PYTHON_ENV_VAR):
        runner.check_interpreters([adapter])
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv(PYTHON_ENV_VAR, str(fake))
    assert runner.check_interpreters([adapter])[adapter.name] == fake


def test_the_tigramite_module_imports_nothing_from_andrey() -> None:
    """The worker in tigramite's environment unpickles the adapter from this module, and that
    environment has no Andrey."""
    code = (
        "import sys, andrey_bench.adapters.tigramite_pcmci; "
        "assert not [m for m in sys.modules if m.split('.')[0] in ('andrey', 'tigramite')]"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=runner.BENCH_ROOT,
        env={"PYTHONPATH": str(runner.BENCH_ROOT)},
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


def test_tigramite_graph_entries_become_links_at_their_lag() -> None:
    graph = np.full((3, 3, 3), "", dtype="<U3")
    graph[0, 1, 1] = "-->"  # X0(t-1) -> X1(t)
    graph[2, 2, 2] = "-->"  # X2(t-2) -> X2(t)
    graph[0, 2, 0] = "o-o"  # lag 0 is not tested at tau_min=1 and is dropped
    links = links_from_graph(graph)
    assert links.shape == (2, 3, 3)
    assert {tuple(map(int, t)) for t in np.argwhere(links)} == {(0, 0, 1), (1, 2, 2)}

    graph[1, 0, 1] = "o-o"
    with pytest.raises(RuntimeError, match="o-o"):
        links_from_graph(graph)


# --- Andrey fits ---------------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", andrey_pcmci_adapters(2), ids=lambda a: a.name)
def test_the_andrey_link_stack_is_the_structure_andrey_returns(adapter) -> None:
    x = _series()
    links = adapter.fit(x, adapter.params())
    out = andrey.pcmci(x, tau_max=2, indep_test=adapter.indep_test)
    assert links.shape == (2, 4, 4) and links.dtype == np.int8
    assert adapter.to_structure(links) == out.structure
    np.testing.assert_array_equal(links, np.asarray(out.metadata["p_values"]) <= 0.05)
    assert links[0, 0, 1] == 1  # the planted link


def test_the_empty_stack_covers_the_window() -> None:
    adapter = baseline_empty_temporal(3)
    links = adapter.fit(_series(), adapter.params())
    assert links.shape == (3, 4, 4) and not links.any()
    assert adapter.to_structure(links).lags == (1, 2, 3)


# --- the tigramite environment ---------------------------------------------------------------------


@needs_tigramite_env
def test_a_tigramite_fit_runs_there_and_is_scored_here(tigramite_env: Path, tmp_path: Path) -> None:
    key = _lagged_dataset(tmp_path, d=5, steps=300, seed=0)
    adapter = tigramite_adapter(2)
    assert adapter.package_version == "5.2.10.1"
    (measures,) = runner.run_task(
        adapter,
        datasets.data_path(tmp_path, key),
        adapter.params(),
        cap_wall_s=240.0,
        repeats=1,
        warmup=0,
    )
    assert measures["status"] == "ok", measures.get("stderr")
    estimate = adapter.to_structure(measures["adj"])
    _, marks = datasets.load(tmp_path, key)
    scores = scoring.score_result(estimate, _truth(marks, "temporal"))
    assert scores["family"] == "temporal"
    assert scores["lag1_recall"] > 0.5


@needs_tigramite_env
@pytest.mark.parametrize("d, steps, seed", [(4, 300, 0), (6, 500, 1)])
def test_tigramite_and_andrey_student_t_find_the_same_graph(
    tigramite_env: Path, tmp_path: Path, d: int, steps: int, seed: int
) -> None:
    """``ParCorr(significance="analytic")`` and Andrey's ``student_t`` share their test."""
    key = _lagged_dataset(tmp_path, d=d, steps=steps, seed=seed)
    tigramite = tigramite_adapter(2)
    (measures,) = runner.run_task(
        tigramite,
        datasets.data_path(tmp_path, key),
        tigramite.params(),
        cap_wall_s=240.0,
        repeats=1,
        warmup=0,
    )
    assert measures["status"] == "ok", measures.get("stderr")
    andrey_student_t = andrey_pcmci_adapters(2)[0]
    series, _ = datasets.load(tmp_path, key)
    links = andrey_student_t.fit(series, andrey_student_t.params())
    np.testing.assert_array_equal(measures["adj"], links)
    assert scoring.structure_hash(tigramite.to_structure(measures["adj"])) == (
        scoring.structure_hash(andrey_student_t.to_structure(links))
    )
    assert links_from_temporal(andrey_student_t.to_structure(links)).sum() == links.sum()
