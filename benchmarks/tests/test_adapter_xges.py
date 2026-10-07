"""The XGES adapter, and the second environment it brought with it.

Two layers, because XGES is the first solution that does not live in bench-env:

* everything that is a fact about the adapter or the runner - metadata, the penalty scale, the
  thread pin, the adjacency conversion, which interpreter a solution is routed to - is checked
  here, in bench-env, with no ``xges`` import;
* everything that is a fact about the environment boundary - that a fit really runs under
  ``numpy<2``, really gets the fast scorer, and really comes back as a graph this process can score
  - is checked by running one, and skipped when the venv has not been built.

The second layer is what makes the first honest: a routing test that never spawns the interpreter
would pass just as well if nothing on the other side worked.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from andrey import GraphStructure
from andrey_bench import runner
from andrey_bench.adapters.andrey_ges import andrey_ges_adapters
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.xges_ges import (
    DEFAULT_ALPHA,
    PYTHON_ENV_VAR,
    SHIPPED_ALPHA,
    XgesGES,
    configure_numba_threads,
    requested_threads,
    xges_adapter,
)
from andrey_bench.contracts import SolutionAdapter, run_id
from andrey_bench.scoring import structure_hash

D_NODES = 8
N_SAMPLES = 800

XGES_PYTHON = Path(runner.BENCH_ROOT) / ".venv-xges" / "bin" / "python"

needs_xges_env = pytest.mark.skipif(
    not XGES_PYTHON.exists(), reason="xges-env not built - run benchmarks/start_xges.sh"
)


@pytest.fixture
def xges_env(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``$ANDREY_BENCH_XGES_PYTHON`` at the built venv for the duration of one test."""
    monkeypatch.setenv(PYTHON_ENV_VAR, str(XGES_PYTHON))
    return XGES_PYTHON


def _fake_interpreter(path: Path) -> Path:
    """An executable stand-in for a venv python: routing is a path question, not a run."""
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def _tiny_dataset() -> np.ndarray:
    """A fixed small linear-Gaussian ER sample from Andrey's own generators (bench-env only)."""
    import andrey.data as data

    ds = data.benchmarks.scm("linear_gauss_er", D_NODES).sample(n=N_SAMPLES, seed=0)
    return np.asarray(ds.data, dtype=np.float64)


# --- metadata and settings (no fit, no xges) -----------------------------------------------------


def test_metadata() -> None:
    """The runner-visible surface, plus the ``python_env`` that routes it elsewhere."""
    adapter = XgesGES()
    assert isinstance(adapter, SolutionAdapter)
    assert adapter.name == "xges.ges"
    assert adapter.package == "xges"
    assert adapter.backend == "numba"
    assert adapter.mode == "serial"
    assert adapter.algorithm == "ges"
    assert adapter.output_type == "cpdag"
    assert adapter.python_env == PYTHON_ENV_VAR


def test_the_campaign_penalty_matches_andreys() -> None:
    """``alpha`` and ``lambda_value`` scale the same ``log n`` per-parameter term.

    XGES ships ``alpha=2.0``, twice Andrey's weight, so leaving both defaults alone would have put a
    sparsity setting in the quality column. This is the claim the campaign rests on; a change to
    either default should fail here rather than in the table.
    """
    andrey_lambda = andrey_ges_adapters()[0].params()["lambda_value"]
    assert XgesGES().params()["alpha"] == andrey_lambda
    assert DEFAULT_ALPHA == andrey_lambda


def test_params_record_that_there_is_no_parent_cap() -> None:
    """XGES has no ``maxP``; the row says so rather than leaving the reader to assume a match."""
    params = XgesGES().params()
    assert params["extended_search"] is True
    assert params["use_fast_numba"] is True
    assert "max_parents" in params and params["max_parents"] is None
    # The comparison the null is against: Andrey's rows carry a number in the same field.
    assert andrey_ges_adapters()[0].params()["max_parents"] == 4
    # Recorded as JSON on every record, so the asymmetry survives into runs.parquet.
    assert json.loads(json.dumps(params, sort_keys=True))["max_parents"] is None


def test_the_penalty_is_carried_through_to_params() -> None:
    """``--xges-alpha`` reaches the record: a driver's value is not replaced by a default."""
    assert XgesGES(alpha=2.0).params()["alpha"] == 2.0


def test_two_penalties_are_two_solutions() -> None:
    """A second penalty needs a second name, because ``params`` are not part of a ``run_id``.

    Sharing a name would put both arms in one part file: the second unit reads as already measured,
    a resume skips it, and the table publishes one penalty under two headings.
    """
    matched = xges_adapter(alpha=DEFAULT_ALPHA)
    shipped = xges_adapter(alpha=SHIPPED_ALPHA, name="xges.ges.shipped")
    assert matched.name != shipped.name
    assert (matched.params()["alpha"], shipped.params()["alpha"]) == (1.0, 2.0)

    env = {
        "backend": "numba",
        "mode": "serial",
        "dtype": "float64",
        "device": "cpu",
        "cap_mem_mb": None,
        "cap_wall_s": 900.0,
        "warmup": 1,
        "machine_id": "m",
        "threads": 16,
        "num_workers": 1,
    }
    ids = {run_id(dataset_id="d", name=a.name, repeat=0, env=env) for a in (matched, shipped)}
    assert len(ids) == 2


def test_the_shipped_arm_is_the_package_default() -> None:
    """The reference arm is a reference only if it is what ``pip install xges`` searches."""
    assert SHIPPED_ALPHA == 2.0
    assert SHIPPED_ALPHA != DEFAULT_ALPHA


# --- the thread pin ------------------------------------------------------------------------------


def test_threads_come_from_the_allotted_budget_not_the_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OMP_NUM_THREADS`` is the harness's split; the affinity mask is the whole job."""
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    # Recorded before the call writes it, so monkeypatch takes it back out afterwards: this process
    # spawns children that inherit its environment, and a leaked pin would follow them.
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    assert requested_threads() == 3
    assert configure_numba_threads() == 3
    assert os.environ["NUMBA_NUM_THREADS"] == "3"


def test_the_thread_budget_falls_back_to_one_when_unpinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unpinned or unparseable budget is one thread, never the node's core count."""
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert requested_threads() == 1
    monkeypatch.setenv("OMP_NUM_THREADS", "not-a-number")
    assert requested_threads() == 1


# --- output conversion ---------------------------------------------------------------------------


def test_adjacency_converts_to_the_marks_xges_means() -> None:
    """``adj[x,y]=1`` is ``x->y``; a symmetric pair is one undirected edge. No transpose.

    Getting the orientation backwards is silent - every metric still computes, on a reversed graph.
    """
    adj = np.zeros((3, 3), dtype=np.int8)
    adj[0, 1] = 1  # 0 -> 1, directed
    adj[1, 2] = adj[2, 1] = 1  # 1 - 2, undirected

    structure = XgesGES().to_structure(adj)
    assert isinstance(structure, GraphStructure)
    assert structure.kind == "cpdag"
    assert structure.n_nodes == 3
    structure.validate()

    marks = np.asarray(structure.to_numpy())
    _TAIL, _ARROW = 1, 2
    assert (marks[0, 1], marks[1, 0]) == (_TAIL, _ARROW)  # tail at the source, arrow at the target
    assert (marks[1, 2], marks[2, 1]) == (_TAIL, _TAIL)  # undirected
    assert marks[0, 2] == marks[2, 0] == 0


def test_an_empty_adjacency_is_the_empty_graph() -> None:
    """The degenerate answer: XGES returning nothing must score as the floor, not as an error."""
    empty = XgesGES().to_structure(np.zeros((5, 5), dtype=np.int8))
    assert structure_hash(empty) == structure_hash(
        BaselineEmpty().to_structure(np.zeros((5, 5), dtype=np.int8))
    )


# --- interpreter routing -------------------------------------------------------------------------


def test_a_solution_with_its_own_environment_is_routed_to_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``python_env`` names the variable; the runner reads it and spawns there."""
    fake = _fake_interpreter(tmp_path / "python")
    monkeypatch.setenv(PYTHON_ENV_VAR, str(fake))
    assert runner._bench_python(XgesGES()) == fake


def test_its_environment_outranks_the_shared_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``ANDREY_BENCH_PYTHON`` redirects the shared build and says nothing about this solution's.

    Were it the other way round, measuring one checkout against another would silently move XGES
    into an interpreter that has no xges in it.
    """
    own = _fake_interpreter(tmp_path / "own")
    shared = _fake_interpreter(tmp_path / "shared")
    monkeypatch.setenv(PYTHON_ENV_VAR, str(own))
    monkeypatch.setenv("ANDREY_BENCH_PYTHON", str(shared))
    assert runner._bench_python(XgesGES()) == own
    # A solution that declares no environment of its own still follows the shared override.
    assert runner._bench_python(BaselineEmpty()) == shared


def test_a_declared_environment_that_is_missing_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unset, absent and non-executable are three ways to name an interpreter that cannot run."""
    monkeypatch.delenv(PYTHON_ENV_VAR, raising=False)
    with pytest.raises(FileNotFoundError, match=PYTHON_ENV_VAR):
        runner._bench_python(XgesGES())

    monkeypatch.setenv(PYTHON_ENV_VAR, str(tmp_path / "not-there"))
    with pytest.raises(FileNotFoundError, match="existing interpreter"):
        runner._bench_python(XgesGES())

    # Present but not executable: spawning it would fail once per fit, deep in a subprocess launch.
    not_exec = tmp_path / "python"
    not_exec.write_text("")
    not_exec.chmod(0o644)
    monkeypatch.setenv(PYTHON_ENV_VAR, str(not_exec))
    with pytest.raises(FileNotFoundError, match="not an executable file"):
        runner._bench_python(XgesGES())

    # A directory, which is the venv itself - the likeliest thing to point this at by mistake, and
    # the one an execute-bit check alone waves through, because on a directory that bit is search
    # permission.
    venv_dir = tmp_path / "venv"
    (venv_dir / "bin").mkdir(parents=True)
    monkeypatch.setenv(PYTHON_ENV_VAR, str(venv_dir))
    assert os.access(venv_dir, os.X_OK), "the directory must be searchable, or this tests nothing"
    with pytest.raises(FileNotFoundError, match="not an executable file"):
        runner._bench_python(XgesGES())


def test_a_missing_environment_stops_the_batch_before_anything_is_measured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pre-flight, not the per-unit handler, is what makes this fail fast.

    ``run_batch`` catches a launch failure per (dataset, solution) and carries on, right for a fit
    that died and wrong for an environment that was never built: without the pre-flight the
    whole ladder runs before anyone learns the campaign's competitor was never measured. Asserted by
    checking that the dataset store is still empty afterwards.
    """
    from andrey_bench.datasets import DatasetKey
    from andrey_bench.integration import run_batch

    monkeypatch.delenv(PYTHON_ENV_VAR, raising=False)
    store = tmp_path / "store"
    key = DatasetKey(
        topology="er",
        functional="linear",
        noise="gaussian",
        standardize="standardized",
        density=2.0,
        d=6,
        n=60,
        seed=0,
    )
    with pytest.raises(FileNotFoundError, match=PYTHON_ENV_VAR):
        run_batch(store, [XgesGES(), BaselineEmpty()], [key], machine_id="test")
    assert not store.exists() or not list(store.glob("*.npz"))


# --- the environment boundary, exercised ---------------------------------------------------------


@needs_xges_env
def test_the_fit_environment_is_the_one_xges_needs(xges_env: Path) -> None:
    """numpy 1 on the other side of the boundary, numpy 2 on this one - in one run.

    This is the whole reason for the second venv, and the only test that would notice bench-env
    quietly acquiring an ``xges`` that could not have run.
    """
    probe = "import numpy, xges; print(numpy.__version__, xges.__version__)"
    out = subprocess.run([str(xges_env), "-c", probe], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    child_numpy, child_xges = out.stdout.split()
    assert int(child_numpy.split(".")[0]) == 1, f"xges-env is on numpy {child_numpy}"
    assert int(np.__version__.split(".")[0]) >= 2, "bench-env is no longer numpy 2"
    assert child_xges


@needs_xges_env
def test_numba_runs_the_thread_count_the_harness_allotted(xges_env: Path) -> None:
    """The pin is applied in the child, where numba reads it - not merely set in the parent.

    Two budgets, because a single value could be the node's core count by coincidence.
    """
    probe = (
        "from andrey_bench.adapters.xges_ges import configure_numba_threads;"
        "want = configure_numba_threads();"
        "import numba;"
        "print(want, numba.get_num_threads(), numba.config.NUMBA_NUM_THREADS)"
    )
    for budget in ("2", "3"):
        env = {**os.environ, "OMP_NUM_THREADS": budget, "PYTHONPATH": str(runner.BENCH_ROOT)}
        env.pop("NUMBA_NUM_THREADS", None)
        out = subprocess.run(
            [str(xges_env), "-c", probe], capture_output=True, text=True, env=env, timeout=300
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.split() == [budget, budget, budget], out.stdout


@needs_xges_env
def test_a_fit_runs_there_and_comes_back_a_graph_this_process_can_score(xges_env: Path) -> None:
    """The whole path: spawn in xges-env, fit, return an adjacency, convert and score here.

    ``run_task`` rather than ``adapter.fit`` on purpose - ``fit`` cannot run in this interpreter at
    all, so calling it directly would test something no benchmark does.
    """
    adapter = XgesGES()
    measures = runner.run_task(
        adapter,
        _tiny_dataset(),
        adapter.params(),
        cap_wall_s=600.0,
        repeats=1,
        warmup=0,
        threads=2,
    )
    assert len(measures) == 1
    measure = measures[0]
    assert measure["status"] == "ok", measure.get("stderr") or measure.get("error_text")
    assert measure["threads"] == 2

    adj = np.asarray(measure["adj"])
    assert adj.shape == (D_NODES, D_NODES)
    assert set(np.unique(adj).tolist()) <= {0, 1}

    structure = adapter.to_structure(adj)
    assert structure.kind == "cpdag"
    assert structure.n_nodes == D_NODES
    structure.validate()
    # A fit that recovered nothing would still convert and score; it is the floor it must clear.
    assert structure_hash(structure) != structure_hash(
        BaselineEmpty().to_structure(np.zeros((D_NODES, D_NODES), dtype=np.int8))
    )


@needs_xges_env
def test_the_fast_scorer_assert_is_what_it_claims(xges_env: Path) -> None:
    """``use_fast_numba=True`` must fail rather than fall back.

    XGES swallows the ``ImportError`` from its numba scorer and carries on with the pure-Python one.
    Hiding the fast scorer reproduces exactly that, so the guard is checked against the failure it
    exists for rather than against a hypothetical.
    """
    probe = (
        "import sys;"
        "sys.modules['xges.bic_scorer_fast'] = None;"
        "import numpy as np;"
        "from andrey_bench.adapters.xges_ges import XgesGES;"
        "a = XgesGES();"
        "a.fit(np.random.default_rng(0).normal(size=(200, 5)), a.params())"
    )
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": str(runner.BENCH_ROOT)}
    out = subprocess.run(
        [str(xges_env), "-c", probe], capture_output=True, text=True, env=env, timeout=300
    )
    assert out.returncode != 0, "the fallback scorer was accepted"
    assert "BICScorerFast" in out.stderr, out.stderr
