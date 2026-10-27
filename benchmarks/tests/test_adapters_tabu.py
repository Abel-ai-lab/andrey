"""The tabu-search adapters and the BIC the tabu campaign scores every fit with.

Andrey's and bnlearn's tabu adapters must run the same search: the same penalty, tabu list, and
stopping rule (bnlearn's Gaussian likelihood differs from Andrey's by an O(p^2 / n) term per node). The campaign's ``bic`` must be the score Andrey's searches report, the same for every DAG of
an equivalence class, so a DAG row and a CPDAG row compare on it.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest
import tabu_cpu
from conftest import needs_r

import andrey
from andrey.data import sample_scm
from andrey.metrics import to_cpdag
from andrey_bench.adapters.andrey_hc import andrey_hc_adapters
from andrey_bench.adapters.andrey_tabu import andrey_tabu_adapters
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.bnlearn_tabu import BnlearnTabu


def _sample(seed: int, d: int = 8):
    """A standardized linear-Gaussian sample on an Erdos-Renyi DAG of average degree 2."""
    return sample_scm(graph="erdos_renyi", d=d, n=500, seed=seed, density=2.0, scale="standardize")


def _bic(adapter, x: np.ndarray) -> float:
    adapter.setup()
    return tabu_cpu.graph_bic(x, adapter.to_structure(adapter.fit(x, adapter.params())))


def test_the_tabu_pair_runs_one_search() -> None:
    """Same BIC units and the same list; bnlearn 5.2.1 stops after `tabu` non-improving moves whatever
    `max.tabu` is, so Andrey's `max_tabu` must equal it too."""
    theirs = BnlearnTabu().params()
    for adapter in andrey_tabu_adapters():
        mine = adapter.params()
        assert (mine["lambda_value"], theirs["score"]) == (1.0, "bic-g")
        assert mine["tabu_length"] == mine["max_tabu"] == theirs["tabu"] == theirs["max_tabu"]
        assert mine["max_iter"] >= 10_000  # bnlearn's max.iter is unbounded


@pytest.mark.parametrize("tabu", andrey_tabu_adapters(), ids=lambda a: a.name)
def test_andrey_tabu_never_scores_worse_than_andrey_hc(tabu) -> None:
    hc = next(a for a in andrey_hc_adapters() if a.mode == tabu.mode)
    for seed in range(5):
        x = _sample(seed).data
        assert _bic(tabu, x) <= _bic(hc, x)


def test_the_bic_is_the_score_andrey_reports() -> None:
    x = _sample(0).data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        out = andrey.tabu(x, ebic_gamma=0.0)
    assert tabu_cpu.graph_bic(x, out.structure) == pytest.approx(out.metadata["score"], rel=1e-12)


def test_the_bic_is_the_same_for_a_dag_and_its_class() -> None:
    sample = _sample(1)
    dag = sample.graph
    assert tabu_cpu.graph_bic(sample.data, dag) == pytest.approx(
        tabu_cpu.graph_bic(sample.data, to_cpdag(dag)), rel=1e-12
    )


def _run(tmp_path, *flags: str) -> pd.DataFrame:
    assert tabu_cpu.main(["--out", str(tmp_path), "--smoke", "--cores", "1", *flags]) == 0
    return pd.read_parquet(tmp_path / "runs.parquet")


def test_the_driver_scores_every_fit_against_hill_climbing(tmp_path, monkeypatch) -> None:
    """Without R: the Andrey solutions and the empty graph, then a rescore of the same run."""
    andrey_only = [a for a in tabu_cpu.SOLUTIONS if a.package != "bnlearn"]
    monkeypatch.setattr(tabu_cpu, "SOLUTIONS", andrey_only)
    monkeypatch.setattr(tabu_cpu.rsession, "check_environment", lambda *packages: {})
    table = _run(tmp_path)
    assert set(table["name"]) == {a.name for a in andrey_only}
    assert (table["status"] == "ok").all() and table["bic"].notna().all()
    gap = tabu_cpu.summary(table).set_index("name")["bic_vs_hc"]
    assert gap[tabu_cpu.REFERENCE] == 0.0
    assert gap["andrey.tabu.numpy.serial"] <= 0.0
    assert gap[BaselineEmpty().name] > 0.0

    rescored = _run(tmp_path, "--rescore")
    pd.testing.assert_series_equal(
        rescored.set_index("run_id")["bic"].sort_index(),
        table.set_index("run_id")["bic"].sort_index(),
    )


def test_the_driver_runs_only_the_named_solutions_and_without_r(tmp_path, monkeypatch) -> None:
    """An Andrey-only run, for sizes bnlearn does not reach, fits its solutions and starts no R."""
    pytest.importorskip("pyarrow", reason="the harness environment is not installed")

    def no_r(*packages):
        raise AssertionError(f"started R for {packages}")

    monkeypatch.setattr(tabu_cpu.rsession, "check_environment", no_r)
    names = ["andrey.tabu.numpy.serial", "andrey.hc.numpy.serial", BaselineEmpty().name]
    table = _run(tmp_path, "--solutions", ",".join(names))
    assert set(table["name"]) == set(names)
    assert (table["status"] == "ok").all()
    assert json.loads((tmp_path / "run_meta.json").read_text())["solution list"] == names


def test_an_unknown_solution_name_stops_the_run(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit):
        tabu_cpu.main(["--out", str(tmp_path), "--solutions", "andrey.tabu.numpy.serial,tabu"])
    assert "unknown solutions ['tabu']" in capsys.readouterr().err


@needs_r()
def test_the_driver_runs_every_solution_with_r(tmp_path) -> None:
    table = _run(tmp_path)
    assert set(table["name"]) == {a.name for a in tabu_cpu.SOLUTIONS}
    assert (table["status"] == "ok").all() and table["bic"].notna().all()
