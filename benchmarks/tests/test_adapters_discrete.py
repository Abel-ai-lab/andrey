"""The categorical-data adapters, their regime, and agreement with causal-learn.

Andrey's chi-square and G-square p-values and BDeu scores follow causal-learn's definitions, so
they must agree with causal-learn's to rounding; on the same data both packages' PC and GES must
return the same graph. Each adapter must also meet the harness contract the shared contract test
checks for the others, bnlearn's and pgmpy's included: they construct without R or pgmpy, so this
file never skips. What each of those packages computes is checked in
``test_adapters_discrete_comparators.py``, where R or pgmpy is installed.
"""

from __future__ import annotations

import inspect
import json
import pickle
import sys
import warnings
from itertools import combinations

import discrete_cpu
import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.core.ci import ChiSquare
from andrey.core.score import BDeuScore, CategoricalBICScore
from andrey.data import sample_discrete
from andrey.search.hc import hc
from andrey_bench import datasets, rsession
from andrey_bench.adapters import andrey_hc
from andrey_bench.adapters.andrey_pc import AndreyPC
from andrey_bench.contracts import SolutionAdapter
from andrey_bench.datasets import DatasetKey
from andrey_bench.scoring import structure_hash

ADAPTERS = [a for pair in discrete_cpu.PAIRS.values() for a in pair]
ANDREY_HC = [discrete_cpu._HC_BDEU, discrete_cpu._HC_BIC]

#: The module each Andrey adapter's ``fit`` imports on first use.
DEFERRED_MODULES = {
    "andrey.pc.chisq": "andrey.constraint.pc",
    "andrey.pc.gsq": "andrey.constraint.pc",
    "andrey.ges.bdeu": "andrey.search.ges",
    "andrey.boss.bdeu": "andrey.search.boss",
    "andrey.grasp.bdeu": "andrey.search.grasp",
    "andrey.hc.BDeu": "andrey.search.hc",
    "andrey.hc.BIC_categorical": "andrey.search.hc",
}


@pytest.mark.parametrize("adapter", discrete_cpu.SOLUTIONS, ids=lambda a: a.name)
def test_an_adapter_meets_the_harness_contract(adapter: SolutionAdapter) -> None:
    assert isinstance(adapter, SolutionAdapter)
    json.dumps(adapter.params(), sort_keys=True)
    clone = pickle.loads(pickle.dumps(adapter))
    assert (clone.name, clone.params()) == (adapter.name, adapter.params())


def test_every_solution_has_its_own_name() -> None:
    names = [a.name for a in discrete_cpu.SOLUTIONS]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("method", list(discrete_cpu.PAIRS))
def test_both_packages_run_one_method_with_one_setting(method: str) -> None:
    mine, theirs = discrete_cpu.PAIRS[method]
    assert (mine.algorithm, mine.output_type, mine.mode) == (
        theirs.algorithm,
        theirs.output_type,
        theirs.mode,
    )
    assert mine.params() == theirs.params()


@pytest.mark.parametrize("pair", list(discrete_cpu.COMPARATOR_PAIRS))
def test_a_comparator_runs_the_method_and_settings_of_the_andrey_solution_it_is_paired_with(
    pair: str,
) -> None:
    """Equal ``params`` are the settings the two sides declare: the test or score, ``alpha`` or the
    priors and BIC weight, the move budget, and the parent cap."""
    mine, theirs = discrete_cpu.COMPARATOR_PAIRS[pair]
    assert mine.package == "andrey" and theirs.package in ("bnlearn", "pgmpy")
    assert (mine.algorithm, mine.mode) == (theirs.algorithm, theirs.mode)
    assert mine.params() == theirs.params()
    assert theirs.output_type == ("cpdag" if theirs.algorithm == "pc" else "dag")


def test_every_comparator_arm_is_a_named_solution_of_the_driver() -> None:
    names = {a.name for a in discrete_cpu.SOLUTIONS}
    for mine, theirs in discrete_cpu.COMPARATOR_PAIRS.values():
        assert {mine.name, theirs.name} <= names


@pytest.mark.parametrize("name", list(DEFERRED_MODULES))
def test_setup_imports_the_module_fit_defers(name: str) -> None:
    (adapter,) = [a for a in discrete_cpu.SOLUTIONS if a.name == name]
    sys.modules.pop(DEFERRED_MODULES[name], None)
    adapter.setup()
    assert DEFERRED_MODULES[name] in sys.modules


def _key(d: int = 6, n: int = 400, seed: int = 0) -> DatasetKey:
    return DatasetKey(**discrete_cpu.regime_slice(), d=d, n=n, seed=seed)


def test_the_regime_stores_category_codes_and_the_true_dag() -> None:
    key = _key()
    call = datasets.generation_kwargs(key)
    assert call["generator"] == "sample_discrete"
    assert {k: call[k] for k in datasets.DISCRETE_SETTINGS} == datasets.DISCRETE_SETTINGS
    data, marks, kind, _ = datasets.generate_one(key)
    assert kind == "dag" and marks.shape == (6, 6)
    assert data.shape == (400, 6) and np.array_equal(data, np.round(data))


def test_the_regime_rejects_rescaled_codes() -> None:
    key = DatasetKey(
        **{**discrete_cpu.regime_slice(), "standardize": "standardized"}, d=5, n=50, seed=0
    )
    with pytest.raises(ValueError, match="raw codes"):
        datasets.generation_kwargs(key)


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name)
def test_a_fit_gives_a_cpdag(adapter: SolutionAdapter) -> None:
    data, _, _, _ = datasets.generate_one(_key())
    adapter.setup()
    structure = adapter.to_structure(np.asarray(adapter.fit(data, adapter.params())))
    assert structure.kind == "cpdag" and structure.n_nodes == 6


# --- agreement with causal-learn ------------------------------------------------------------------


def _codes(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = 300
    x = np.column_stack([rng.integers(0, k, n) for k in (2, 3, 4, 3, 2)]).astype(float)
    x[:, 1] = np.where(rng.random(n) < 0.6, x[:, 0], x[:, 1])
    x[:, 2] = 3 * x[:, 2] - 7  # codes need not start at 0
    x[:, 4] = x[:, 3] % 2  # constant within each value of column 3: no degrees of freedom
    return x


@pytest.mark.parametrize("method", ["chisq", "gsq"])
def test_p_values_agree_with_causal_learn(method: str) -> None:
    from causallearn.utils.cit import CIT

    for seed in range(3):
        x = _codes(seed)
        mine, theirs = ChiSquare(x, g_square=method == "gsq"), CIT(x, method)
        for a, b in combinations(range(x.shape[1]), 2):
            rest = [c for c in range(x.shape[1]) if c not in (a, b)]
            for size in range(3):
                for s in combinations(rest, size):
                    assert mine(a, b, s) == pytest.approx(
                        theirs(a, b, list(s)), rel=1e-9, abs=1e-12
                    )


def test_bdeu_is_minus_twice_causal_learns_score() -> None:
    from causallearn.score.LocalScoreFunction import local_score_BDeu

    x = _codes(0)
    mine = BDeuScore(x)
    for node in range(x.shape[1]):
        rest = [c for c in range(x.shape[1]) if c != node]
        for size in range(3):
            for parents in combinations(rest, size):
                expected = -2 * local_score_BDeu(x, node, list(parents))
                assert mine.score(node, list(parents)) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("method", discrete_cpu.SAME_GRAPH_PAIRS)
def test_both_packages_return_the_same_graph(method: str, seed: int) -> None:
    """causal-learn's PC orients conflicting colliders in column order, so Andrey's PC runs that
    rule here; its default p-value order can orient them differently. The campaign runs the
    default, which this check does not cover."""
    data = sample_discrete(d=8, n=2000, seed=seed).data
    hashes = []
    for adapter in discrete_cpu.PAIRS[method]:
        adapter.setup()
        params = adapter.params()
        if isinstance(adapter, AndreyPC):
            params["collider_order"] = "column"
        native = adapter.fit(data, params)
        hashes.append(structure_hash(adapter.to_structure(np.asarray(native))))
    assert hashes[0] == hashes[1]


# --- Andrey's hill climbing with a categorical score ----------------------------------------------


@pytest.mark.parametrize("adapter", ANDREY_HC, ids=lambda a: a.name)
def test_a_hill_climbing_fit_runs_its_score_and_beats_the_empty_graph(adapter) -> None:
    data = sample_discrete(d=6, n=600, seed=1, concentration=0.5).data
    adapter.setup()
    native = np.asarray(adapter.fit(data, adapter.params()))
    structure = adapter.to_structure(native)
    assert structure.kind == "cpdag" and structure.n_nodes == 6
    cpdag, total = hc(data, score_func=adapter.score_func)
    assert np.array_equal(native, cpdag.to_numpy())
    score = (
        BDeuScore(data) if adapter.score_func == "local_score_BDeu" else CategoricalBICScore(data)
    )
    assert total < sum(score.score(j, []) for j in range(6))


def test_the_declared_settings_are_the_ones_andreys_hill_climbing_runs_with() -> None:
    """The adapters declare the priors and weight; ``hc`` builds them itself, so they must be its
    defaults."""
    bdeu = inspect.signature(BDeuScore).parameters
    declared = discrete_cpu._HC_BDEU.params()
    assert declared["sample_prior"] == bdeu["sample_prior"].default
    assert declared["structure_prior"] == bdeu["structure_prior"].default
    accepted = inspect.signature(hc).parameters
    assert discrete_cpu._HC_BIC.params()["lambda_value"] == accepted["lambda_value"].default
    assert "max_parents" not in accepted and "maxP" not in accepted  # no parent cap
    assert all(a.params()["max_parents"] is None for a in ANDREY_HC)
    assert "lambda_value" not in discrete_cpu._HC_BDEU.params()  # BDeu has no weight


def test_a_move_budget_stops_hill_climbing_early() -> None:
    data = sample_discrete(d=8, n=1000, seed=2).data
    adapter = discrete_cpu._HC_BIC
    free = adapter.fit(data, adapter.params())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.SearchLimitWarning)
        capped = adapter.fit(data, {**adapter.params(), "max_iter": 1})
    assert int((capped != 0).sum()) < int((free != 0).sum())


def test_a_categorical_hill_climbing_adapter_takes_only_a_categorical_score() -> None:
    with pytest.raises(ValueError, match="categorical score"):
        andrey_hc.andrey_hc_categorical("local_score_BIC_from_cov")


def test_the_continuous_hill_climbing_adapter_keeps_its_declared_settings() -> None:
    """Other lanes record these in ``run_meta.json``; a change would break their ``--resume``."""
    (adapter,) = andrey_hc.andrey_hc_adapters()
    assert adapter.params() == {
        "score_func": "local_score_BIC_from_cov",
        "lambda_value": 1.0,
        "max_iter": 100_000,
    }


# --- the driver -----------------------------------------------------------------------------------


def _run(tmp_path, *flags: str) -> pd.DataFrame:
    assert discrete_cpu.main(["--out", str(tmp_path), "--smoke", "--cores", "1", *flags]) == 0
    return pd.read_parquet(tmp_path / "runs.parquet")


def test_the_driver_runs_only_the_named_solutions_and_starts_no_r(tmp_path, monkeypatch) -> None:
    """A run of Andrey's new arms fits only those, needs neither R nor pgmpy, and scores a CPDAG."""

    def no_r(*packages):
        raise AssertionError(f"started R for {packages}")

    monkeypatch.setattr(rsession, "check_environment", no_r)
    names = ["andrey.hc.BDeu", "andrey.hc.BIC_categorical", "baseline.empty"]
    table = _run(tmp_path, "--solutions", *names)
    assert set(table["name"]) == set(names)
    assert (table["status"] == "ok").all() and table["shd"].notna().all()
    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert meta["solution list"] == names and "r environment" not in meta


def test_a_run_without_r_or_pgmpy_solutions_writes_no_extra_run_meta(tmp_path) -> None:
    """Such a run's run_meta.json has the keys of the lane before the comparators, so a run of the
    causal-learn lane's solutions still resumes."""
    _run(tmp_path, "--solutions", "andrey.pc.chisq", "andrey.ges.bdeu", "baseline.empty")
    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert set(meta) == {
        "driver",
        "solution list",
        "slice",
        "generator settings",
        "ladder",
        "n",
        "seeds",
        "repeats",
        "warmup",
        "cap_wall_s",
        "cap_mem_mb",
        "mem_route",
        "cores",
        "machine_id",
    }


def test_an_unknown_solution_name_stops_the_driver(tmp_path) -> None:
    with pytest.raises(SystemExit, match="unknown solutions"):
        discrete_cpu.main(["--out", str(tmp_path), "--solutions", "bnlearn.pc-stable.x2"])


def test_a_dags_comparable_columns_are_its_mec_ones() -> None:
    row = pd.Series(
        {
            "output_type": "dag",
            "shd": 9.0,
            "mec_shd": 2.0,
            "arrowhead_f1": 0.1,
            "mec_arrowhead_f1": 0.7,
        }
    )
    assert discrete_cpu.comparable(row, "shd") == 2.0
    assert discrete_cpu.comparable(row, "arrowhead_f1") == 0.7
    row["output_type"] = "cpdag"
    assert discrete_cpu.comparable(row, "shd") == 9.0
