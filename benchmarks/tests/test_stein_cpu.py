"""The Stein-score screen: the papers' simulation, the oracle arm, the order solutions, the driver.

Facts about the simulation, the oracle, and the driver are checked in bench-env. The reference
solutions (the authors' SCORE, NoGAM in dodiscover, CausalDisco) run in ``.venv-stein``; they are
checked by running each on one small dataset, and skipped when that environment has not been built.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import stein_cpu

from andrey_bench import datasets, gp_models, runner, score_oracle
from andrey_bench.adapters.andrey_cam import AndreyCAMOrder
from andrey_bench.adapters.andrey_nogam import AndreyNoGAM, AndreyNoGAMOrder
from andrey_bench.adapters.andrey_score_matching import (
    AndreyScoreMatching,
    AndreyScoreMatchingOrder,
)
from andrey_bench.adapters.causaldisco_r2sort import CausalDiscoR2Sort, CausalDiscoR2SortnRegress
from andrey_bench.adapters.dodiscover_nogam import DodiscoverNoGAM, DodiscoverNoGAMOrder
from andrey_bench.adapters.order_dag import order_dag
from andrey_bench.adapters.rolland_score import CODE_ENV, RollandSCORE, RollandSCOREOrder
from andrey_bench.datasets import DatasetKey

# Andrey endpoint marks.
_TAIL, _ARROW = 1, 2

STEIN_PYTHON = Path(runner.BENCH_ROOT) / ".venv-stein" / "bin" / "python"
SCORE_CODE = Path(runner.BENCH_ROOT) / ".score"
needs_stein_env = pytest.mark.skipif(
    not STEIN_PYTHON.exists() or not (SCORE_CODE / "stein.py").exists(),
    reason="stein env not built - run benchmarks/start_stein.sh",
)


def _edges(marks: np.ndarray) -> set[tuple[int, int]]:
    m = np.asarray(marks)
    return {(int(i), int(j)) for i, j in zip(*np.nonzero((m == _TAIL) & (m.T == _ARROW)))}


def _sample(
    noise="gaussian", standardize="raw", *, d=10, n=1000, seed=0, density=2.0, shuffle=True
):
    return gp_models.sample(
        "score2022",
        d=d,
        n=n,
        seed=seed,
        density=density,
        noise=noise,
        standardize=standardize,
        shuffle=shuffle,
    )


def _key(functional: str, noise: str, standardize: str, *, d: int, n: int, seed: int) -> DatasetKey:
    return DatasetKey(
        topology="er",
        functional=functional,
        noise=noise,
        density=2.0,
        d=d,
        n=n,
        seed=seed,
        standardize=standardize,
    )


# --- the papers' simulation ------------------------------------------------------------------------


@pytest.mark.parametrize(("density", "per_node"), [(2.0, 1), (8.0, 4)])
def test_er1_and_er4_have_d_and_4d_edges(density: float, per_node: int) -> None:
    for seed in range(5):
        data, marks, _ = _sample(d=12, n=50, seed=seed, density=density)
        assert data.shape == (50, 12) and marks.shape == (12, 12)
        assert len(_edges(marks)) == per_node * 12


def test_a_root_carries_noise_of_variance_between_04_and_08() -> None:
    variances = []
    for seed in range(30):
        data, marks, _ = _sample(d=4, n=2000, seed=seed)
        children = {c for _, c in _edges(marks)}
        variances += [data[:, j].var() for j in range(4) if j not in children]
    assert 0.38 < min(variances) and max(variances) < 0.84
    assert 0.55 < np.mean(variances) < 0.65


def test_the_three_scales_of_one_draw() -> None:
    raw, marks, _ = _sample(seed=3)
    z, z_marks, _ = _sample("gaussian", "standardized", seed=3)
    ancestral, a_marks, _ = _sample("gaussian", "ancestral", seed=3)
    np.testing.assert_array_equal(marks, z_marks)
    np.testing.assert_array_equal(marks, a_marks)
    np.testing.assert_allclose(z, (raw - raw.mean(axis=0)) / raw.std(axis=0))
    np.testing.assert_allclose(ancestral.std(axis=0), 1.0)
    # Rescaled in causal order, the links read inputs of unit variance: another draw than raw's.
    assert not np.allclose(ancestral, raw / raw.std(axis=0))


def test_the_shuffle_lists_the_columns_and_the_truth_in_one_order() -> None:
    plain, plain_marks, identity = _sample(seed=4, shuffle=False)
    data, marks, order = _sample(seed=4)
    np.testing.assert_array_equal(identity, np.arange(10))
    assert sorted(order.tolist()) == list(range(10)) and order.tolist() != list(range(10))
    np.testing.assert_array_equal(data, plain[:, order])
    np.testing.assert_array_equal(marks, plain_marks[np.ix_(order, order)])


def test_every_noise_and_scale_of_the_regimes_and_nothing_else() -> None:
    for regime in stein_cpu.REGIMES:
        _, functional, noise, scale = datasets.REGIMES[regime]
        key = _key(functional, noise, scale, d=5, n=30, seed=0)
        data, marks, kind, _ = datasets.generate_one(key)
        assert kind == "dag" and data.shape == (30, 5) and len(_edges(marks)) == 5
    with pytest.raises(ValueError):
        _sample("uniform")
    with pytest.raises(ValueError):
        _sample("gaussian", "rescale")


@pytest.mark.parametrize(("density", "paper_mean", "paper_sd"), [(2.0, 1.9, 1.1), (8.0, 9.7, 3.1)])
def test_varsort_orders_the_raw_data_as_in_the_score_paper(density, paper_mean, paper_sd) -> None:
    """Sorting by variance leaves as many true edges reversed as the SCORE paper reports.

    Rolland et al. (2022), Table 1, VarSort row: ``D_top`` at ``d = 10`` with Gaussian noise over
    10 runs, ER1 and ER4. The harness's mean over seeds 0-9 lies within one of the paper's standard
    deviations of the paper's mean.
    """
    reversed_edges = []
    for seed in range(10):
        data, marks, _ = _sample(seed=seed, density=density)
        full = order_dag(np.argsort(data.var(axis=0)))
        reversed_edges.append(sum(1 for i, j in _edges(marks) if not full[i, j]))
    assert abs(np.mean(reversed_edges) - paper_mean) <= paper_sd


# --- order solutions -------------------------------------------------------------------------------


def test_the_full_dag_of_an_order_joins_each_variable_to_every_later_one() -> None:
    expected = np.zeros((4, 4), dtype=np.int8)
    for i, j in [(2, 0), (2, 3), (2, 1), (0, 3), (0, 1), (3, 1)]:
        expected[i, j] = 1
    np.testing.assert_array_equal(order_dag([2, 0, 3, 1]), expected)


def test_andreys_cam_order_is_returned_as_endpoint_marks() -> None:
    import andrey

    data, _, _ = _sample(d=4, n=300, seed=1)
    marks = AndreyCAMOrder().fit(data, {})
    ordering = list(andrey.cam(data, alpha=0.001).ordering)
    expected = {(ordering[a], ordering[b]) for a in range(4) for b in range(a + 1, 4)}
    assert _edges(marks) == expected
    assert np.count_nonzero(marks) == 2 * len(expected)


@pytest.mark.parametrize(
    ("graph", "order", "method"),
    [
        (AndreyScoreMatching, AndreyScoreMatchingOrder, "score_matching"),
        (AndreyNoGAM, AndreyNoGAMOrder, "nogam"),
    ],
)
def test_andreys_stein_arms_return_the_facades_dag_and_the_full_dag_of_its_order(
    graph, order, method: str
) -> None:
    import andrey

    data, _, _ = _sample(d=4, n=300, seed=1)
    out = getattr(andrey, method)(data, alpha=0.001)
    marks = graph().fit(data, {})
    np.testing.assert_array_equal(marks, out.structure.to_numpy())
    assert graph().params() == {"alpha": 0.001}
    ordering = list(out.ordering)
    expected = {(ordering[a], ordering[b]) for a in range(4) for b in range(a + 1, 4)}
    order_marks = order().fit(data, {})
    assert _edges(order_marks) == expected
    assert np.count_nonzero(order_marks) == 2 * len(expected)
    assert graph.name == f"andrey.{method}.numpy" and order.name == f"andrey.{method}.order.numpy"


# --- the oracle arm --------------------------------------------------------------------------------


def _log_density(noise: str, r: np.ndarray, sigma: float) -> np.ndarray:
    if noise == "gaussian":
        return -(r**2) / (2 * sigma**2)
    beta = sigma * np.sqrt(6.0) / np.pi
    z = (r + beta * 0.5772156649015329) / beta
    return -np.log(beta) - z - np.exp(-z)


@pytest.mark.parametrize("noise", ["gaussian", "gumbel"])
def test_the_oracle_hessian_is_the_second_derivative_of_log_p(noise: str) -> None:
    """The closed form against a central difference of ``log p`` of the additive-noise model."""
    edges = np.array([[0, 1], [1, 2], [0, 2]])
    weights = np.array([0.8, -1.2, 0.5])
    sigmas = np.array([0.7, 1.0, 1.3])
    w = np.zeros((3, 3))
    w[edges[:, 0], edges[:, 1]] = weights

    def log_p(x: np.ndarray) -> np.ndarray:
        r = x - (x + np.sin(x)) @ w
        return sum(_log_density(noise, r[:, j], sigmas[j]) for j in range(3))

    x = np.random.default_rng(0).normal(size=(6, 3))
    closed = score_oracle.hessian_diagonal(x, edges, weights, sigmas, noise, [0, 1, 2])
    h = 1e-4
    for k in range(3):
        step = np.zeros(3)
        step[k] = h
        numeric = (log_p(x + step) - 2 * log_p(x) + log_p(x - step)) / h**2
        np.testing.assert_allclose(closed[:, k], numeric, rtol=1e-4, atol=1e-4)


def test_the_oracle_residuals_are_the_generators_noise() -> None:
    """The oracle's link ``z + sin z`` and noise scales are those of ``andrey.data``'s ``anm``."""
    import andrey.data as data

    key = _key("anm", "gumbel", "raw", d=6, n=20000, seed=1)
    ds = data.sample_scm(**{**datasets.generation_kwargs(key), "scale": "raw"})  # ty: ignore[invalid-argument-type]
    x, params = np.asarray(ds.data), ds.params
    w = np.zeros((6, 6))
    w[params.edges[:, 0], params.edges[:, 1]] = params.weights
    residual = x - (x + np.sin(x)) @ w
    np.testing.assert_allclose(residual.std(axis=0), params.noise_scales, rtol=0.03)
    np.testing.assert_allclose(residual.mean(axis=0), 0.0, atol=0.05)


@pytest.mark.parametrize("noise", ["gaussian", "laplace"])
def test_on_the_true_hessian_the_leaf_rule_is_exact_at_either_scale(noise: str) -> None:
    """A leaf's entry is constant under Gaussian noise and zero under Laplace noise."""
    for scale in ("raw", "standardized"):
        for seed in range(3):
            key = _key("anm", noise, scale, d=6, n=500, seed=seed)
            row = score_oracle.oracle_steps(key)
            assert row["steps"] == 5 and row["wrong_steps"] == 0


def test_the_oracle_takes_additive_noise_data_only() -> None:
    with pytest.raises(ValueError, match="additive-noise"):
        score_oracle.oracle_steps(_key("score2022", "gaussian", "raw", d=5, n=50, seed=0))


# --- the driver ------------------------------------------------------------------------------------


def test_the_driver_runs_the_papers_cells_and_the_oracles_anm_cells() -> None:
    assert set(stein_cpu.REGIMES) | set(stein_cpu.ORACLE_REGIMES) <= set(datasets.REGIMES)
    assert {datasets.REGIMES[r][1] for r in stein_cpu.REGIMES} == {"score2022"}
    assert {datasets.REGIMES[r][1:] for r in stein_cpu.ORACLE_REGIMES} == {
        ("anm", "gaussian", "standardized"),
        ("anm", "laplace", "standardized"),
        ("anm", "gumbel", "standardized"),
    }
    assert stein_cpu.DENSITIES == {"er1": 2.0, "er4": 8.0} and stein_cpu.ROWS == 1000


def test_the_oracle_refuses_the_papers_cells(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--oracle"):
        stein_cpu.main(
            ["--out", str(tmp_path), "--oracle", "--regimes", "score_paper", "--ladder", "5"]
        )


def test_the_oracle_writes_one_line_per_dataset_and_resumes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(datasets.DATA_STORE_ENV, raising=False)
    argv = ["--out", str(tmp_path), "--oracle", "--regimes", "anm_er", "--ladder", "5"]
    assert stein_cpu.main([*argv, "--densities", "er1", "--seeds", "2"]) == 0
    assert stein_cpu.main([*argv, "--densities", "er1", "--seeds", "3"]) == 0
    rows = [json.loads(line) for line in (tmp_path / "oracle.jsonl").read_text().splitlines()]
    assert [r["seed"] for r in rows] == [0, 1, 2]
    assert all(r["steps"] == 4 and r["d"] == 5 for r in rows)


def test_the_driver_scores_every_solution_on_the_same_datasets(monkeypatch, tmp_path: Path) -> None:
    import sys

    monkeypatch.delenv(datasets.DATA_STORE_ENV, raising=False)
    monkeypatch.setenv("ANDREY_BENCH_CAM_PYTHON", sys.executable)
    pairs = [
        ("andrey.cam.numpy", "andrey.cam.order.numpy"),
        ("andrey.score_matching.numpy", "andrey.score_matching.order.numpy"),
        ("andrey.nogam.numpy", "andrey.nogam.order.numpy"),
    ]
    solutions = [name for pair in pairs for name in pair] + ["baseline.empty"]
    argv = ["--out", str(tmp_path), "--regimes", "score_paper_standardized", "--ladder", "5"]
    code = stein_cpu.main([*argv, "--densities", "er1", "--seeds", "1", "--solutions", *solutions])
    assert code == 0
    table = pd.read_parquet(tmp_path / "runs.parquet").set_index("name")
    assert sorted(table.index) == sorted(solutions) and set(table["status"]) == {"ok"}
    assert table.loc["baseline.empty", "shd"] == 5
    # The pruned graph keeps edges of its order's full DAG only, so it orients no more true edges.
    recall = table["arrowhead_recall"]
    for graph, order in pairs:
        assert recall[graph] <= recall[order]
    assert table["dataset_id"].nunique() == 1


# --- the references, in their own environment ------------------------------------------------------


@needs_stein_env
def test_the_references_run_there_and_return_a_dag(monkeypatch, tmp_path: Path) -> None:
    """Each reference fits one small dataset; an order solution returns the full DAG of an order."""
    monkeypatch.setenv("ANDREY_BENCH_STEIN_PYTHON", str(STEIN_PYTHON))
    monkeypatch.setenv(CODE_ENV, str(SCORE_CODE))
    key = _key("score2022", "gaussian", "standardized", d=4, n=200, seed=0)
    datasets.materialize([key], tmp_path)
    field = [RollandSCOREOrder(), DodiscoverNoGAMOrder(), CausalDiscoR2Sort()]
    field += [DodiscoverNoGAM(), CausalDiscoR2SortnRegress()]
    if shutil.which("Rscript") and os.environ.get("R_LIBS_USER"):
        field.append(RollandSCORE())  # its pruning needs Rscript with mgcv
    for adapter in field:
        (measures,) = runner.run_task(
            adapter,
            datasets.data_path(tmp_path, key),
            adapter.params(),
            cap_wall_s=600.0,
            repeats=1,
            warmup=0,
        )
        assert measures["status"] == "ok", (adapter.name, measures.get("stderr"))
        adjacency = np.asarray(measures["adj"])
        assert adjacency.shape == (4, 4) and not np.any(adjacency & adjacency.T)
        if adapter.name.endswith(".order"):
            order = np.argsort(-adjacency.sum(axis=1))
            np.testing.assert_array_equal(adjacency, order_dag(order))
