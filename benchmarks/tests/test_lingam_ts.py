"""The LiNGAM time-series lane: the paper simulations, the coefficient scores, and both adapters.

Facts about the simulations, the scores, and Andrey's adapters are checked in bench-env. Facts about
the lingam boundary (a fit runs in ``.venv-lingam-ts`` and comes back as a stack this process scores)
are checked by running one, and skipped when that environment has not been built.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import andrey
from andrey_bench import datasets, lingam_models, rundir, runner, scoring
from andrey_bench.adapters.andrey_lingam_ts import AndreyLongitudinalLiNGAM, AndreyVarmaLiNGAM
from andrey_bench.adapters.lingam_ts import (
    PYTHON_ENV_VAR,
    lingam_longitudinal_adapter,
    lingam_varma_adapter,
)
from andrey_bench.coefficients import CoefficientStack
from andrey_bench.integration import _truth

BENCH = Path(__file__).resolve().parents[1]
LINGAM_PYTHON = Path(runner.BENCH_ROOT) / ".venv-lingam-ts" / "bin" / "python"
needs_lingam_env = pytest.mark.skipif(
    not LINGAM_PYTHON.exists(), reason="lingam-ts env not built - run benchmarks/start_lingam_ts.sh"
)


@pytest.fixture
def lingam_env(monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(PYTHON_ENV_VAR, str(LINGAM_PYTHON))
    return LINGAM_PYTHON


def _key(regime: str, d: int, n: int, seed: int = 0, times: int = 0) -> datasets.DatasetKey:
    topology, functional, noise, standardize = datasets.REGIMES[regime]
    return datasets.DatasetKey(
        topology=topology,
        functional=functional,
        noise=noise,
        standardize=standardize,
        density=datasets.TEMPORAL_REGIMES[regime],
        d=d,
        n=n,
        seed=seed,
        times=times,
    )


# --- the simulations -------------------------------------------------------------------------------


@pytest.mark.parametrize("d", [5, 10, 40])
def test_the_varma_model_keeps_its_links_per_variable_at_every_size(d: int) -> None:
    data, stack = lingam_models.sample_varma("varma2010", d, n=300, seed=0)
    b0, b1, m1 = stack
    assert data.shape == (300, d)
    assert not np.triu(b0).any()  # strictly lower triangular: the causal order is 0..N-1
    assert np.count_nonzero(b0) == round(0.8 * d)
    assert np.count_nonzero(b1) == np.count_nonzero(m1) == 2 * d
    for block in stack:
        magnitudes = np.abs(block[block != 0])
        assert magnitudes.min() >= 0.05 and magnitudes.max() <= 0.5
    inverse = np.linalg.inv(np.eye(d) - b0)
    assert np.max(np.abs(np.linalg.eigvals(inverse @ b1))) < 1
    assert np.max(np.abs(np.linalg.eigvals(m1))) < 1


def test_the_varma_noise_is_super_gaussian_and_the_draw_repeats() -> None:
    data, stack = lingam_models.sample_varma("varma2010", 5, n=5000, seed=1)
    again, stack_again = lingam_models.sample_varma("varma2010", 5, n=5000, seed=1)
    np.testing.assert_array_equal(data, again)
    np.testing.assert_array_equal(stack, stack_again)
    b0, b1, m1 = stack
    # Recover the noise from the recursion: e_t = (I - B0) x_t - B1 x_{t-1} - M1 e_{t-1}.
    noise = np.zeros_like(data)
    for t in range(1, len(data)):
        noise[t] = (np.eye(5) - b0) @ data[t] - b1 @ data[t - 1] - m1 @ noise[t - 1]
    centered = noise[100:] - noise[100:].mean(axis=0)
    kurtosis = (centered**4).mean(axis=0) / (centered**2).mean(axis=0) ** 2 - 3
    assert (kurtosis > 0.5).all()


@pytest.mark.parametrize("d", [3, 10])
def test_the_panel_model_draws_a_dense_dag_and_a_dense_lag_block_per_time_point(d: int) -> None:
    times, m = 6, 400
    data, truth = lingam_models.sample_panel("panel2013", d, times, m, seed=0)
    assert data.shape == (times, m, d) and truth.shape == (times, 2, d, d)
    assert not truth[0, 1].any()  # nothing before the first time point
    for t in range(times):
        b = truth[t, 0]
        # A fully dense DAG under one variable order: d(d-1)/2 links and a nilpotent matrix.
        assert np.count_nonzero(b) == d * (d - 1) // 2
        assert not np.linalg.matrix_power(np.abs(b) > 0, d).any()
        assert (np.abs(b[b != 0]) >= 0.5).all() and (np.abs(b) <= 1.5).all()
        if t:
            assert np.count_nonzero(truth[t, 1]) == d * d and (np.abs(truth[t, 1]) <= 0.5).all()
    # Every time point draws its own matrices.
    assert not np.array_equal(truth[1, 0], truth[2, 0])
    assert np.isfinite(data).all()


@pytest.mark.parametrize("density", lingam_models.DENSITIES)
def test_each_noise_density_has_zero_mean_and_unit_variance(density: str) -> None:
    draws = lingam_models._standard_noise(np.random.default_rng(0), density, (400_000,))
    assert abs(draws.mean()) < 0.02 and abs(draws.var() - 1) < 0.05


# --- the dataset store -----------------------------------------------------------------------------


def test_a_panel_key_carries_its_time_points_and_the_other_keys_keep_their_identity() -> None:
    lagged = datasets.DatasetKey(
        topology="er", functional="var", noise="gaussian", standardize="raw", density=0.0, d=5,
        n=300, seed=0,
    )  # fmt: skip
    assert lagged.digest == "d74ca17b53ba" and "-T" not in lagged.label
    panel = _key("longitudinal_lingam_paper", d=3, n=500, times=20)
    assert panel.label.endswith("-T20") and datasets.generation_kwargs(panel)["times"] == 20
    assert panel.digest != dataclasses.replace(panel, times=21).digest
    with pytest.raises(ValueError, match="times"):
        datasets.generation_kwargs(dataclasses.replace(panel, times=0))
    with pytest.raises(ValueError, match="times"):
        datasets.generation_kwargs(_key("varma_lingam_paper", d=3, n=300, times=5))


def test_each_lingam_key_stores_a_coefficient_truth(tmp_path) -> None:
    keys = [_key("varma_lingam_paper", 4, 300), _key("longitudinal_lingam_paper", 3, 200, times=4)]
    registry = datasets.materialize(keys, tmp_path)
    kinds = [registry["keys"][k.label]["graph_kind"] for k in keys]
    assert kinds == ["temporal_coefficients", "panel_coefficients"]
    (series, varma), (panel, stack) = (datasets.load(tmp_path, k) for k in keys)
    assert series.shape == (300, 4) and panel.shape == (4, 200, 3)
    assert _truth(varma, kinds[0]).kind == "varma" and _truth(stack, kinds[1]).kind == "panel"


# --- scores ----------------------------------------------------------------------------------------


def _varma(b0=(), b1=(), m1=(), n: int = 3) -> CoefficientStack:
    stack = np.zeros((3, n, n))
    for block, links in enumerate((b0, b1, m1)):
        for effect, cause, value in links:
            stack[block, effect, cause] = value
    return CoefficientStack(stack, kind="varma")


def test_the_truth_scores_perfectly_and_a_link_counts_by_lag_cause_and_effect() -> None:
    truth = _varma(b0=[(1, 0, 0.5)], b1=[(2, 1, 0.3), (0, 0, 0.4)], m1=[(1, 1, 0.2)])
    perfect = scoring.score_result(truth, truth)
    assert perfect["f1"] == 1.0 and perfect["instantaneous_rmse"] == perfect["ma_rmse"] == 0.0
    # The self-link at lag 1 is not a cross-link; a reversed lag-0 link is a miss and an extra.
    reversed_ = _varma(b0=[(0, 1, 0.5)], b1=[(2, 1, 0.3)])
    scores = scoring.score_result(reversed_, truth)
    assert scores["instantaneous_precision"] == scores["instantaneous_recall"] == 0.0
    assert scores["lagged_f1"] == 1.0 and scores["precision"] == 0.5
    assert scores["lagged_rmse"] == pytest.approx(np.sqrt(0.4**2 / 9))


def test_a_panel_is_scored_from_its_second_time_point() -> None:
    truth = np.zeros((3, 2, 2, 2))
    truth[:, 0, 1, 0] = 1.0
    estimate = truth.copy()
    estimate[0] = np.nan  # the first time point is not estimated
    scores = scoring.score_result(
        CoefficientStack(estimate, "panel"), CoefficientStack(truth, "panel")
    )
    assert scores["f1"] == 1.0 and scores["instantaneous_rmse"] == 0.0


def test_the_structure_hash_reads_links_not_values_and_survives_the_run(tmp_path) -> None:
    a = _varma(b0=[(1, 0, 0.5)])
    b = _varma(b0=[(1, 0, 0.7)])
    c = _varma(b0=[(0, 1, 0.5)])
    assert scoring.structure_hash(a) == scoring.structure_hash(b) != scoring.structure_hash(c)
    rundir.write_structure(tmp_path, "abc:andrey.varma_lingam.numpy:0:def", a)
    restored = rundir.read_structure(tmp_path, "abc:andrey.varma_lingam.numpy:0:def")
    np.testing.assert_array_equal(restored.stack, a.stack)


# --- the adapters ----------------------------------------------------------------------------------


def test_the_andrey_varma_stack_is_the_structure_andrey_returns() -> None:
    data, _ = lingam_models.sample_varma("varma2010", 3, n=300, seed=0)
    adapter = AndreyVarmaLiNGAM()
    stack = adapter.fit(data, adapter.params())
    out = andrey.varma_lingam(data, order=(1, 1), criterion=None, prune=True)
    assert stack.shape == (3, 3, 3)
    np.testing.assert_array_equal(
        stack[:2], np.asarray(out.structure.lag_weights).transpose(0, 2, 1)
    )
    np.testing.assert_array_equal(stack[2], np.asarray(out.structure.lag_weights_ma)[0].T)


def test_the_andrey_panel_stack_covers_every_time_point() -> None:
    panel, truth = lingam_models.sample_panel("panel2013", 3, 4, 300, seed=0)
    adapter = AndreyLongitudinalLiNGAM()
    stack = adapter.fit(panel, adapter.params())
    assert stack.shape == truth.shape and np.isnan(stack[0]).all() and np.isfinite(stack[1:]).all()
    scores = scoring.score_result(adapter.to_structure(stack), CoefficientStack(truth, "panel"))
    assert scores["instantaneous_f1"] > 0.8


def test_the_lingam_module_imports_nothing_from_andrey() -> None:
    import subprocess

    code = (
        "import sys, andrey_bench.adapters.lingam_ts, andrey_bench.coefficients; "
        "assert not [m for m in sys.modules if m.split('.')[0] in ('andrey', 'lingam')]"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=runner.BENCH_ROOT,
        env={"PYTHONPATH": str(runner.BENCH_ROOT)},
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


@needs_lingam_env
@pytest.mark.parametrize("method", ["varma", "longitudinal"])
def test_a_lingam_fit_runs_there_and_matches_andreys_links(
    lingam_env: Path, tmp_path: Path, method: str
) -> None:
    if method == "varma":
        key, ours, theirs = (
            _key("varma_lingam_paper", 3, 500),
            AndreyVarmaLiNGAM(),
            lingam_varma_adapter(),
        )
    else:
        key = _key("longitudinal_lingam_paper", 3, 300, times=4)
        ours, theirs = AndreyLongitudinalLiNGAM(), lingam_longitudinal_adapter()
    assert theirs.package_version == "1.13.0"
    datasets.materialize([key], tmp_path)
    (measures,) = runner.run_task(
        theirs,
        datasets.data_path(tmp_path, key),
        theirs.params(),
        cap_wall_s=600.0,
        repeats=1,
        warmup=0,
    )
    assert measures["status"] == "ok", measures.get("stderr")
    data, _ = datasets.load(tmp_path, key)
    mine = ours.fit(data, ours.params())
    assert scoring.structure_hash(ours.to_structure(mine)) == scoring.structure_hash(
        theirs.to_structure(measures["adj"])
    )
    if method == "longitudinal":  # the same estimator, step for step
        np.testing.assert_allclose(mine, measures["adj"], rtol=0, atol=1e-9)


# --- the driver ------------------------------------------------------------------------------------


def _driver():
    spec = importlib.util.spec_from_file_location("lingam_ts_driver", BENCH / "lingam_ts.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_driver_keys_a_series_by_length_and_a_panel_by_units_and_time_points() -> None:
    driver = _driver()
    series = driver.ladder_keys("varma", [3, 5], [300], None, seeds=2)
    assert [(k.d, k.n, k.times, k.seed) for k in series] == [
        (d, 300, 0, s) for d in (3, 5) for s in (0, 1)
    ]
    panel = driver.ladder_keys("longitudinal", [3], [20], [500, 1000], seeds=1)
    assert [(k.d, k.n, k.times) for k in panel] == [(3, 500, 20), (3, 1000, 20)]
