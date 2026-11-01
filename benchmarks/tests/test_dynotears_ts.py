"""The DYNOTEARS lane: the paper's structural VARs, the edge scores, both adapters, and the driver.

Facts about the simulations, the scores, and Andrey's adapter are checked in bench-env. Facts about
the CausalNex boundary (a fit runs in ``.venv-causalnex`` and comes back as marks this process
scores) are checked by running one, and skipped when that environment has not been built.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

import andrey
from andrey_bench import datasets, runner, scoring, svar_models
from andrey_bench.adapters.andrey_dynotears import AndreyDYNOTEARS, window_from_temporal
from andrey_bench.adapters.causalnex_dynotears import PYTHON_ENV_VAR, CausalnexDYNOTEARS
from andrey_bench.window import ARROW, TAIL, WindowGraph, marks_from_edges

BENCH = Path(__file__).resolve().parents[1]
CAUSALNEX_PYTHON = Path(runner.BENCH_ROOT) / ".venv-causalnex" / "bin" / "python"
needs_causalnex_env = pytest.mark.skipif(
    not CAUSALNEX_PYTHON.exists(),
    reason="causalnex env not built - run benchmarks/start_causalnex.sh",
)


def _driver():
    spec = importlib.util.spec_from_file_location("dynotears_ts", BENCH / "dynotears_ts.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["dynotears_ts"] = module
    spec.loader.exec_module(module)
    return module


def _key(regime: str, d: int, n: int, seed: int = 0) -> datasets.DatasetKey:
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
    )


# --- the simulations -------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(svar_models.MODELS))
def test_a_model_draws_the_papers_weights_on_a_dag_and_a_stationary_lag(name: str) -> None:
    for seed in range(5):
        w, a, _ = svar_models.draw(name, 20, seed)
        same, lagged = np.abs(w[w != 0]), np.abs(a[a != 0])
        assert same.min() >= 0.5 and same.max() <= 2.0
        assert lagged.min() >= 0.3 and lagged.max() <= 0.5
        assert not np.diag(w).any()
        marks_from_edges(w, a[None])  # raises on a same-step cycle
        if svar_models.MODELS[name].link == "linear":
            assert svar_models.spectral_radius(w, a) < 1.0


def test_the_erdos_renyi_model_keeps_its_degrees_at_every_size() -> None:
    """Mean degree 2 at the same step and mean in-degree 2 at lag 1, a little lower for the draws
    the stationarity check keeps."""
    for d in (10, 50):
        draws = [svar_models.draw("dynotears2020", d, seed)[:2] for seed in range(30)]
        same = np.mean([2 * np.count_nonzero(w) / d for w, _ in draws])
        lagged = np.mean([np.count_nonzero(a) / d for _, a in draws])
        assert 1.6 < same < 2.2 and 1.7 < lagged < 2.2


def test_a_series_fits_n_steps_and_its_truth_is_the_drawn_graph() -> None:
    data, marks = svar_models.sample(
        "dynotears2020", 6, 50, seed=1, noise="gaussian", standardize="raw"
    )
    w, a, _ = svar_models.draw("dynotears2020", 6, 1)
    assert data.shape == (51, 6)
    np.testing.assert_array_equal((marks[0] == TAIL) & (marks[0].T == ARROW), w != 0)
    np.testing.assert_array_equal(marks[1] == TAIL, a != 0)


def test_noise_of_mean_one_adds_the_process_mean_to_the_centered_series() -> None:
    """Both series take the same exponential draws, one of them less 1, so they differ by the
    process mean ``1 (I - W - A)^-1``."""
    w, a, _ = svar_models.draw("dynotears2020", 5, 0)
    centered, _ = svar_models.sample("dynotears2020", 5, 200, 0, "exponential", "raw")
    raw, _ = svar_models.sample("dynotears2020", 5, 200, 0, "exponential_uncentered", "raw")
    mean = np.ones(5) @ np.linalg.inv(np.eye(5) - w - a)
    np.testing.assert_allclose(raw - centered, np.broadcast_to(mean, raw.shape), rtol=1e-6)


def test_the_standardized_series_is_the_raw_one_z_scored() -> None:
    raw, truth = svar_models.sample("dynotears2020", 5, 300, 2, "gaussian", "raw")
    scaled, same = svar_models.sample("dynotears2020", 5, 300, 2, "gaussian", "standardized")
    np.testing.assert_allclose(scaled, (raw - raw.mean(axis=0)) / raw.std(axis=0))
    np.testing.assert_array_equal(truth, same)


def test_a_model_without_a_stationary_draw_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    strong = svar_models.SVARModel(
        "er", 2.0, "er", 2.0, (0.5, 2.0), (5.0, 6.0), block_ratio=0.3, link="linear"
    )
    monkeypatch.setitem(svar_models.MODELS, "dynotears2020", strong)
    monkeypatch.setattr(svar_models, "MAX_DRAWS", 3)
    with pytest.raises(RuntimeError, match="stationary"):
        svar_models.draw("dynotears2020", 10, 0)


#: SHA-256 of the weights of every model over ``d`` in (5, 20) and seeds 0 to 2. A change to any
#: drawn value changes it, and with it the datasets a campaign has stored.
SVAR_DRAWS = "f778e3658e4203879c8da282dd6805c8c9d4c37e65a6af5149db22a8839b5045"


def test_a_model_keeps_its_draws() -> None:
    digest = hashlib.sha256()
    for name in sorted(svar_models.MODELS):
        for d in (5, 20):
            for seed in range(3):
                w, a, _ = svar_models.draw(name, d, seed)
                digest.update(np.ascontiguousarray(np.r_[w, a]).tobytes())
    assert digest.hexdigest() == SVAR_DRAWS


@pytest.mark.parametrize(
    "regime", [r for r, spec in datasets.REGIMES.items() if spec[1] in svar_models.MODELS]
)
def test_a_dynotears_regime_stores_a_window_truth(regime: str, tmp_path: Path) -> None:
    key = _key(regime, 5, 40)
    registry = datasets.materialize([key], tmp_path)
    entry = registry["keys"][key.label]
    assert entry["graph_kind"] == datasets.WINDOW_KIND
    assert entry["generation_call"]["generator"] == "andrey_bench.svar_models.sample"
    data, marks = datasets.load(tmp_path, key)
    assert data.shape == (41, 5) and marks.shape == (2, 5, 5)


def test_noise_and_scale_make_their_own_datasets() -> None:
    keys = [_key(r, 10, 500) for r in ("dynotears_paper", "dynotears_paper_exponential")]
    keys += [
        _key(r, 10, 500) for r in ("dynotears_paper_standardized", "dynotears_paper_uncentered")
    ]
    assert len({k.digest for k in keys}) == 4


# --- the edge scores -------------------------------------------------------------------------------


def _window(same_step: list[tuple[int, int]], lagged: list[tuple[int, int]], n: int = 4):
    w, a = np.zeros((n, n), bool), np.zeros((1, n, n), bool)
    for i, j in same_step:
        w[i, j] = True
    for i, j in lagged:
        a[0, i, j] = True
    return WindowGraph(marks_from_edges(w, a), lag0="dag")


TRUTH = _window([(0, 1), (1, 2)], [(0, 0), (2, 3)])


def test_the_truth_scores_perfectly_and_the_empty_window_costs_one_per_edge() -> None:
    perfect = scoring.score_result(TRUTH, TRUTH)
    assert (perfect["shd"], perfect["f1"], perfect["lag0_f1"], perfect["lagged_f1"]) == (0, 1, 1, 1)
    empty = scoring.score_result(_window([], []), TRUTH)
    assert (empty["lag0_shd"], empty["lagged_shd"], empty["shd"]) == (2.0, 2.0, 4.0)
    assert empty["f1"] == 0.0
    assert empty["endpoint_shd"] == 2 * empty["shd"]


def test_a_reversed_edge_costs_one_and_is_a_false_and_a_missed_edge() -> None:
    reversed_ = scoring.score_result(_window([(1, 0), (1, 2)], [(0, 0), (2, 3)]), TRUTH)
    assert (reversed_["lag0_shd"], reversed_["lagged_shd"]) == (1.0, 0.0)
    assert reversed_["lag0_f1"] == 0.5 and reversed_["lagged_f1"] == 1.0
    assert reversed_["f1"] == 0.75


def test_an_extra_lagged_link_costs_one() -> None:
    extra = scoring.score_result(_window([(0, 1), (1, 2)], [(0, 0), (2, 3), (3, 3)]), TRUTH)
    assert (extra["lag0_shd"], extra["lagged_shd"]) == (0.0, 1.0)
    assert extra["lagged_f1"] == pytest.approx(0.8)


def test_only_a_dag_estimate_has_edge_scores() -> None:
    cpdag = WindowGraph(TRUTH.marks, lag0="cpdag")
    assert "shd" not in scoring.score_result(cpdag, TRUTH)


def test_a_same_step_cycle_has_no_window_marks() -> None:
    with pytest.raises(RuntimeError, match="cycle"):
        marks_from_edges(np.array([[0, 1], [1, 0]]), np.zeros((1, 2, 2)))
    with pytest.raises(RuntimeError, match="cycle"):
        marks_from_edges(np.eye(2), np.zeros((1, 2, 2)))
    assert marks_from_edges(np.zeros((2, 2)), np.eye(2)[None])[1].trace() == 2 * TAIL


# --- the adapters and the driver -------------------------------------------------------------------


def test_the_settings_for_500_steps_are_andreys_defaults() -> None:
    defaults = inspect.signature(andrey.dynotears).parameters
    for name, value in _driver().SETTINGS["n500"].items():
        assert defaults[name].default == value


@pytest.mark.parametrize("settings", ["n500", "n50"])
def test_both_packages_run_the_same_settings(settings: str) -> None:
    driver = _driver()
    field = driver.solutions(settings, centered=True)
    names = [a.name for a in field]
    assert len(set(names)) == len(names) and all(name.endswith(settings) for name in names[:3])
    ours, theirs, centered = (dict(a.params()) for a in field[:3])
    assert ours == {k: v for k, v in theirs.items() if k != "centered"}
    assert (theirs["centered"], centered["centered"]) == (False, True)
    assert {k: ours[k] for k in driver.SETTINGS[settings]} == driver.SETTINGS[settings]


def test_a_regime_is_in_claim_only_at_the_papers_settings_for_its_length() -> None:
    claim = _driver().claim
    assert (
        claim("dynotears_paper", 500, "n500") == claim("dynotears_paper_ba_sbm", 50, "n50") == "in"
    )
    assert claim("dynotears_paper", 50, "n500") == "out"
    for regime in (
        "dynotears_paper_standardized",
        "dynotears_paper_uncentered",
        "dynotears_sin_tanh",
    ):
        assert claim(regime, 500, "n500") == "out"


def test_andreys_window_is_its_structure_self_lags_included() -> None:
    data, _ = svar_models.sample("dynotears2020", 5, 500, 3, "gaussian", "raw")
    out = andrey.dynotears(data)
    marks = window_from_temporal(out.structure)
    weights = np.asarray(out.structure.lag_weights)
    np.testing.assert_array_equal((marks[0] == TAIL) & (marks[0].T == ARROW), weights[0] != 0)
    np.testing.assert_array_equal(marks[1] == TAIL, weights[1] != 0)
    assert np.diag(weights[1]).any()
    adapter = AndreyDYNOTEARS(name="andrey.dynotears.n500", **_driver().SETTINGS["n500"])
    np.testing.assert_array_equal(adapter.fit(data, adapter.params()), marks)


@needs_causalnex_env
@pytest.mark.parametrize("centered", [False, True])
def test_a_causalnex_fit_runs_there_and_matches_andreys_graph(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, centered: bool
) -> None:
    monkeypatch.setenv(PYTHON_ENV_VAR, str(CAUSALNEX_PYTHON))
    settings = _driver().SETTINGS["n500"]
    theirs = CausalnexDYNOTEARS(name="causalnex.test", centered=centered, **settings)
    ours = AndreyDYNOTEARS(name="andrey.test", **settings)
    key = _key("dynotears_paper", 5, 500)
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
    data, truth = datasets.load(tmp_path, key)
    mine = ours.to_structure(ours.fit(data, ours.params()))
    estimate = theirs.to_structure(measures["adj"])
    assert scoring.structure_hash(estimate) == scoring.structure_hash(mine)
    assert scoring.score_result(estimate, WindowGraph(truth, lag0="dag"))["shd"] == 0.0
