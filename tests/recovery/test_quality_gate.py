"""Check method coverage, baselines, rejection controls, and dataset settings."""

from __future__ import annotations

import gzip
import importlib
from pathlib import Path

import numpy as np
import pytest

from andrey import GraphStructure
from andrey.core import LATENT
from andrey.data import real
from qa import quality_gate as qg


def test_every_cell_has_a_baseline_and_every_method_a_tolerance() -> None:
    ledger = qg.recorded()
    assert set(ledger["tolerances"]) == set(qg.SUPPORTED)
    assert all(t["median_shd"] >= 1 and t["reason"] for t in ledger["tolerances"].values())
    assert {m: set(ledger["baselines"][m]) for m in qg.SUPPORTED} == {
        m: set(qg.SUPPORTED[m].datasets) for m in qg.SUPPORTED
    }


def empty_answer(draw: qg.Draw, target: qg.Target) -> object:
    return qg.empty_estimate(draw, target)


def shifted_truth(draw: qg.Draw, target: qg.Target) -> object:
    """The true graph with each variable moved one column: an off-by-one in column handling. A
    measurement model moves its indicators and keeps its latents last."""
    if target in qg.SERIES_TARGETS:
        return shifted_series_truth(draw.truth, target)
    truth = qg.target_truth(draw.truth, target)
    marks = truth.to_numpy()
    shift = np.arange(len(marks))
    types = None if target != "clusters" else np.asarray(truth.node_types)
    observed = shift if types is None else np.flatnonzero(types != LATENT)
    shift[observed] = np.roll(observed, 1)
    return GraphStructure.from_numpy(marks[np.ix_(shift, shift)], kind=truth.kind, node_types=types)


def shifted_series_truth(truth: object, target: qg.Target) -> object:
    """A time-series truth with each variable moved one place, over the observed variables.

    A coefficient stack swaps its same-step and lag-1 blocks instead, an off-by-one in the lag: its
    regimes' graphs can be dense, and moving every variable of a complete graph changes nothing.
    """
    if target == "lagged":
        links = qg.links_from_temporal(truth)
        return qg.temporal_from_links(np.roll(links, 1, axis=(1, 2)))
    if target == "window":
        n = truth.n_nodes - truth.latents
        return qg.WindowGraph(np.roll(truth.marks[:, :n, :n], 1, axis=(1, 2)), lag0="cpdag")
    if target == "cross_links":
        blocks = truth.stack[[1, 0, 2]] if truth.kind == "varma" else truth.stack[:, ::-1]
        return qg.CoefficientStack(blocks, kind=truth.kind)
    return np.roll(truth, 1, axis=(1, 2))


@pytest.mark.parametrize("method", sorted(qg.SUPPORTED))
def test_gate_rejects_the_empty_graph(method: str) -> None:
    target = qg.SUPPORTED[method].target
    for dataset in qg.SUPPORTED[method].datasets:
        result = qg.evaluate(method, dataset, fit=lambda draw: empty_answer(draw, target))
        assert any("empty graph" in f for f in result.failures), f"{method} on {dataset}"


@pytest.mark.parametrize("method", sorted(qg.SUPPORTED))
def test_gate_rejects_a_broken_result(method: str) -> None:
    target = qg.SUPPORTED[method].target
    for dataset in qg.SUPPORTED[method].datasets:
        result = qg.evaluate(method, dataset, fit=lambda draw: shifted_truth(draw, target))
        assert not result.passed, f"{method} on {dataset} accepted SHD {result.shd}"


def test_tolerance_is_two_sided() -> None:
    def cell(median: int) -> qg.CellResult:
        return qg.CellResult("PC", "x", shd=[median], empty_shd=[20], baseline=5, tolerance=1)

    assert cell(4).passed and cell(6).passed
    assert not cell(3).passed and not cell(7).passed
    assert qg.CellResult("PC", "x", shd=[5], empty_shd=[5], baseline=5, tolerance=1).failures


def test_regimes_match_the_benchmark_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate draws each regime with the arguments the published benchmarks used."""
    pytest.importorskip("fcntl")  # the harness's dataset store locks files with fcntl
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    bench = importlib.import_module("andrey_bench.datasets")
    for regime in qg.REGIMES:
        topology, functional, noise, standardize = bench.REGIMES[regime]
        for d, n in qg.sizes(regime):
            key = bench.DatasetKey(
                topology=topology,
                functional=functional,
                noise=noise,
                density=bench.CANONICAL_DENSITY,
                d=d,
                n=n,
                seed=3,
                standardize=standardize,
                latents=bench.latents_for(regime, d),
            )
            assert qg.regime_kwargs(regime, d, n, 3) == bench.generation_kwargs(key)


def test_model_regimes_match_the_benchmark_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate draws CD-NOD's, MultiGroupDirectLiNGAM's, and GIN's regimes as the harness does."""
    pytest.importorskip("fcntl")  # the harness's dataset store locks files with fcntl
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    bench = importlib.import_module("andrey_bench.datasets")
    for regime, (model, noise) in qg.MODEL_REGIMES.items():
        topology, functional, bench_noise, standardize = bench.REGIMES[regime]
        assert (functional, bench_noise) == (model, noise)
        d, n = qg.MODEL_SIZES[regime][0]
        spec = qg.regime_models.MODELS[model]
        key = bench.DatasetKey(
            topology=topology,
            functional=functional,
            noise=noise,
            density=getattr(spec, "indicators", bench.CANONICAL_DENSITY),
            d=d,
            n=n,
            seed=3,
            standardize=standardize,
            latents=bench.latents_for(regime, d),
        )
        data, marks, kind, _ = bench.generate_one(key)
        draw = qg.model_draws(regime, d, n)[3]
        assert np.array_equal(draw.data, data) and np.array_equal(draw.truth.to_numpy(), marks)
        assert draw.truth.kind == kind


def test_pairs_match_the_benchmark_drivers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate draws each cause-effect pair with the arguments its benchmark driver used."""
    pytest.importorskip("fcntl")  # the harness's dataset store locks files with fcntl
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    bench = importlib.import_module("andrey_bench.datasets")
    drivers = {"anm_pair": "anm_pairs", "pnl_pair": "pnl_pairs"}
    assert set(drivers) == set(qg.PAIRS)
    for regime, driver in drivers.items():
        module = importlib.import_module(driver)
        assert module.REGIME == regime
        for key in module.pair_keys([qg.PAIR_N], 3):
            assert qg.pair_kwargs(regime, qg.PAIR_N, key.seed) == bench.generation_kwargs(key)


def test_time_series_match_the_benchmark_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate's time-series regimes are the harness's, and its Granger draws its driver's."""
    pytest.importorskip("fcntl")  # the harness's dataset store locks files with fcntl
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    bench = importlib.import_module("andrey_bench.datasets")
    for regime, model in qg.SERIES_MODELS.items():
        assert bench.REGIMES[regime][1] == model
    assert bench.REGIMES["longitudinal_lingam_paper"][1] == qg.PANEL_MODEL
    driver = importlib.import_module("granger_var")
    assert (driver.LAGS, driver.LAG_DEGREE) == (qg.GRANGER_LAGS, qg.GRANGER_LAG_DEGREE)
    d, n = qg.GRANGER_SIZES[0]
    draw = qg.granger_draws(d, n)[3]
    data, truth = driver.sample(d, n, 3)
    assert np.array_equal(draw.data, data) and np.array_equal(draw.truth, truth)


def test_the_projection_holds_the_edges_and_pairs_the_benchmarks_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate's latent projection has the benchmark scorer's edges and confounded pairs."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    scoring = importlib.import_module("andrey_bench.scoring")
    found = 0
    for d, n in qg.sizes("latent_lingam_er"):
        for seed in range(5):
            truth = qg.data.sample_scm(**qg.regime_kwargs("latent_lingam_er", d, n, seed)).graph
            edges = qg.projection(truth).oriented_edges()
            pairs = scoring.true_confounded_pairs(truth, "hidden_cause")
            projected = {
                e for e in scoring.projected_directed(truth) if tuple(sorted(e)) not in pairs
            }
            assert {(i, j) for i, j, kind in edges if kind == "directed"} == projected
            assert {(i, j) for i, j, kind in edges if kind == "bidirected"} == pairs
            assert {kind for _, _, kind in edges} <= {"directed", "bidirected"}
            found += len(pairs)
    assert found, "no draw has a confounded pair, so the check above tests nothing"


def test_the_gates_asia_is_bnlearns():
    """The gate keeps its own arc order, which fixes each seed's weights; the arcs are bnlearn's."""
    bif = Path(__file__).resolve().parents[1] / "unit" / "fixtures" / "asia.bif.gz"
    nodes, arcs = real.read_bif(gzip.decompress(bif.read_bytes()).decode("utf-8"))
    assert nodes == qg.ASIA_NODES and set(arcs) == set(qg.ASIA_ARCS) and len(arcs) == 8
    assert len(set(qg.ASIA_ARCS)) == len(qg.ASIA_ARCS)
