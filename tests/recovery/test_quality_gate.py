"""Check method coverage, baselines, rejection controls, and dataset settings."""

from __future__ import annotations

import gzip
import importlib
from pathlib import Path

import numpy as np
import pytest

from andrey import GraphStructure, data
from andrey.core import LATENT
from andrey.data import real
from andrey.data.sortability import causal_depth, index_correlation
from qa import quality_gate as qg


def test_every_cell_has_a_baseline_and_every_method_a_tolerance() -> None:
    ledger = qg.recorded()
    assert set(ledger["tolerances"]) == set(qg.CHECKS)
    assert all(t["median_shd"] >= 1 and t["reason"] for t in ledger["tolerances"].values())
    assert {m: set(ledger["baselines"][m]) for m in qg.CHECKS} == {
        m: set(qg.CHECKS[m].datasets) for m in qg.CHECKS
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


@pytest.mark.parametrize("method", sorted(qg.CHECKS))
def test_gate_rejects_the_empty_graph(method: str) -> None:
    target = qg.CHECKS[method].target
    for dataset in qg.CHECKS[method].datasets:
        result = qg.evaluate(method, dataset, fit=lambda draw: empty_answer(draw, target))
        assert any("empty graph" in f for f in result.failures), f"{method} on {dataset}"


@pytest.mark.parametrize("method", sorted(qg.CHECKS))
def test_gate_rejects_a_broken_result(method: str) -> None:
    target = qg.CHECKS[method].target
    for dataset in qg.CHECKS[method].datasets:
        result = qg.evaluate(method, dataset, fit=lambda draw: shifted_truth(draw, target))
        assert not result.passed, f"{method} on {dataset} accepted SHD {result.shd}"


def test_tolerance_is_two_sided() -> None:
    def cell(median: int) -> qg.CellResult:
        return qg.CellResult("PC", "x", shd=[median], empty_shd=[20], baseline=5, tolerance=1)

    assert cell(4).passed and cell(6).passed
    assert not cell(3).passed and not cell(7).passed
    assert qg.CellResult("PC", "x", shd=[5], empty_shd=[5], baseline=5, tolerance=1).failures


def test_the_slowest_cells_are_cells_of_the_gate_and_start_first() -> None:
    every = qg.cells()
    assert len(set(qg.SLOWEST_FIRST)) == len(qg.SLOWEST_FIRST)
    assert set(qg.SLOWEST_FIRST) <= set(every)
    order = qg.start_order(every)
    assert sorted(order) == list(range(len(every)))
    started = [every[i] for i in order]
    assert started[: len(qg.SLOWEST_FIRST)] == list(qg.SLOWEST_FIRST)
    rest = [cell for cell in every if cell not in qg.SLOWEST_FIRST]
    assert started[len(qg.SLOWEST_FIRST) :] == rest


def test_several_jobs_return_the_same_results_in_the_same_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    todo = [("PC", "asia_gauss"), ("GES", "asia_gauss"), ("PC", "linear_gauss_er_d20")]
    # The last cell starts first, so the results arrive out of order and are held back.
    monkeypatch.setattr(qg, "SLOWEST_FIRST", (todo[2],))
    assert qg.start_order(todo) == [2, 0, 1]

    def rows(jobs: int) -> list[tuple]:
        return [
            (r.method, r.dataset, r.shd, r.empty_shd, r.baseline, r.tolerance)
            for r in qg.run_cells(todo, jobs)
        ]

    parallel = rows(2)
    assert parallel == rows(1)
    assert [row[:2] for row in parallel] == todo


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


#: Seeds of the column-order checks, in place of the gate's ten; the bound is four standard
#: deviations of the mean correlation over them.
_ORDER_SEEDS = range(200)


def _correlations(draws, by_depth: bool = False) -> tuple[float, float]:
    """The mean rank correlation of column index and causal depth over ``draws``, and its bound.

    Under a uniform column order the correlation on ``k`` columns has mean 0 and variance
    ``1 / (k - 1)``. ``by_depth`` sorts each draw's columns by causal depth first: an order that
    follows the graph, the positive control where the drawn order is random already.
    """
    rhos, ks = [], []
    for draw in draws:
        truth, k = draw.truth, draw.data.shape[1]
        edges = np.array([e[:2] for e in truth.oriented_edges(index=True)], dtype=np.int64)
        edges = edges.reshape(-1, 2)
        if by_depth:
            depth = causal_depth(edges, truth.n_nodes)
            column = np.arange(truth.n_nodes)
            column[:k] = np.argsort(np.argsort(depth[:k], kind="stable"), kind="stable")
            edges = column[edges]
        rho = index_correlation(edges, truth.n_nodes, n_observed=k)
        if np.isfinite(rho):
            rhos.append(rho)
            ks.append(k)
    return float(np.mean(rhos)), 4 * float(np.sqrt(np.sum(1 / (np.array(ks) - 1)))) / len(ks)


def _regime_draws(regime: str) -> list:
    """The gate's draws of ``regime`` at its smaller size, skipping a seed that hides too many."""
    d = qg.sizes(regime)[0][0]
    out = []
    for seed in _ORDER_SEEDS:
        try:
            ds = data.sample_scm(**{**qg.regime_kwargs(regime, d, 5, seed), "n": 5})
        except ValueError as err:  # fewer nodes with two children than nodes to hide
            assert "cannot hide" in str(err)
            continue
        out.append(qg.Draw(ds.data, ds.graph))
    return out


@pytest.mark.parametrize("cell", [*qg.REGIMES, *qg.PAIRS, "asia"])
def test_the_gates_columns_say_nothing_of_the_causal_order(cell, monkeypatch) -> None:
    """The gate's regimes, pairs, and ASIA list their columns in an order drawn from the seed.

    The control: ASIA's unshuffled columns, which follow its file; for the regimes and pairs, the
    same draws sorted by causal depth, since an Erdos-Renyi draw already numbers its nodes at
    random. Sachs is measured data in its file's order, a correlation of -0.08 with the depth of
    its consensus network.
    """
    monkeypatch.setattr(qg, "SEEDS", _ORDER_SEEDS)
    monkeypatch.setattr(qg, "ASIA_N", 5)
    if cell == "asia":
        draws = qg.asia_draws("gaussian")
        scm = data.SCM(
            graph=data.graphs.FixedDAG(nodes=qg.ASIA_NODES, arcs=qg.ASIA_ARCS),
            functional=data.functional.linear(),
            noise=data.noise.gaussian(),
        )
        plain = [scm.sample(n=5, seed=s, shuffle=False) for s in _ORDER_SEEDS]
        control, _ = _correlations(qg.Draw(ds.data, ds.graph) for ds in plain)
    else:
        draws = qg.pair_draws(cell, 5) if cell in qg.PAIRS else _regime_draws(cell)
        control, _ = _correlations(draws, by_depth=True)
    mean, bound = _correlations(draws)
    assert abs(mean) < bound
    assert abs(control) > bound
