"""Check method coverage, baselines, rejection controls, and dataset settings."""

from __future__ import annotations

import gzip
import importlib
from pathlib import Path

import numpy as np
import pytest

from andrey import GraphStructure
from andrey.core import Kind
from andrey.data import real
from qa import quality_gate as qg


def test_every_cell_has_a_baseline_and_every_method_a_tolerance() -> None:
    ledger = qg.recorded()
    assert set(ledger["tolerances"]) == set(qg.SUPPORTED)
    assert all(t["median_shd"] >= 1 and t["reason"] for t in ledger["tolerances"].values())
    assert {m: set(ledger["baselines"][m]) for m in qg.SUPPORTED} == {
        m: set(qg.SUPPORTED[m].datasets) for m in qg.SUPPORTED
    }


def empty_answer(draw: qg.Draw, target: Kind) -> GraphStructure:
    return qg.empty_graph(draw.data.shape[1], target)


def shifted_truth(draw: qg.Draw, target: Kind) -> GraphStructure:
    """The true graph with each variable moved one column: an off-by-one in column handling."""
    marks = qg.target_truth(draw.truth, target).to_numpy()
    shift = np.roll(np.arange(len(marks)), 1)
    return GraphStructure.from_numpy(marks[np.ix_(shift, shift)], kind=target)


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
        for d, n in qg.SIZES:
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


def test_the_gates_asia_is_bnlearns():
    """The gate keeps its own arc order, which fixes each seed's weights; the arcs are bnlearn's."""
    bif = Path(__file__).resolve().parents[1] / "unit" / "fixtures" / "asia.bif.gz"
    nodes, arcs = real.read_bif(gzip.decompress(bif.read_bytes()).decode("utf-8"))
    assert nodes == qg.ASIA_NODES and set(arcs) == set(qg.ASIA_ARCS) and len(arcs) == 8
    assert len(set(qg.ASIA_ARCS)) == len(qg.ASIA_ARCS)
