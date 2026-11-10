"""Measurement columns in ``andrey.data``: ``measurement.noisy_children`` and ``measurements=``."""

from __future__ import annotations

import numpy as np
import pytest

import andrey.data as data
from andrey.core.structure import LATENT, OBSERVED
from andrey.data import measurement, missingness
from andrey.data.interventions import hard

SCALES = ("standardize", "unitless", "rescale", "raw")


def _edges(graph) -> set[tuple[int, int]]:
    return {(parent, child) for parent, child, _ in graph.oriented_edges(index=True)}


def _sample(spec, *, n=4000, seed=0, **kwargs):
    return data.sample_scm(d=6, n=n, seed=seed, density=2, measurements=spec, **kwargs)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_measurements_are_appended_and_leave_the_sample_as_it_is(scale, dtype):
    plain = data.sample_scm(d=6, n=500, seed=0, density=2, scale=scale, dtype=dtype)
    ds = _sample(measurement.noisy_children([1, 4]), n=500, scale=scale, dtype=dtype)
    assert ds.data.shape == (500, 8)
    assert ds.data.dtype == plain.data.dtype
    assert ds.data.flags.c_contiguous
    assert np.array_equal(ds.data[:, :6], plain.data)
    assert np.array_equal(ds.params.edges, plain.params.edges)
    assert np.array_equal(ds.params.weights, plain.params.weights)
    assert ds.report == plain.report
    assert ds.provenance["column_order"] == plain.provenance["column_order"]


def test_the_truth_records_each_measurement_as_a_child_of_its_column():
    plain = data.sample_scm(d=6, n=500, seed=0, density=2)
    ds = _sample(measurement.noisy_children([4, 1]), n=500)
    assert ds.measurements.parents == {6: 1, 7: 4}
    assert ds.measurements.noise == "gaussian"
    assert ds.measurements.noise_scale == 0.5
    assert ds.graph.n_nodes == 8
    ds.graph.validate()
    assert _edges(ds.graph) == _edges(plain.graph) | {(1, 6), (4, 7)}
    assert ds.observed_graph == ds.graph
    np.testing.assert_allclose(ds.params.noise_scales[:6], plain.params.noise_scales)
    np.testing.assert_allclose(ds.params.noise_scales[6:], 0.5 * plain.data[:, [1, 4]].std(axis=0))
    assert ds.provenance["config"]["measurements"] == {
        "columns": [1, 4],
        "noise_scale": 0.5,
        "noise": "gaussian",
    }
    assert plain.measurements is None


@pytest.mark.parametrize("noise_scale", [0.5, 1.0])
def test_a_raw_measurement_is_its_column_plus_independent_noise(noise_scale):
    ds = _sample(measurement.noisy_children([0, 3], noise_scale=noise_scale), n=40_000, scale="raw")
    for column, parent in ds.measurements.parents.items():
        x = ds.data[:, parent]
        error = ds.data[:, column] - x
        assert error.std() == pytest.approx(noise_scale * x.std(), rel=0.02)
        assert abs(error.mean()) < 0.03 * x.std()
        others = [c for c in range(ds.data.shape[1]) if c != column]
        correlations = [np.corrcoef(error, ds.data[:, c])[0, 1] for c in others]
        assert np.max(np.abs(correlations)) < 0.02


@pytest.mark.parametrize("scale", SCALES)
def test_a_measurement_has_its_reliability_and_its_scale_in_every_mode(scale):
    """The squared correlation with the measured column is ``1 / (1 + noise_scale**2)``."""
    ds = _sample(measurement.noisy_children(noise_scale=1.0), n=40_000, scale=scale)
    assert ds.measurements.parents == {6 + j: j for j in range(6)}
    for column, parent in ds.measurements.parents.items():
        r = np.corrcoef(ds.data[:, column], ds.data[:, parent])[0, 1]
        assert r**2 == pytest.approx(0.5, rel=0.04)
    measured = ds.data[:, 6:]
    if scale == "standardize":
        np.testing.assert_allclose(measured.mean(axis=0), 0.0, atol=1e-12)
    if scale in ("standardize", "rescale"):
        np.testing.assert_allclose(measured.std(axis=0), 1.0)
    if scale == "unitless":
        np.testing.assert_allclose(measured.std(axis=0), 1.0, rtol=0.03)
    if scale == "raw":
        np.testing.assert_allclose(
            measured.var(axis=0), 2.0 * ds.data[:, :6].var(axis=0), rtol=0.03
        )


def test_the_measurement_noise_takes_its_family():
    spec = measurement.noisy_children([2], noise=data.noise.uniform())
    ds = _sample(spec, n=5000, scale="raw")
    x = ds.data[:, 2]
    error = ds.data[:, 6] - x
    assert np.max(np.abs(error)) <= np.sqrt(3.0) * 0.5 * x.std() * (1 + 1e-12)
    assert error.std() == pytest.approx(0.5 * x.std(), rel=0.03)
    assert ds.measurements.noise == "uniform"
    spec = measurement.noisy_children([2], noise=data.noise.student_t(df=6))
    config = _sample(spec, n=100).provenance["config"]["measurements"]
    assert config["noise"] == "student_t"
    assert config["noise_params"] == {"df": 6.0}


def test_a_share_draws_the_measured_columns_from_the_model_seed():
    spec = measurement.noisy_children(share=0.5)
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=10, avg_degree=2),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    a = scm.sample(n=300, seed=1, measurements=spec)
    assert a.data.shape == (300, 15)
    assert len(set(a.measurements.parents.values())) == 5
    assert list(a.measurements.parents.values()) == sorted(a.measurements.parents.values())
    assert np.array_equal(a.data, scm.sample(n=300, seed=1, measurements=spec).data)
    chosen = {
        tuple(scm.sample(n=20, seed=s, measurements=spec).measurements.parents.values())
        for s in range(8)
    }
    assert len(chosen) > 1

    first = scm.sample(n=300, seed=1, model_seed=7, measurements=spec)
    second = scm.sample(n=300, seed=2, model_seed=7, measurements=spec)
    assert first.measurements == second.measurements
    assert first.graph == second.graph
    assert np.array_equal(first.params.edges, second.params.edges)
    assert not np.array_equal(first.data[:, 10:], second.data[:, 10:])


def test_measurement_nodes_come_before_the_latent_nodes():
    plain = data.sample_scm(d=8, n=300, seed=0, density=3, latents=2)
    ds = data.sample_scm(
        d=8, n=300, seed=0, density=3, latents=2, measurements=measurement.noisy_children([0, 5])
    )
    assert ds.data.shape == (300, 8)
    assert np.array_equal(ds.data[:, :6], plain.data)
    assert ds.graph.n_nodes == 10
    assert ds.graph.node_types.tolist() == [OBSERVED] * 8 + [LATENT] * 2
    moved = {(p + 2 * (p >= 6), c + 2 * (c >= 6)) for p, c in _edges(plain.graph)}
    assert _edges(ds.graph) == moved | {(0, 6), (5, 7)}
    assert {tuple(e) for e in ds.params.edges.tolist()} == moved
    assert ds.params.noise_scales.shape == (10,)
    observed = ds.observed_graph
    assert observed.n_nodes == 8
    assert _edges(observed) == {(p, c) for p, c in _edges(ds.graph) if p < 8 and c < 8}
    assert {(0, 6), (5, 7)} <= _edges(observed)
    assert data.latent.marginal(ds.graph, "mag").n_nodes == 8


def test_missing_values_are_drawn_over_the_measurement_columns_too():
    spec = measurement.noisy_children([1, 2])
    complete = _sample(spec, n=1000)
    ds = _sample(spec, n=1000, missing=missingness.mcar(0.2))
    assert ds.missing.mask.shape == (1000, 8)
    assert ds.missing.mask[:, 6:].mean() == pytest.approx(0.2, abs=0.04)
    assert np.array_equal(ds.data[~ds.missing.mask], complete.data[~ds.missing.mask])
    assert ds.measurements == complete.measurements
    assert {"missing", "measurements"} <= set(ds.provenance["config"])


def test_measurements_follow_the_rows_of_every_environment():
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=6, avg_degree=2),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    envs = [{}, {2: hard(3.0)}]
    plain = scm.sample(n=[400, 400], seed=0, envs=envs)
    ds = scm.sample(n=[400, 400], seed=0, envs=envs, measurements=measurement.noisy_children([2]))
    assert np.array_equal(ds.data[:, :6], plain.data)
    assert ds.targets == plain.targets == {0: [], 1: [2]}
    assert np.array_equal(ds.env, plain.env)
    rows = ds.env == 1
    assert ds.data[rows, 6].mean() > ds.data[~rows, 6].mean() + 1.0


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: measurement.noisy_children([0], share=0.5), "columns or a share, not both"),
        (lambda: measurement.noisy_children([]), "distinct whole numbers >= 0"),
        (lambda: measurement.noisy_children([1, 1]), "distinct whole numbers >= 0"),
        (lambda: measurement.noisy_children([-1]), "distinct whole numbers >= 0"),
        (lambda: measurement.noisy_children([0.5]), "distinct whole numbers >= 0"),
        (lambda: measurement.noisy_children(share=0.0), r"share must be in \(0, 1\]"),
        (lambda: measurement.noisy_children(share=1.5), r"share must be in \(0, 1\]"),
        (lambda: measurement.noisy_children(noise_scale=0.0), "noise_scale must be finite"),
        (lambda: measurement.noisy_children(noise="gaussian"), "family of andrey.data.noise"),
        (
            lambda: measurement.noisy_children(noise=data.noise.gaussian(heteroscedastic=True)),
            "takes its scale from noise_scale",
        ),
    ],
)
def test_noisy_children_rejects_arguments_outside_their_range(build, match):
    with pytest.raises(ValueError, match=match):
        build()


def test_a_sample_rejects_measurements_it_cannot_draw():
    with pytest.raises(ValueError, match=r"name columns \[2, 6\], and the data have 6"):
        _sample(measurement.noisy_children([2, 6]), n=50)
    with pytest.raises(ValueError, match="temporal functionals take no measurements"):
        data.sample_scm(
            functional="var",
            d=6,
            n=50,
            seed=0,
            density=2,
            measurements=measurement.noisy_children(),
        )
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=6, avg_degree=2),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    with pytest.raises(ValueError, match=r"columns \[2\] are constant"):
        scm.do({2: hard(1.0)}).sample(n=50, seed=0, measurements=measurement.noisy_children([2]))
