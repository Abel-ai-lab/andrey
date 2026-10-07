"""Check QA reports, scaling modes, varsortability controls, and faithfulness screening."""

from __future__ import annotations

import numpy as np
import pytest

import andrey.data as data
from andrey.data.sortability import directed_adjacency, varsortability


def test_report_populated_and_scalable():
    ds = data.sample_scm(d=40, n=1000, seed=0)
    assert isinstance(ds.report.varsortability, float)
    assert isinstance(ds.report.r2sortability, float)  # d<=100, n>d -> computed
    assert ds.report.scale == "raw"
    big = data.sample_scm(d=2000, n=64, seed=0)  # r2 skipped at scale, varsortability still runs
    assert np.isfinite(big.report.varsortability)
    assert np.isnan(big.report.r2sortability)


def test_standardize_mode_neutralizes_varsortability():
    """Standardization gives unit column variance and report varsortability of 0.5.

    Equal variances fall within the report's tie tolerance, so every edge scores 0.5.
    """
    ds = data.sample_scm(d=30, n=1000, seed=1, scale="standardize")
    assert abs(ds.report.varsortability - 0.5) < 1e-9
    assert ds.report.scale == "standardize"
    assert np.allclose(ds.data.std(axis=0), 1.0, atol=1e-4)


def test_rescale_mode_kills_varsortability():
    """Ancestral rescaling gives report varsortability of 0.5."""
    ds = data.sample_scm(
        functional="linear", noise="uniform", d=8, n=1000, seed=0, density=2.0, scale="rescale"
    )
    assert abs(ds.report.varsortability - 0.5) < 1e-9


def test_permuted_truth_sensitivity_control():
    """Varsortability is high for the true graph and near 0.5 for permuted truth."""
    ds = data.sample_scm(d=10, n=1000, seed=0, density=2.0)
    x = ds.data.astype(np.float64)
    vs_true = varsortability(x, directed_adjacency(ds.params.edges, 10))
    pi = np.random.default_rng(0).permutation(10)
    vs_perm = varsortability(x, directed_adjacency(pi[ds.params.edges], 10))
    assert vs_true > 0.85
    assert abs(vs_perm - 0.5) < 0.2


def test_faithfulness_screen_accepts_and_records():
    screen = data.FaithfulnessScreen(min_abs_corr=0.01, max_attempts=10)
    ds = data.SCM(
        graph=data.graphs.erdos_renyi(8, 2.0),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    ).sample(n=500, seed=0, screen=screen)
    assert ds.report.faithfulness_accept_rate is not None
    assert 0.0 < ds.report.faithfulness_accept_rate <= 1.0


def test_faithfulness_screen_bounded_raises():
    """The faithfulness screen raises after the configured attempt limit."""
    screen = data.FaithfulnessScreen(min_abs_corr=0.999, max_attempts=5)
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(10, 2.0),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    with pytest.raises(RuntimeError, match="faithfulness screen"):
        scm.sample(n=500, seed=0, screen=screen)


def test_raw_default_unchanged_by_new_params():
    """Default sampling matches explicit raw scaling for the same seed."""
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(12, 2.0),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    assert np.array_equal(
        scm.sample(n=200, seed=3).data, scm.sample(n=200, seed=3, scale="raw").data
    )


def test_faithfulness_screen_with_latents_does_not_crash():
    """Faithfulness screening accepts samples with latent variables."""
    screen = data.FaithfulnessScreen(min_abs_corr=0.01, max_attempts=10)
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(30, 3.0),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
        latents=3,
    )
    ds = scm.sample(n=300, seed=0, screen=screen)
    assert ds.data.shape[1] == 27
    assert ds.report.faithfulness_accept_rate is not None
