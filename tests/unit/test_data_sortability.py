"""Sortability diagnostics, matched against the reference definitions.

Refutations: the chain case matches the literature, raw linear-Gaussian varsortability is high,
standardization drives varsortability to *exactly* 0.5, R^2-sortability *survives*
standardization unchanged (a caveat the tests check explicitly), R^2-sortability stays
above chance, the n>d singularity gate, and degenerate-input handling.
"""

from __future__ import annotations

import numpy as np

import andrey.data as data
from andrey.data.sortability import (
    directed_adjacency,
    r2sortability,
    standardize,
    varsortability,
    varsortability_edges,
)


def _linear_gauss(d, n, seed, density=2.0):
    ds = data.sample_scm(
        graph="erdos_renyi",
        functional="linear",
        noise="gaussian",
        d=d,
        n=n,
        seed=seed,
        density=density,
    )
    return ds.data.astype(np.float64), directed_adjacency(ds.params.edges, d)


def test_matches_reference_on_a_chain():
    """On a chain 0->1->2 with strictly increasing variance, every path is ordered -> exactly
    1.0."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5000, 3)) * np.array(
        [1.0, 2.0, 3.0]
    )  # increasing variance along 0<1<2
    adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # 0->1->2 (path 0->2 too)
    assert varsortability(x, adj) == 1.0
    # reversed variances -> every path is descending -> exactly 0.0
    x_rev = rng.standard_normal((5000, 3)) * np.array([3.0, 2.0, 1.0])
    assert varsortability(x_rev, adj) == 0.0


def test_raw_linear_gaussian_is_high():
    """Raw linear-Gaussian ER data has high varsortability (variance grows along the order)."""
    vals = [varsortability(*_linear_gauss(10, 1000, s)) for s in range(20)]
    assert np.mean(vals) > 0.85, np.mean(vals)


def test_standardize_is_exactly_half():
    """Standardizing every column makes all variances equal -> varsortability is exactly 0.5."""
    x, adj = _linear_gauss(12, 1000, 3)
    v = varsortability(standardize(x), adj)
    assert abs(v - 0.5) < 1e-9, v


def test_r2sortability_survives_standardization():
    """The caveat: R^2-sortability is bit-for-bit unchanged by standardization
    (scale-invariant)."""
    x, adj = _linear_gauss(8, 1000, 1)
    assert abs(r2sortability(x, adj) - r2sortability(standardize(x), adj)) < 1e-9


def test_r2sortability_above_chance():
    vals = [r2sortability(*_linear_gauss(10, 1000, s)) for s in range(20)]
    assert np.nanmean(vals) > 0.55, np.nanmean(vals)


def test_r2sortability_singular_when_n_leq_d():
    """r2sortability returns nan (never garbage) when n <= d makes corrcoef singular."""
    x, adj = _linear_gauss(20, 15, 0)  # n < d
    assert np.isnan(r2sortability(x, adj))


def test_degenerate_inputs_no_nan_leak():
    x = np.random.default_rng(0).standard_normal((100, 4))
    # path-free graph -> nan (defined), never a silent 0
    assert np.isnan(varsortability(x, np.zeros((4, 4), dtype=int)))
    assert np.isnan(varsortability_edges(x, np.empty((0, 2), dtype=int)))
    # a constant column must not leak nan into a well-defined varsortability
    x[:, 1] = 3.0
    adj = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    assert np.isfinite(varsortability(x, adj))


def test_edge_only_variant_scales():
    """The O(nnz) edge-only variant is a valid probability and needs no dense (d, d)."""
    ds = data.sample_scm(d=2000, n=64, seed=0, density=2.0)
    v = varsortability_edges(ds.data.astype(np.float64), ds.params.edges)
    assert 0.0 <= v <= 1.0
