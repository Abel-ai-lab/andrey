"""Sortability diagnostics, matched against the reference definitions.

Refutations: the chain case matches the literature, raw linear-Gaussian varsortability is high,
standardization drives varsortability to *exactly* 0.5, R^2-sortability *survives*
standardization unchanged (a caveat the tests check explicitly), R^2-sortability stays
above chance, the n>d singularity gate, and degenerate-input handling. The column-order
correlation has its exact moments under every order of a small graph.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey.data as data
from andrey.data.sortability import (
    causal_depth,
    directed_adjacency,
    index_correlation,
    r2sortability,
    standardize,
    varsortability,
    varsortability_edges,
)


def _linear_gauss(d, n, seed, density=2.0, scale="raw"):
    ds = data.sample_scm(
        graph="erdos_renyi",
        functional="linear",
        noise="gaussian",
        d=d,
        n=n,
        seed=seed,
        density=density,
        scale=scale,
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


def test_unitless_r2sortability_is_near_chance():
    """On random graphs the unitless draws are R^2-sortable near chance, where the raw (and so the
    standardized) draws of the same graphs are well above it.

    Over 30 graphs the mean's standard error is about 0.03, so the 0.1 margin is three of them.
    """
    seeds = range(30)
    unitless = [r2sortability(*_linear_gauss(20, 400, s, scale="unitless")) for s in seeds]
    raw = [r2sortability(*_linear_gauss(20, 400, s)) for s in seeds]
    assert abs(np.nanmean(unitless) - 0.5) < 0.1, np.nanmean(unitless)
    assert np.nanmean(raw) > 0.65, np.nanmean(raw)


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
    ds = data.sample_scm(d=2000, n=64, seed=0, density=2.0, scale="raw")
    v = varsortability_edges(ds.data.astype(np.float64), ds.params.edges)
    assert 0.0 <= v <= 1.0


def test_causal_depth_is_the_longest_path_into_each_node():
    # 0 -> 1 -> 3 and 0 -> 3: node 3 sits two edges deep; 2 is a root.
    edges = np.array([[0, 1], [1, 3], [0, 3], [2, 3]])
    np.testing.assert_array_equal(causal_depth(edges, 5), [0, 1, 0, 2, 0])
    with pytest.raises(ValueError, match="cycle"):
        causal_depth(np.array([[0, 1], [1, 0]]), 2)


def test_index_correlation_has_mean_zero_and_variance_one_over_k_minus_one():
    """Over every order of five columns, ties included, the moments are exact."""
    from itertools import permutations

    edges = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])  # depths 0, 1, 1, 2 and an isolated 0
    rho = [
        index_correlation(np.argsort(order)[edges], 5)
        for order in map(np.array, permutations(range(5)))
    ]
    assert np.mean(rho) == pytest.approx(0.0, abs=1e-12)
    assert np.var(rho) == pytest.approx(1 / 4, abs=1e-12)
    assert np.isnan(index_correlation(np.empty((0, 2), dtype=int), 3))


def test_index_correlation_counts_hidden_nodes_toward_the_depths_only():
    # The hidden node 2 causes column 0, which causes column 1.
    edges = np.array([[2, 0], [0, 1]])
    assert index_correlation(edges, 3, n_observed=2) == pytest.approx(1.0)
    assert index_correlation(np.array([[2, 1], [1, 0]]), 3, n_observed=2) == pytest.approx(-1.0)
