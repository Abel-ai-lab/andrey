"""The nonlinear conditional-independence tests: fixed cases, reference values, and the contract.

Each test is checked on fixed seeded inputs with one certain answer (a dependence Fisher-Z misses,
an independence given the common cause, a dependence through shared noise), on closed forms its
parts must reproduce, and on the ``CITest`` contract: argument order, determinism, and input
validation. The permutation tests run with fewer draws than their defaults to keep the suite fast;
a permutation p-value ``(b + 1) / (B + 1)`` keeps its level at any number of draws ``B``. Their
rejection rates over many draws are a measurement, made by ``qa/ci_calibration.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2

import andrey
from andrey.core import ci
from andrey.core.ci_cmiknn import CMIknn, _cmi, _distances, _rank_columns
from andrey.core.ci_gpdc import GPDC, _centered_distances, _squared_distance_correlation
from andrey.core.ci_kernel import (
    KCI,
    RCoT,
    _fourier_features,
    _hall_buckley_eagleson_pvalue,
    _kci_moments,
)
from andrey.core.independence import hsic_test

ALPHA = 0.05

# name -> (class, settings for the fast suite)
TESTS = {
    "kci": (KCI, {}),
    "rcot": (RCoT, {}),
    "cmiknn": (CMIknn, {"n_permutations": 100}),
    "gpdc": (GPDC, {"n_null": 300}),
}


def _build(name: str, data: np.ndarray):
    cls, settings = TESTS[name]
    return cls(data, **settings)


def _common_cause(seed: int, n: int, n_cond: int, shared: float = 0.0) -> np.ndarray:
    """Columns ``[x, y, z_1, ..., z_k]``: ``x`` and ``y`` are different nonlinear functions of the
    conditioning variables plus noise, so ``x`` is independent of ``y`` given ``z``; ``shared``
    adds one noise term to both, which makes them dependent."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, n_cond))
    s = z.sum(axis=1) / np.sqrt(n_cond)
    u = rng.standard_normal(n)
    x = np.tanh(2.0 * s) + 0.4 * rng.standard_normal(n) + shared * u
    y = s**2 + 0.4 * rng.standard_normal(n) + shared * u
    return np.column_stack([x, y, z])


# ---- fixed cases -------------------------------------------------------------------------------


@pytest.mark.parametrize("name", TESTS)
def test_detects_a_dependence_partial_correlation_misses(name):
    """``y = x^2 + noise`` with symmetric ``x``: uncorrelated, yet strongly dependent. RCoT's few
    features need more than 200 rows to see it, so the case has 500."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(500)
    data = np.column_stack([x, x**2 + 0.3 * rng.standard_normal(500)])
    assert ci.FisherZ(data)(0, 1) > ALPHA
    assert _build(name, data)(0, 1) <= 0.01


@pytest.mark.parametrize("name", TESTS)
def test_separates_given_the_common_cause(name):
    """``x = tanh(2 z) + noise`` and ``y = z + z^2 / 2 + noise``: dependent through ``z`` alone."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal(200)
    x = np.tanh(2.0 * z) + 0.4 * rng.standard_normal(200)
    y = z + 0.5 * z * z + 0.4 * rng.standard_normal(200)
    test = _build(name, np.column_stack([x, y, z]))
    assert test(0, 1) <= 0.01
    assert test(0, 1, (2,)) > ALPHA


@pytest.mark.parametrize("n_cond", [1, 2])
@pytest.mark.parametrize("name", TESTS)
def test_rejects_a_conditional_dependence(name, n_cond):
    """Noise shared by ``x`` and ``y`` keeps them dependent given the conditioning variables."""
    data = _common_cause(seed=0, n=200, n_cond=n_cond, shared=0.7)
    assert _build(name, data)(0, 1, tuple(range(2, 2 + n_cond))) <= 0.01


@pytest.mark.parametrize(("cls", "draws"), [(CMIknn, "n_permutations"), (GPDC, "n_null")])
def test_permutation_pvalue_counts_the_observed_statistic(cls, draws):
    """A dependence no null draw reaches gets ``1 / (B + 1)``, never zero."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    data = np.column_stack([x, x + 0.1 * rng.standard_normal(200)])
    assert cls(data, **{draws: 49})(0, 1) == pytest.approx(1.0 / 50)


@pytest.mark.parametrize("name", TESTS)
def test_independent_discrete_columns_stay_independent(name):
    """Two independent binary columns: ties must not turn the row order into a dependence."""
    data = np.random.default_rng(0).integers(0, 2, size=(200, 2)).astype(float)
    assert _build(name, data)(0, 1) > ALPHA


# ---- reference values --------------------------------------------------------------------------


def test_kci_empty_set_is_the_gamma_hsic_test():
    """With no conditioning set KCI is the gamma-approximated HSIC test at the empirical width."""
    data = np.random.default_rng(1).standard_normal((300, 2))
    np.testing.assert_allclose(KCI(data)(0, 1), hsic_test(data[:, 0], data[:, 1]), rtol=1e-9)


def test_kci_null_moments_match_the_explicit_construction():
    """The mean ``tr(w w') / n`` and variance ``2 tr((w w')^2) / n^2`` of KCI's null law, built
    from ``w_t = vec(psi_t phi_t')`` with ``A = Psi Psi'`` and ``B = Phi Phi'``."""
    rng = np.random.default_rng(0)
    n = 25
    a, b = (m @ m.T / n for m in rng.standard_normal((2, n, n)))
    roots = []
    for kernel in (a, b):
        eigval, eigvec = np.linalg.eigh(kernel)
        roots.append(eigvec * np.sqrt(np.clip(eigval, 0.0, None)))
    psi, phi = roots
    w = np.stack([np.outer(psi[t], phi[t]).ravel() for t in range(n)], axis=1)
    ww = w @ w.T
    expected = (np.trace(a @ b) / n, np.trace(ww) / n, 2.0 * np.trace(ww @ ww) / n**2)
    np.testing.assert_allclose(_kci_moments(a, b), expected, rtol=1e-10)


@pytest.mark.parametrize("statistic", [0.5, 1.8, 4.0])
def test_rcot_pvalue_is_exact_for_equal_weights(statistic):
    """``0.3 * chi2(6)`` is a sum of six equally weighted squares; its law is matched exactly."""
    expected = chi2.sf(statistic / 0.3, df=6)
    assert _hall_buckley_eagleson_pvalue(statistic, np.full(6, 0.3)) == pytest.approx(expected)


@pytest.mark.parametrize("n_columns", [1, 3])
def test_rcot_features_approximate_the_median_bandwidth_kernel(n_columns):
    """Many features of a few rows average to ``exp(-d^2 / (2 s^2))``, ``s`` the median distance
    (Strobl et al. 2019, Sect. 4.1: Gaussian kernels with the median-distance bandwidth)."""
    values = np.random.default_rng(3).standard_normal((12, n_columns))
    features = _fourier_features(values, 40_000, np.random.default_rng(4))
    dists = squareform(pdist(values))
    scale = np.median(pdist(values))
    expected = np.exp(-(dists**2) / (2.0 * scale**2))
    np.testing.assert_allclose(features @ features.T / 40_000, expected, atol=0.03)


@pytest.mark.parametrize(("rho", "n_cond"), [(0.0, 0), (0.6, 0), (0.6, 1)])
def test_cmiknn_estimates_gaussian_mutual_information(rho, n_cond):
    """For jointly Gaussian variables with (partial) correlation ``rho`` the conditional mutual
    information is ``-log(1 - rho^2) / 2``; the ranks do not change it."""
    rng = np.random.default_rng(1)
    n = 1000
    z = rng.standard_normal(n)
    x = rng.standard_normal(n)
    y = rho * x + np.sqrt(1.0 - rho**2) * rng.standard_normal(n)
    if n_cond:  # z drives both, and rho is their correlation given z
        x, y = x + z, y + 0.8 * z
    ranks = _rank_columns(np.column_stack([x, y, z]), rng)
    dx, dy, dz = (_distances(ranks[:, c], np.int16) for c in range(3))
    estimate = _cmi(dx, dy, dz if n_cond else np.zeros_like(dx), k=50)
    assert estimate == pytest.approx(-0.5 * np.log(1.0 - rho**2), abs=0.03)


def test_gpdc_statistic_is_the_squared_distance_correlation():
    """The V-statistic ``dCov^2 = S1 + S2 - 2 S3`` of Szekely, Rizzo, and Bakirov (2007), over the
    product of the two distance variances."""
    rng = np.random.default_rng(0)
    u, v = rng.permutation(40), rng.permutation(40)

    def dcov2(p, q):
        dp, dq = np.abs(p[:, None] - p[None, :]), np.abs(q[:, None] - q[None, :])
        return (dp * dq).mean() + dp.mean() * dq.mean() - 2.0 * (dp.mean(1) * dq.mean(1)).mean()

    expected = dcov2(u, v) / np.sqrt(dcov2(u, u) * dcov2(v, v))
    statistic = _squared_distance_correlation(_centered_distances(u), _centered_distances(v))
    assert statistic == pytest.approx(expected, rel=1e-12)


# ---- the contract ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", TESTS)
def test_is_symmetric_and_deterministic(name):
    data = _common_cause(seed=2, n=120, n_cond=2)
    p = _build(name, data)(0, 1, (2, 3))
    assert _build(name, data)(1, 0, (3, 2)) == p
    assert _build(name, data)(0, 1, [3, 2, 3]) == p


@pytest.mark.parametrize("name", ["rcot", "cmiknn", "gpdc"])
def test_seed_reproduces_the_pvalue(name):
    """A random test's p-value follows its seed; ``None`` reads ``ANDREY_SEED``, default 0."""
    cls, settings = TESTS[name]
    data = _common_cause(seed=4, n=120, n_cond=1)
    pvalues = {seed: cls(data, seed=seed, **settings)(0, 1, (2,)) for seed in range(4)}
    assert cls(data, **settings)(0, 1, (2,)) == pvalues[0]
    assert len(set(pvalues.values())) > 1


@pytest.mark.parametrize("constant", [[0], [0, 1]])
@pytest.mark.parametrize("name", TESTS)
def test_constant_column_is_independent(name, constant):
    """A constant column has no variation to depend on, nor do two. CMIknn's random tie-breaking
    turns it into random ranks, so its p-value is a draw from the null rather than exactly 1."""
    data = _common_cause(seed=3, n=100, n_cond=1)
    data[:, constant] = 0.1
    test = _build(name, data)
    if name == "cmiknn":
        assert test(0, 1) > ALPHA
    else:
        assert test(0, 1) == 1.0
    assert test(0, 1, (2,)) > ALPHA


@pytest.mark.parametrize("name", TESTS)
@pytest.mark.parametrize(
    ("data", "match"),
    [(np.zeros(10), "2-D"), (np.full((10, 3), np.nan), "NaN or Inf")],
)
def test_rejects_bad_data(name, data, match):
    with pytest.raises(ValueError, match=match):
        _build(name, data)


@pytest.mark.parametrize(
    ("cls", "kwargs", "match"),
    [
        (KCI, {"ridge": 0.0}, "ridge"),
        (RCoT, {"n_features": 0}, "feature counts"),
        (CMIknn, {"k_cmi": 1.0}, "k_cmi"),
        (CMIknn, {"k_perm": 0}, "k_perm"),
        (GPDC, {"n_null": 0}, "n_null"),
    ],
)
def test_rejects_bad_settings(cls, kwargs, match):
    with pytest.raises(ValueError, match=match):
        cls(np.zeros((10, 3)), **kwargs)


# ---- the registry ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", TESTS)
def test_ships_as_experimental(name):
    """Each nonlinear test is selectable by name in PC and warns that it is experimental."""
    assert name in ci.SHIPPED_INDEP_TESTS
    andrey.core.warning_policy.reset()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = andrey.pc(_common_cause(seed=0, n=60, n_cond=1), indep_test=name)
    assert out.structure.kind == "cpdag"
    experimental = [w for w in caught if w.category is andrey.ExperimentalWarning]
    assert len(experimental) == 1
    assert f"indep_test={name!r} is experimental" in str(experimental[0].message)


@pytest.mark.parametrize(
    ("method", "kind"),
    [("fci", "pag"), ("gfci", "pag"), ("cdnod", "cpdag")],
)
def test_every_constraint_method_takes_the_nonlinear_tests(method, kind):
    """FCI, GFCI, and CD-NOD select the nonlinear tests by name, as PC does."""
    data = _common_cause(seed=0, n=60, n_cond=1)
    args = (data, np.repeat([0.0, 1.0], 30)) if method == "cdnod" else (data,)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        for name in TESTS:
            assert getattr(andrey, method)(*args, indep_test=name).structure.kind == kind
