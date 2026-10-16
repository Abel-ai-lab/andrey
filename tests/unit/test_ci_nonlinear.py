"""The nonlinear conditional-independence tests: fixed cases, reference values, and the contract.

Each test is checked on fixed seeded inputs with one certain answer (a dependence Fisher-Z misses,
an independence given the common cause, a dependence through shared noise), on closed forms its
parts must reproduce, and on the ``CITest`` contract: argument order, determinism, and input
validation. The permutation tests run with fewer draws than their defaults to keep the suite fast;
a permutation p-value ``(b + 1) / (B + 1)`` keeps its level at any number of draws ``B``. Their
rejection rates over many draws are a measurement, made by ``qa/ci_calibration.py``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2, gamma
from sklearn.gaussian_process import GaussianProcessRegressor
from threadpoolctl import threadpool_info, threadpool_limits

import andrey
from andrey.constraint.pc import pc as pc_engine
from andrey.core import backend, ci, ci_cmiknn, ci_kernel
from andrey.core.ci_cmiknn import (
    CMIknn,
    _cmi,
    _distances,
    _local_permutation,
    _nearest_in_z,
    _rank_columns,
)
from andrey.core.ci_gpdc import GPDC, _centered_distances, _squared_distance_correlation
from andrey.core.ci_kernel import (
    KCI,
    RCoT,
    _fourier_features,
    _gamma_pvalue,
    _hall_buckley_eagleson_pvalue,
    _kci_moments,
)

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


def test_kci_empty_set_is_the_unconditional_gamma_law():
    """Zhang et al. (2011), Sect. 3.4: ``T = tr(Kx Ky) / n`` against the gamma law with mean
    ``tr(Kx) tr(Ky) / n^2`` and variance ``2 tr(Kx^2) tr(Ky^2) / n^4``, ``K`` the centered
    Gaussian kernel of width 0.5 (300 rows) on standardized columns."""
    data = np.random.default_rng(1).standard_normal((300, 2))
    data[:, 1] += 0.2 * data[:, 0] ** 2
    n = 300
    center = np.eye(n) - 1.0 / n

    def kernel(column):
        z = (column - column.mean()) / column.std(ddof=1)
        return center @ np.exp(-((z[:, None] - z[None, :]) ** 2) / (2.0 * 0.5**2)) @ center

    kx, ky = kernel(data[:, 0]), kernel(data[:, 1])
    statistic = np.trace(kx @ ky) / n
    mean = np.trace(kx) * np.trace(ky) / n**2
    var = 2.0 * np.trace(kx @ kx) * np.trace(ky @ ky) / n**4
    expected = gamma.sf(statistic, mean**2 / var, scale=var / mean)
    np.testing.assert_allclose(KCI(data)(0, 1), expected, rtol=1e-9)


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


@pytest.mark.parametrize("n_cond", [1, 2])
def test_kci_low_rank_matches_the_exact_test(n_cond):
    """With a rank that reproduces every kernel, the low-rank p-value is the exact one; with a
    smaller rank it stays close."""
    data = _common_cause(seed=6, n=150, n_cond=n_cond, shared=0.2)
    cond = tuple(range(2, 2 + n_cond))
    exact = KCI(data)(0, 1, cond)
    assert KCI(data, rank=150)(0, 1, cond) == pytest.approx(exact, rel=1e-6, abs=1e-12)
    assert KCI(data, rank=40)(0, 1, cond) == pytest.approx(exact, rel=0.05, abs=1e-3)


@pytest.mark.parametrize("statistic", [0.5, 1.8, 4.0])
def test_rcot_pvalue_is_exact_for_equal_weights(statistic):
    """``0.3 * chi2(6)`` is a sum of six equally weighted squares; its law is matched exactly."""
    expected = chi2.sf(statistic / 0.3, df=6)
    assert _hall_buckley_eagleson_pvalue(statistic, np.full(6, 0.3)) == pytest.approx(expected)


def test_rcot_pvalue_matches_the_exact_tail_for_unequal_weights():
    """``2 chi2(1) + 0.1 chi2(20)`` four standard deviations above its mean: matching three
    cumulants gives the exact tail within 5%; matching two (a gamma law) gives half of it."""
    weights = np.array([2.0] + [0.1] * 20)
    statistic = weights.sum() + 4.0 * np.sqrt(2.0 * np.sum(weights**2))
    inner, _ = quad(
        lambda v: chi2.pdf(v, 20) * chi2.sf((statistic - 0.1 * v) / 2.0, 1), 0.0, 10.0 * statistic
    )
    exact = inner + chi2.sf(10.0 * statistic, 20)
    assert _hall_buckley_eagleson_pvalue(statistic, weights) == pytest.approx(exact, rel=0.05)


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


def test_cmiknn_local_shuffle_stays_near_in_z():
    """Each row takes ``x`` from a row no farther in ``z`` than its ``k_perm``-th nearest, rows
    repeat rarely, and the dependence of ``x`` on ``z`` stays."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal(300)
    x = z + 0.3 * rng.standard_normal(300)
    ranks = _rank_columns(np.column_stack([x, z]), rng)
    dz = _distances(ranks[:, 1], np.int16)
    perm = _local_permutation(_nearest_in_z(dz, 5, rng), rng)
    kth_nearest = np.sort(dz, axis=1)[:, 4]  # position 0 holds the row itself
    assert np.all(dz[np.arange(300), perm] <= kth_nearest)
    assert len(set(perm)) >= 0.9 * 300
    assert np.corrcoef(x[perm], z)[0, 1] == pytest.approx(np.corrcoef(x, z)[0, 1], abs=0.02)


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


def test_gpdc_fits_each_variable_and_conditioning_set_once(monkeypatch):
    """A regression is reused by later queries on the same variable and conditioning set, and the
    p-values equal those of a test that fits every query afresh."""
    data = _common_cause(seed=3, n=120, n_cond=2)
    fits = []
    fit = GaussianProcessRegressor.fit
    monkeypatch.setattr(
        GaussianProcessRegressor, "fit", lambda self, X, y: fits.append(1) or fit(self, X, y)
    )
    queries = [(0, 1, (2,)), (0, 3, (2,)), (1, 3, (2,)), (0, 1, (2, 3))]
    test = _build("gpdc", data)
    pvalues = [test(*query) for query in queries]
    assert len(fits) == 5  # (0, {2}), (1, {2}), (3, {2}), (0, {2, 3}), (1, {2, 3})
    assert [_build("gpdc", data)(*query) for query in queries] == pvalues


def test_rcot_draws_the_features_of_each_column_set_once(monkeypatch):
    """Queries share the features of a variable and of a conditioning set, and the p-values equal
    those of a test that draws every query's features afresh."""
    draws = []
    draw = ci_kernel._fourier_features
    monkeypatch.setattr(
        ci_kernel,
        "_fourier_features",
        lambda v, count, rng: draws.append(count) or draw(v, count, rng),
    )
    data = _common_cause(seed=3, n=120, n_cond=2)
    queries = [(0, 1, (2,)), (0, 3, (2,)), (1, 3, (2,)), (0, 1, ()), (0, 1, (2, 3))]
    test = RCoT(data)
    pvalues = [test(*query) for query in queries]
    assert sorted(draws) == [5, 5, 5, 100, 100]  # columns 0, 1, 3; sets {2}, {2, 3}
    assert [RCoT(data)(*query) for query in queries] == pvalues


def test_cmiknn_draws_the_shuffles_of_each_conditioning_set_once(monkeypatch):
    """Queries with the same conditioning set share its shuffles, and the p-values equal those of a
    test that draws every query's shuffles afresh."""
    draws = []
    shuffle = ci_cmiknn._local_permutation
    monkeypatch.setattr(
        ci_cmiknn, "_local_permutation", lambda *args: draws.append(1) or shuffle(*args)
    )
    data = _common_cause(seed=3, n=120, n_cond=2)
    queries = [(0, 1, (2,)), (0, 3, (2,)), (1, 3, (2,)), (0, 1, (2, 3))]
    test = CMIknn(data, n_permutations=20)
    pvalues = [test(*query) for query in queries]
    assert len(draws) == 2 * 20  # sets {2} and {2, 3}
    assert [CMIknn(data, n_permutations=20)(*query) for query in queries] == pvalues


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.3])
def test_cmiknn_bounded_pvalue_decides_as_every_shuffle(alpha):
    """With ``alpha`` the shuffles stop once the side of ``alpha`` is certain: the p-value bounds
    the full one on the same side, for queries on both sides."""
    rng = np.random.default_rng(5)
    queries = [(0, 1, ()), (0, 1, (2,)), (0, 1, (2, 3))]
    sides, stopped = set(), 0
    for shared in (0.0, 0.1, 0.2, 0.3, 1.0):
        data = _common_cause(seed=int(rng.integers(100)), n=150, n_cond=2, shared=shared)
        test = CMIknn(data, n_permutations=100)
        for query in queries:
            full, bounded = test(*query), test._bounded_pvalue(*query, alpha)
            assert (bounded > alpha) == (full > alpha)
            assert bounded <= full if full > alpha else bounded >= full
            sides.add(full > alpha)
            stopped += bounded != full
    assert sides == {True, False} and stopped


def test_pc_with_cmiknn_decides_as_with_every_shuffle(monkeypatch):
    """PC's skeleton with CMIknn's early stop keeps the graph and every test's decision."""
    data = _common_cause(seed=7, n=150, n_cond=3, shared=0.3)

    def run(early_stop: bool):
        def factory(x):
            test = CMIknn(x, n_permutations=100)
            # Without _bounded_pvalue the scan calls test(a, b, subset).
            return test if early_stop else lambda a, b, subset=(): test(a, b, subset)

        monkeypatch.setitem(ci._CI_REGISTRY, "cmiknn_run", factory)
        trace = []
        graph = pc_engine(data, indep_test="cmiknn_run", _trace=trace)
        tests = [(r["x"], r["y"], r["S"], r["removed"]) for r in trace if r["step"] == "test"]
        return graph.to_numpy(), tests

    graph, tests = run(early_stop=True)
    full_graph, full_tests = run(early_stop=False)
    assert np.array_equal(graph, full_graph) and tests == full_tests


# ---- the contract ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", TESTS)
def test_is_symmetric_and_deterministic(name):
    data = _common_cause(seed=2, n=120, n_cond=2)
    p = _build(name, data)(0, 1, (2, 3))
    assert _build(name, data)(1, 0, (3, 2)) == p
    assert _build(name, data)(0, 1, [3, 2, 3]) == p


@pytest.mark.parametrize("name", ["rcot", "cmiknn", "gpdc"])
def test_seed_reproduces_the_pvalue(name, monkeypatch):
    """A random test's p-value follows its seed, not the queries before it; ``None`` reads
    ``ANDREY_SEED``, default 0."""
    cls, settings = TESTS[name]
    data = _common_cause(seed=4, n=120, n_cond=1)
    pvalues = {seed: cls(data, seed=seed, **settings)(0, 1, (2,)) for seed in range(4)}
    assert len(set(pvalues.values())) > 1
    monkeypatch.delenv("ANDREY_SEED", raising=False)
    test = cls(data, **settings)
    assert test(0, 1, (2,)) == pvalues[0]
    test(0, 2)
    assert test(0, 1, (2,)) == pvalues[0]
    monkeypatch.setenv("ANDREY_SEED", "2")
    assert cls(data, **settings)(0, 1, (2,)) == pvalues[2]


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
    [
        (np.zeros(10), "2-D"),
        (np.full((10, 3), np.nan), "NaN or Inf"),
        (np.zeros((0, 3)), "at least 2 rows"),
        (np.ones((1, 3)), "at least 2 rows"),
    ],
)
def test_rejects_bad_data(name, data, match):
    with pytest.raises(ValueError, match=match):
        _build(name, data)


@pytest.mark.parametrize("name", ["kci", "rcot", "gpdc"])
def test_rejects_data_too_large_to_standardize(name):
    """Values near 1e200 overflow the variance; the test raises rather than read every column as
    constant."""
    data = _common_cause(seed=0, n=100, n_cond=1) * 1e200
    with pytest.raises(ValueError, match="too large"):
        _build(name, data)


@pytest.mark.parametrize("name", ["kci", "rcot", "gpdc"])
def test_rejects_data_too_small_to_standardize(name):
    """Values near 1e-170 underflow the variance to zero while the range stays positive."""
    data = _common_cause(seed=0, n=100, n_cond=1) * 1e-170
    with pytest.raises(ValueError, match="too small"):
        _build(name, data)


@pytest.mark.parametrize(
    "pvalue",
    [
        lambda: _gamma_pvalue(1.0, np.nan, 1.0),
        lambda: _hall_buckley_eagleson_pvalue(1.0, np.full(3, np.nan)),
    ],
    ids=["kci", "rcot"],
)
def test_null_law_of_nan_raises(pvalue):
    """A NaN reaching the null law is an error, not a constant variable."""
    with pytest.raises(FloatingPointError):
        pvalue()


@pytest.mark.parametrize(
    ("cls", "kwargs", "match"),
    [
        (KCI, {"ridge": 0.0}, "ridge"),
        (KCI, {"ridge": np.inf}, "ridge"),
        (KCI, {"rank": 0}, "rank"),
        (RCoT, {"n_features": 0}, "feature counts"),
        (CMIknn, {"k_cmi": 1.0}, "k_cmi"),
        (CMIknn, {"k_perm": 0}, "k_perm"),
        (GPDC, {"n_null": 0}, "n_null"),
    ],
)
def test_rejects_bad_settings(cls, kwargs, match):
    with pytest.raises(ValueError, match=match):
        cls(np.zeros((10, 3)), **kwargs)


@pytest.mark.parametrize(
    ("n", "condition_set", "warns"), [(499, (2,), True), (499, (), False), (500, (2,), False)]
)
def test_rcot_warns_on_a_conditional_query_below_500_rows(n, condition_set, warns):
    """Below 500 rows RCoT warns on a query with a conditioning set, and still answers; an
    unconditional query, or 500 rows, does not warn."""
    andrey.core.warning_policy.reset()
    test = RCoT(np.random.default_rng(0).standard_normal((n, 3)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pvalue = test(0, 1, condition_set)
    assert 0.0 <= pvalue <= 1.0
    assert [w.category for w in caught] == ([andrey.SampleSizeWarning] if warns else [])


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


def _blas_threads() -> int:
    return max(info["num_threads"] for info in threadpool_info() if info["user_api"] == "blas")


@pytest.mark.parametrize("method", ["pc", "fci"])
@pytest.mark.parametrize("pin_blas", [None, False])
def test_skeleton_pins_blas_unless_the_test_opts_out(method, pin_blas, monkeypatch):
    """The skeleton scan runs a test on one BLAS thread, as Fisher-Z needs, unless the test sets
    ``pin_blas = False``, as the nonlinear tests do."""
    assert all(cls.pin_blas is False for cls, _ in TESTS.values())
    assert getattr(ci.FisherZ, "pin_blas", True) and getattr(ci.StudentT, "pin_blas", True)
    seen = []

    class Recording:
        def __init__(self, data):
            self._test = ci.FisherZ(data)

        def __call__(self, x, y, condition_set=()):
            seen.append(_blas_threads())
            return self._test(x, y, condition_set)

    if pin_blas is not None:
        Recording.pin_blas = pin_blas
    monkeypatch.setitem(ci._CI_REGISTRY, "recording", Recording)
    with threadpool_limits(limits=2, user_api="blas"):
        if _blas_threads() < 2:
            pytest.skip("the BLAS library runs one thread")
        getattr(andrey, method)(_common_cause(seed=0, n=60, n_cond=2), indep_test="recording")
    assert seen and set(seen) == {1 if pin_blas is None else 2}


# ---- device dispatch ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "module"), [("kci", ci_kernel), ("cmiknn", ci_cmiknn)])
def test_device_dispatch_gates_on_rows_squared(name, module, monkeypatch):
    """KCI and CMIknn ask for a device by rows squared, may start one, and never take ``mps``."""
    seen = []

    def spy(pinnable, auto_order, *, metric=None, thresholds=None, amortized=True, op=None):
        seen.append((metric, amortized, op, "mps" in pinnable))
        return "numpy"

    monkeypatch.setattr(module.backend, "select", spy)
    _build(name, _common_cause(seed=0, n=60, n_cond=1))(0, 1, [2])
    assert set(seen) == {(60 * 60, True, name, False)}


@pytest.mark.parametrize(
    ("choose", "cutoffs", "var"),
    [
        (ci_kernel._kci_backend, ci_kernel._KCI_GPU_THRESHOLDS, "ANDREY_KCI_GPU_THRESHOLD"),
        (
            ci_cmiknn._cmiknn_backend,
            ci_cmiknn._CMIKNN_GPU_THRESHOLDS,
            "ANDREY_CMIKNN_GPU_THRESHOLD",
        ),
    ],
)
def test_device_cutoff_is_rows_squared_and_reads_its_variable(choose, cutoffs, var, monkeypatch):
    for name in ("ANDREY_BACKEND", "ANDREY_DEVICE", var):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(backend, "_is_available", lambda b: b in ("cuda", "mps", "numpy"))
    first = math.isqrt(cutoffs["cuda"] - 1) + 1  # the fewest rows whose square reaches the cutoff
    assert (choose(first - 1), choose(first)) == ("numpy", "cuda")
    monkeypatch.setenv(var, "100")
    assert (choose(9), choose(10)) == ("numpy", "cuda")
    with andrey.config(backend="mps"):
        assert choose(10) == "cuda"  # float32 mps is not a backend of these tests


def test_device_cutoffs_never_choose_mps():
    assert ci_kernel._KCI_GPU_THRESHOLDS["mps"] is None
    assert ci_cmiknn._CMIKNN_GPU_THRESHOLDS["mps"] is None


@pytest.mark.parametrize("n_cond", [0, 1, 2])
def test_torch_paths_match_numpy(n_cond):
    """On torch's CPU backend, KCI's p-value matches within rounding and CMIknn's is the same,
    with and without an ``alpha`` that stops the shuffles."""
    pytest.importorskip("torch")
    x = _common_cause(seed=3, n=150, n_cond=max(n_cond, 1), shared=0.3)
    kci, cmiknn = KCI(x), CMIknn(x, n_permutations=60)
    cond = list(range(2, 2 + n_cond))
    expected = (kci(0, 1, cond), [cmiknn._bounded_pvalue(0, 1, cond, a) for a in (None, 0.05)])
    with andrey.config(backend="cpu"):
        got = (kci(0, 1, cond), [cmiknn._bounded_pvalue(0, 1, cond, a) for a in (None, 0.05)])
    assert got[0] == pytest.approx(expected[0], abs=1e-9)
    assert got[1] == expected[1]
