"""The nonlinear conditional-independence tests: known cases, null calibration, and the contract.

Each test is checked on fixed seeded cases with one certain answer (a dependence Fisher-Z misses,
an independence given the common cause), on its rejection rate under the null over fixed seeds
(it must sit near ``alpha``), on its power when a shared noise term makes the pair dependent, and
on the ``CITest`` contract: argument order, determinism, and input validation. The permutation
tests run with fewer permutations than their defaults to keep the suite fast; a permutation
p-value is calibrated whatever their number.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
from andrey.core import ci
from andrey.core.ci_cmiknn import CMIknn
from andrey.core.ci_gpdc import GPDC
from andrey.core.ci_kernel import KCI, RCoT
from andrey.core.independence import hsic_test

ALPHA = 0.05

# name -> (class, settings for the fast suite)
TESTS = {
    "kci": (KCI, {}),
    "rcot": (RCoT, {}),
    "cmiknn": (CMIknn, {"n_permutations": 100}),
    "gpdc": (GPDC, {"n_null": 300}),
}

# name -> (rows, conditioning-set sizes) for the calibration and power checks. KCI's fixed kernel
# widths hold their level with one conditioning variable only (see its Limitations).
CALIBRATION = {
    "kci": (200, [1]),
    "rcot": (500, [1, 2]),
    "cmiknn": (100, [1, 2]),
    "gpdc": (100, [1, 2]),
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


def _rejections(name: str, seeds, n: int, n_cond: int, shared: float = 0.0) -> int:
    cond = tuple(range(2, 2 + n_cond))
    return sum(
        _build(name, _common_cause(seed, n, n_cond, shared))(0, 1, cond) <= ALPHA for seed in seeds
    )


def _calibration_cases():
    return [(name, k) for name, (_, sizes) in CALIBRATION.items() for k in sizes]


# ---- known cases -------------------------------------------------------------------------------


@pytest.mark.parametrize("name", TESTS)
def test_detects_a_dependence_partial_correlation_misses(name):
    """``y = x^2 + noise`` with symmetric ``x``: uncorrelated, yet strongly dependent."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    data = np.column_stack([x, x**2 + 0.3 * rng.standard_normal(200)])
    assert ci.FisherZ(data)(0, 1) > ALPHA
    assert _build(name, data)(0, 1) <= 0.01


@pytest.mark.parametrize("name", TESTS)
def test_separates_given_the_common_cause(name):
    test = _build(name, _common_cause(seed=0, n=200, n_cond=1))
    assert test(0, 1) <= 0.01  # marginally dependent through z
    assert test(0, 1, (2,)) > ALPHA


def test_kci_empty_set_is_the_gamma_hsic_test():
    """With no conditioning set KCI is the gamma-approximated HSIC test at the empirical width."""
    data = np.random.default_rng(1).standard_normal((300, 2))
    np.testing.assert_allclose(KCI(data)(0, 1), hsic_test(data[:, 0], data[:, 1]), rtol=1e-9)


# ---- calibration and power ---------------------------------------------------------------------


@pytest.mark.parametrize(("name", "n_cond"), _calibration_cases())
def test_null_rejection_rate_is_near_alpha(name, n_cond):
    """Over 200 seeded draws with ``x`` independent of ``y`` given ``z``, about 5% reject."""
    n, _ = CALIBRATION[name]
    rejected = _rejections(name, range(200), n=n, n_cond=n_cond)
    assert 3 <= rejected <= 20, rejected


@pytest.mark.parametrize(("name", "n_cond"), _calibration_cases())
def test_rejects_a_conditional_dependence(name, n_cond):
    n, _ = CALIBRATION[name]
    rejected = _rejections(name, range(20), n=n, n_cond=n_cond, shared=0.7)
    assert rejected >= 16, rejected


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


@pytest.mark.parametrize("name", TESTS)
def test_constant_column_is_independent(name):
    """A constant column has no variation to depend on. CMIknn's jitter turns it into random
    ranks, so its p-value is a draw from the null rather than exactly 1."""
    data = _common_cause(seed=3, n=100, n_cond=1)
    data[:, 0] = 1.0
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
    """Each nonlinear test is selectable by name and warns that it is experimental."""
    assert name in ci.SHIPPED_INDEP_TESTS
    andrey.core.warning_policy.reset()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ci.make_indep_test(name, _common_cause(seed=0, n=50, n_cond=1))
    assert [w.category for w in caught] == [andrey.ExperimentalWarning]
    assert f"indep_test={name!r} is experimental" in str(caught[0].message)
