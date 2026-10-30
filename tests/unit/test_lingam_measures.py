"""The skew and tanh pairwise measures of the LiNGAM order search (Hyvarinen and Smith 2013).

Both build the matrix ``M[i, j] = rho_ij E{x_i g(x_j) - g(x_i) x_j}`` of the standardized candidate
columns (Eq. 18): ``g = tanh`` (Eq. 5) and ``g(u) = -u^2`` after multiplying each column by the
sign of its skewness (Eqs. 10 and 13). The tests check the properties the paper proves and the
search relies on: the formula, antisymmetry, Theorem 2, invariance to each column's scale and sign,
the direction on large samples, and the order following the columns.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import andrey
from andrey.lingam import direct, multi_group, pwling
from andrey.spec.concepts import _CONCEPTS

_NEW = ("skew", "tanh")


def _standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / x.std(axis=0)


def _chain(noise, n: int, seed: int, rho: float = 0.7, d: int = 3) -> np.ndarray:
    """The chain ``x0 -> x1 -> ...`` with ``x_{k+1} = rho x_k + e_{k+1}``; ``noise`` draws ``e``."""
    rng = np.random.default_rng(seed)
    e = noise(rng, (n, d))
    x = np.empty((n, d))
    x[:, 0] = e[:, 0]
    for k in range(1, d):
        x[:, k] = rho * x[:, k - 1] + e[:, k]
    return x


def _laplace(rng: np.random.Generator, size) -> np.ndarray:
    return rng.laplace(size=size)


def _exponential(rng: np.random.Generator, size) -> np.ndarray:
    return rng.exponential(size=size) - 1.0


def _negative_exponential(rng: np.random.Generator, size) -> np.ndarray:
    return 1.0 - rng.exponential(size=size)


def test_measures_are_the_concept_choices() -> None:
    """One list of measures serves the search and the parameter every method checks."""
    concept = next(c for c in _CONCEPTS if c.key == "measure")
    assert concept.param is not None
    assert concept.param.choices == pwling.MEASURES
    assert concept.param.default == "pwling" == pwling.MEASURES[0]


@pytest.mark.parametrize("name", ["tanh", "skew"])
def test_matrix_is_the_pairwise_formula_and_antisymmetric(name: str) -> None:
    """``M`` equals ``rho E{x g(y) - g(x) y}`` pair by pair, and swapping the pair negates it."""
    x = _standardize(_chain(_exponential, 4000, seed=1, d=4))
    g = np.tanh if name == "tanh" else lambda u: -(u * u)
    m = pwling.nonlinear_correlations(x, g)
    for i, j in itertools.permutations(range(4), 2):
        rho = np.mean(x[:, i] * x[:, j])
        expected = rho * np.mean(x[:, i] * g(x[:, j]) - g(x[:, i]) * x[:, j])
        assert m[i, j] == pytest.approx(expected, rel=1e-10, abs=1e-14)
    np.testing.assert_allclose(m, -m.T, atol=1e-15)
    assert not np.diag(m).any()


def test_skew_entry_is_the_third_order_cumulant_statistic() -> None:
    """Eq. 10 with ``g(u) = -u^2`` is ``rho E{x^2 y - x y^2}``, the statistic of Theorem 2."""
    x = _standardize(_chain(_exponential, 4000, seed=2, d=2))
    m = pwling.nonlinear_correlations(x, lambda u: -(u * u))
    rho = np.mean(x[:, 0] * x[:, 1])
    assert m[0, 1] == pytest.approx(rho * np.mean(x[:, 0] ** 2 * x[:, 1] - x[:, 0] * x[:, 1] ** 2))


@pytest.mark.parametrize("rho", [0.6, -0.6])
def test_theorem_2_the_cause_gives_skew_times_rho_squared_minus_rho_cubed(rho: float) -> None:
    """Under x -> y, ``R = skew(x) (rho^2 - rho^3)`` for standardized columns (Theorem 2)."""
    rng = np.random.default_rng(3)
    e = rng.exponential(size=(400_000, 2)) - 1.0
    x = _standardize(np.column_stack([e[:, 0], rho * e[:, 0] + np.sqrt(1 - rho**2) * e[:, 1]]))
    skew_x = np.mean(x[:, 0] ** 3)
    r = np.mean(x[:, 0] * x[:, 1])
    m = pwling.nonlinear_correlations(x, lambda u: -(u * u))
    assert m[0, 1] == pytest.approx(skew_x * (r**2 - r**3), rel=0.05)


@pytest.mark.parametrize(
    ("name", "noise"),
    [
        ("skew", _exponential),
        ("skew", _negative_exponential),
        ("tanh", _exponential),
        ("tanh", _negative_exponential),
        ("tanh", _laplace),
    ],
)
def test_the_cause_scores_higher_in_large_samples(name: str, noise) -> None:
    """With skewed (either sign) or, for tanh, sparse noise, the cause of a pair scores higher."""
    x = _chain(noise, 50_000, seed=4, d=2)
    forward = pwling.scorer(name)(x, [0, 1])
    assert forward[0] == 0 and forward[1] < 0
    assert pwling.find_causal_order(x, measure=name) == [0, 1]
    assert pwling.find_causal_order(x[:, ::-1], measure=name) == [1, 0]


@pytest.mark.parametrize("name", _NEW)
def test_swapping_two_columns_swaps_their_scores(name: str) -> None:
    x = _chain(_exponential, 3000, seed=5, d=2)
    score = pwling.scorer(name)
    np.testing.assert_array_equal(score(x, [0, 1]), score(x[:, ::-1], [0, 1])[::-1])


@pytest.mark.parametrize("name", _NEW)
def test_the_order_ignores_each_columns_scale_and_sign(name: str) -> None:
    """Standardizing makes the measure unit-free; the skew correction and tanh make it sign-free."""
    x = _chain(_exponential, 3000, seed=6, d=4)
    order = pwling.find_causal_order(x, measure=name)
    scale = np.array([2.0**-9, 3.7, 2.0**11, 0.01])
    assert pwling.find_causal_order(x * scale, measure=name) == order
    assert pwling.find_causal_order(x * [1.0, -1.0, -1.0, 1.0], measure=name) == order
    # A power of two scales every float exactly, so the scores do not move at all.
    exact = 2.0 ** np.array([-9, 3, 11, 0])
    score = pwling.scorer(name)
    np.testing.assert_array_equal(score(x * exact, [0, 1, 2, 3]), score(x, [0, 1, 2, 3]))


@pytest.mark.parametrize("name", _NEW)
def test_the_order_follows_the_columns(name: str) -> None:
    """Any permutation of the columns gives the same order, relabeled (``column_order``)."""
    x = _chain(_exponential, 800, seed=7, d=3)
    order = pwling.find_causal_order(x, measure=name)
    for perm in itertools.permutations(range(3)):
        relabeled = pwling.find_causal_order(x[:, list(perm)], measure=name)
        assert [perm[j] for j in relabeled] == order, perm


@pytest.mark.parametrize("name", _NEW)
def test_a_copy_follows_its_original(name: str) -> None:
    """Copies are removed before scoring, so the measure never sees a column predicted exactly."""
    x = _chain(_exponential, 1000, seed=8, d=3)
    order = pwling.find_causal_order(np.column_stack([x, 3.0 * x[:, 1]]), measure=name)
    assert order.index(3) == order.index(1) + 1 or order.index(1) == order.index(3) + 1


@pytest.mark.parametrize("name", _NEW)
def test_precedence_is_kept(name: str) -> None:
    """A column that must come first does, though the data say otherwise."""
    x = _chain(_exponential, 20_000, seed=9, d=3)
    precedence = np.zeros((3, 3), dtype=bool)
    precedence[2, 0] = True  # x2 must precede x0
    order = pwling.find_causal_order(x, precedence=precedence, measure=name)
    assert order.index(2) < order.index(0)


@pytest.mark.parametrize("name", _NEW)
def test_every_ordering_method_takes_the_measure(name: str) -> None:
    """DirectLiNGAM, its multi-group form, and the two temporal LiNGAMs run with the measure."""
    x = _chain(_exponential, 4000, seed=10, d=3)
    assert andrey.direct_lingam(x, measure=name).ordering == (0, 1, 2)
    outs = andrey.multi_group_direct_lingam([x[:2000], x[2000:]], measure=name)
    assert all(o.ordering == (0, 1, 2) for o in outs)
    assert andrey.varma_lingam(x[:1500], order=(1, 0), measure=name) is not None
    panel = [x[k * 1000 : (k + 1) * 1000] for k in range(3)]
    assert andrey.longitudinal_lingam(panel, tau_max=1, measure=name) is not None


@pytest.mark.parametrize("fit", [direct.direct_lingam, multi_group.multi_group_direct_lingam])
def test_an_unknown_measure_raises_before_any_fit(fit) -> None:
    x = _chain(_exponential, 200, seed=11, d=3)
    data = x if fit is direct.direct_lingam else [x, x]
    with pytest.raises(ValueError, match="'pwling', 'skew', 'tanh'"):
        fit(data, measure="kernel")


def test_tanh_reads_sub_gaussian_data_backwards() -> None:
    """Uniform noise has negative kurtosis, where tanh's sign reverses (Theorem 1): its limit."""
    x = _chain(lambda r, s: r.uniform(-1, 1, s), 50_000, seed=12, d=2)
    assert pwling.find_causal_order(x, measure="tanh") == [1, 0]
    assert pwling.find_causal_order(x, measure="pwling") == [0, 1]
