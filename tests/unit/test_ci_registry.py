"""The pluggable CI registry: any registered ``CITest`` is selectable end-to-end via ``indep_test``.

The constraint facades (pc / fci / cdnod / gfci) resolve ``indep_test`` through
:mod:`andrey.core.ci`'s registry. These tests pin its surface: every shipped name builds, an
unregistered name still raises, the default Fisher-Z path is unchanged, and each constraint facade
takes the partial-correlation tests by name. ``test_concepts.py`` checks that every search asks a
registered test with the bare ``__call__`` Protocol. CALM is out of scope (internal Markov-blanket
pruning, no facade ``indep_test``).
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.core import ci
from tests.recovery.helpers.ci_tests import PValueOnly
from tests.unit.methods import domains


@pytest.fixture
def clean_registry():
    """Snapshot and restore the CI registry so a test's registrations do not leak into others."""
    snapshot = dict(ci._CI_REGISTRY)
    try:
        yield
    finally:
        ci._CI_REGISTRY.clear()
        ci._CI_REGISTRY.update(snapshot)


def _chain_data(seed: int = 0, n: int = 500, d: int = 6) -> np.ndarray:
    """A linear chain so a sparse skeleton leaves non-adjacent pairs for the CI test to query."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    for j in range(1, d):
        x[:, j] += 0.6 * x[:, j - 1]
    return x


def test_the_statistic_of_a_p_value_grows_as_it_shrinks():
    p_values = np.array([1.0, 0.5, 0.05, 1e-10, 1e-300])
    statistics = ci.p_value_statistic(p_values)
    assert statistics[0] == 0.0
    assert (np.diff(statistics) > 0).all()
    assert np.isfinite(ci.p_value_statistic(0.0))  # an underflowed p-value stays finite


def test_a_test_with_p_values_only_answers_a_batch_query_by_query():
    """The adapter the PCMCI family wraps such a test in gives the test's p-values exactly, and
    statistics that order the queries by p-value, smallest first."""
    test = PValueOnly(_chain_data(d=5))
    x, y = np.array([0, 0, 0, 1, 3]), np.array([4, 3, 2, 4, 4])
    conditions = np.array([[1], [1], [1], [2], [0]])
    statistics, p_values = ci.PValueTest(test).statistic_pvalues(x, y, conditions)
    assert p_values.tolist() == [test(a, b, c) for a, b, c in zip(x, y, conditions.tolist())]
    assert np.array_equal(np.argsort(-statistics), np.argsort(p_values))


def test_unknown_indep_test_raises():
    """An unregistered name raises ValueError, naming what is available."""
    with pytest.raises(ValueError, match="indep_test must be one of .*'fisherz'"):
        andrey.pc(_chain_data(), indep_test="does_not_exist")


def test_fisherz_default_matches_explicit():
    """The default path and an explicit ``indep_test='fisherz'`` give the identical graph."""
    x = _chain_data()
    default = andrey.pc(x).structure.to_numpy()
    explicit = andrey.pc(x, indep_test="fisherz").structure.to_numpy()
    assert np.array_equal(default, explicit)


def test_registry_surface(clean_registry):
    """``register`` / ``make`` / ``available`` behave as documented; every shipped name builds."""
    assert set(ci.SHIPPED_INDEP_TESTS) <= set(ci.available_indep_tests())
    assert {"fisherz", "student_t", "normal_scores"} < set(ci.SHIPPED_INDEP_TESTS)
    assert set(ci.DISCRETE_INDEP_TESTS) < set(ci.SHIPPED_INDEP_TESTS)
    for name in ci.SHIPPED_INDEP_TESTS:
        assert 0.0 <= ci.make_indep_test(name, np.round(_chain_data()))(0, 1) <= 1.0
    ci.register_indep_test("delegate", lambda data: ci.FisherZ(data))
    assert "delegate" in ci.available_indep_tests()
    assert isinstance(ci.make_indep_test("fisherz", _chain_data()), ci.FisherZ)
    assert isinstance(ci.make_indep_test("student_t", _chain_data()), ci.StudentT)
    assert isinstance(ci.make_indep_test("normal_scores", _chain_data()), ci.NormalScores)


@pytest.mark.parametrize("indep_test", ["student_t", "normal_scores"])
@pytest.mark.parametrize(
    "facade",
    [
        *("pc", "fci", "gfci", "boss_fci", "grasp_fci", "cdnod"),
        *("iamb", "inter_iamb", "mmpc", "mmhc", "rsmax2"),
    ],
)
def test_a_partial_correlation_test_is_selectable_by_every_constraint_facade(facade, indep_test):
    """Each constraint facade accepts the test by name and returns a valid graph."""
    x = _chain_data()
    if facade == "cdnod":
        out = andrey.cdnod(x, domains(len(x)), indep_test=indep_test)
    else:
        out = getattr(andrey, facade)(x, indep_test=indep_test)
    out.structure.validate()
    assert out.structure.n_nodes == x.shape[1]


@pytest.mark.parametrize("indep_test", ["fisherz", "student_t"])
def test_pc_recovers_a_collider_with_each_test(indep_test):
    """X -> Z <- Y: both tests keep both edges, drop X - Y, and orient into the collider."""
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((2, 1000))
    z = x + y + 0.5 * rng.standard_normal(1000)
    out = andrey.pc(np.column_stack([x, y, z]), indep_test=indep_test)
    assert set(out.structure.oriented_edges(index=True)) == {(0, 2, "directed"), (1, 2, "directed")}
