"""Missing values in ``andrey.data``: the masks of ``andrey.data.missingness`` and ``missing=``.

The checks on how a mask depends on the data draw it over independent standard normal columns, so
an indicator that depends on a column it does not record shows as an association with that
column. They use fixed seeds, large samples, and a level of ``1e-3`` split over the columns tested.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import andrey.data as data
from andrey.data import missingness as m

ALPHA = 1e-3
N = 4000


def _normal(n=N, d=6, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d))


def _associated(values, indicator):
    """The p-value of a difference between the values of the rows where the indicator is 1 and 0."""
    return stats.ttest_ind(values[indicator], values[~indicator], equal_var=False).pvalue


MECHANISMS = [
    pytest.param(m.mcar(0.2), id="mcar"),
    pytest.param(m.mar(0.2), id="mar-logistic"),
    pytest.param(m.mar(0.2, form="quantile"), id="mar-quantile"),
    pytest.param(m.mnar(0.2), id="mnar-logistic"),
    pytest.param(m.mnar(0.2, form="quantile"), id="mnar-quantile"),
    pytest.param(m.self_masking(0.2), id="self_masking-logistic"),
    pytest.param(m.self_masking(0.2, form="quantile"), id="self_masking-quantile"),
]


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_each_column_that_can_go_missing_loses_about_rate_of_its_values(mechanism):
    record = m.draw(_normal(), mechanism, seed=1)
    shares = record.mask[:, sorted(record.parents)].mean(axis=0)
    assert np.allclose(shares, 0.2, atol=0.03), shares
    always = [c for c in range(6) if c not in record.parents]
    assert not record.mask[:, always].any()
    assert record.mechanism == mechanism.name and record.form == mechanism.form


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_each_indicator_depends_on_its_recorded_parents_and_no_other_column(mechanism):
    X = _normal()
    record = m.draw(X, mechanism, seed=2)
    level = ALPHA / (len(record.parents) * X.shape[1])
    for column, parents in record.parents.items():
        for other in range(X.shape[1]):
            p = _associated(X[:, other], record.mask[:, column])
            if other not in parents:
                assert p > level, (column, other)
            elif len(parents) == 1:  # each of several logistic weights can be near 0
                assert p < level, (column, other)


def test_mcar_indicators_have_no_parents_and_are_independent_of_every_column():
    X = _normal()
    record = m.draw(X, m.mcar(0.3, observed=0.5), seed=3)
    assert len(record.parents) == 3 and all(p == [] for p in record.parents.values())
    for column in record.parents:
        for other in range(X.shape[1]):
            assert _associated(X[:, other], record.mask[:, column]) > ALPHA / 18


def test_mar_parents_are_always_observed_and_mnar_parents_can_go_missing():
    X = _normal(d=10)
    mar = m.draw(X, m.mar(0.2, observed=0.3), seed=4).parents
    assert all(set(pa) and not set(pa) & set(mar) for pa in mar.values())
    assert len(mar) == 7
    mnar = m.draw(X, m.mnar(0.2, form="quantile"), seed=4).parents
    assert all(len(pa) == 1 and pa[0] in mnar and pa[0] != c for c, pa in mnar.items())
    selfm = m.draw(X, m.self_masking(0.2), seed=4).parents
    assert all(pa == [c] for c, pa in selfm.items())


def test_a_self_masking_indicator_depends_on_its_own_column():
    X = _normal()
    record = m.draw(X, m.self_masking(0.3, form="quantile"), seed=5)
    for column in record.parents:
        assert _associated(X[:, column], record.mask[:, column]) < ALPHA
        assert X[record.mask[:, column], column].mean() < X[~record.mask[:, column], column].mean()


def test_given_parents_are_the_recorded_ones_and_the_quantile_form_has_its_two_probabilities():
    X = _normal(n=20000)
    record = m.draw(X, m.mar(0.3, form="quantile", parents={3: [0], 4: [1]}), seed=6)
    assert record.parents == {3: [0], 4: [1]}
    q = (0.3 - m.P_LOW) / (m.P_HIGH - m.P_LOW)
    below = X[:, 0] < np.quantile(X[:, 0], q)
    assert record.mask[below, 3].mean() == pytest.approx(m.P_HIGH, abs=0.02)
    assert record.mask[~below, 3].mean() == pytest.approx(m.P_LOW, abs=0.01)


def test_the_same_seed_without_missing_gives_the_complete_data():
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=8, avg_degree=2),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    complete = scm.sample(n=500, seed=7)
    masked = scm.sample(n=500, seed=7, missing=m.mnar(0.2))
    mask = masked.missing.mask
    assert mask.any() and np.array_equal(np.isnan(masked.data), mask)
    assert np.array_equal(masked.data[~mask], complete.data[~mask])
    assert np.array_equal(masked.params.edges, complete.params.edges)
    assert np.array_equal(masked.params.weights, complete.params.weights)
    assert masked.report == complete.report and complete.missing is None
    assert np.array_equal(m.draw(complete.data, m.mnar(0.2), seed=7).mask, mask)
    config = masked.provenance["config"]["missing"]
    assert config["mechanism"] == "mnar" and config["rate"] == 0.2
    assert config["parents"] == {str(c): pa for c, pa in masked.missing.parents.items()}


def test_sample_scm_forwards_missing_and_another_seed_draws_another_mask():
    a = data.sample_scm(d=6, n=300, seed=8, density=2, missing=m.mcar(0.2))
    b = data.sample_scm(d=6, n=300, seed=9, density=2, missing=m.mcar(0.2))
    assert a.missing is not None and not np.array_equal(a.missing.mask, b.missing.mask)


def test_every_column_that_can_go_missing_keeps_two_observed_values():
    for seed in range(20):
        record = m.draw(_normal(n=4, d=3), m.mcar(0.5), seed=seed)
        assert ((~record.mask).sum(axis=0) >= m.MIN_OBSERVED).all()
    with pytest.raises(ValueError, match="no draw in 100"):
        m.draw(_normal(n=2, d=3), m.mcar(0.99), seed=0)


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: m.mcar(0.0), "rate"),
        (lambda: m.mcar(1.0), "rate"),
        (lambda: m.mar(0.95, form="quantile"), "rate"),
        (lambda: m.mar(0.2, observed=0.0), "observed"),
        (lambda: m.mnar(0.2, observed=1.0), "observed"),
        (lambda: m.mar(0.2, form="probit"), "form"),
        (lambda: m.mar(0.2, parents={3: [0], 0: [1]}), "always observed"),
        (lambda: m.mar(0.2, parents={3: []}), "distinct"),
        (lambda: m.mar(0.2, form="quantile", parents={3: [0, 1]}), "one parent"),
        (lambda: m.mnar(0.2, parents={3: [3]}), "another column"),
        (lambda: m.mnar(0.2, parents={3: [0]}), "another column"),
    ],
)
def test_a_mechanism_outside_its_rules_raises(build, match):
    with pytest.raises(ValueError, match=match):
        build()


@pytest.mark.parametrize(
    ("X", "mechanism", "match"),
    [
        (_normal(n=50, d=3), m.mar(0.2, parents={5: [0]}), "the data have 3"),
        (_normal(n=50, d=2), m.mnar(0.2, observed=0.5), "needs 2"),
        (np.full((50, 3), np.nan), m.mcar(0.2), "complete data"),
        (np.ones((50, 3)), m.mar(0.2, parents={2: [0]}), "constant"),
    ],
)
def test_a_draw_the_data_cannot_hold_raises(X, mechanism, match):
    with pytest.raises(ValueError, match=match):
        m.draw(X, mechanism, seed=0)
