"""Background knowledge in the Granger fit: what each regression drops and keeps."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from statsmodels.regression.linear_model import OLS

import andrey
from andrey.temporal.granger import granger_f_test
from tests.recovery.helpers import datagen

LAGS = 2


@pytest.fixture(scope="module")
def series() -> np.ndarray:
    return datagen.generate("var_4v_stable")


def design_of(data: np.ndarray) -> np.ndarray:
    n = len(data)
    return np.hstack([data[LAGS - k : n - k] for k in range(1, LAGS + 1)])


def marks(d: int, *links: tuple[int, int, int]) -> np.ndarray:
    """A ``(LAGS, d, d)`` boolean array with the ``(cause, effect, lag)`` links set."""
    out = np.zeros((LAGS, d, d), dtype=bool)
    for cause, effect, lag in links:
        out[lag - 1, cause, effect] = True
    return out


def test_a_forbidden_regressor_is_dropped_from_its_regression(series):
    """The row is the regression without the forbidden columns, with 0 and p = 1 on them."""
    d = series.shape[1]
    forbidden = marks(d, (1, 0, 1), (2, 0, 2), (3, 2, 1))
    design = design_of(series)
    coeff, p_values = granger_f_test(series, maxlag=LAGS, alpha=1.0, forbidden=forbidden)
    for effect, dropped in ((0, [d * 0 + 1, d * 1 + 2]), (2, [3])):
        keep = np.setdiff1d(np.arange(d * LAGS), dropped)
        fit = OLS(series[LAGS:, effect], np.column_stack([np.ones(len(design)), design[:, keep]]))
        fit = fit.fit()
        np.testing.assert_allclose(coeff[effect, keep], fit.params[1:], rtol=1e-8)
        reference = 2 * stats.t.sf(np.abs(fit.tvalues[1:]), fit.df_resid)
        np.testing.assert_allclose(p_values[effect, keep], reference, rtol=1e-6, atol=1e-300)
        assert not coeff[effect, dropped].any() and (p_values[effect, dropped] == 1).all()


def test_a_regression_that_knowledge_leaves_alone_is_fitted_as_without_knowledge(series):
    d = series.shape[1]
    plain = granger_f_test(series, maxlag=LAGS)
    known = granger_f_test(
        series, maxlag=LAGS, forbidden=marks(d, (1, 0, 1)), required=marks(d, (2, 0, 2))
    )
    for a, b in zip(known, plain):
        np.testing.assert_allclose(a[1:], b[1:], rtol=1e-10, atol=1e-300)
    assert not np.array_equal(known[1][0], plain[1][0])


def test_a_required_link_is_in_the_graph_with_its_regression_coefficient(series):
    d = series.shape[1]
    effect, column = 0, d * 1 + 3  # variable 3 at lag 2 is not a cause of variable 0
    plain_coeff, plain_p = granger_f_test(series, maxlag=LAGS)
    assert plain_coeff[effect, column] == 0 and plain_p[effect, column] > 0.05
    coeff, p_values = granger_f_test(series, maxlag=LAGS, required=marks(d, (3, 0, 2)))
    assert coeff[effect, column] != 0 and p_values[effect, column] == 0
    out = andrey.granger(series, tau_max=LAGS, knowledge=andrey.Knowledge(required=[(3, 0, 2)]))
    assert (3, 0) in {(i, j) for i, j, _ in out.structure.lag(2).oriented_edges()}
    assert out.metadata["p_values"][1][3][0] == 0


def test_every_regressor_forbidden_leaves_an_empty_row(series):
    d = series.shape[1]
    forbidden = np.zeros((LAGS, d, d), dtype=bool)
    forbidden[:, :, 1] = True
    coeff, p_values = granger_f_test(series, maxlag=LAGS, forbidden=forbidden)
    assert not coeff[1].any() and (p_values[1] == 1).all()


def test_the_facade_reads_a_link_as_cause_effect_lag(series):
    """``(cause, effect, lag)``: forbidding x0 -> x1 at lag 1 empties that weight and leaves the
    other variables' regressions alone."""
    plain = andrey.granger(series, tau_max=LAGS, fdr_method="none")
    knowledge = andrey.Knowledge(forbidden=[(0, 1, 1)])
    known = andrey.granger(series, tau_max=LAGS, fdr_method="none", knowledge=knowledge)
    # lag_weights[k - 1][cause, effect]
    assert plain.structure.lag_weights[0][0, 1] != 0 and known.structure.lag_weights[0][0, 1] == 0
    assert known.metadata["p_values"][0][0][1] == 1
    others = [0, 2, 3]
    assert np.array_equal(
        known.structure.lag_weights[:, :, others], plain.structure.lag_weights[:, :, others]
    )
