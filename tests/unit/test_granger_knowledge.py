"""Background knowledge in the lasso Granger fit: what each regression drops and keeps."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LassoCV, LinearRegression
from threadpoolctl import threadpool_limits

import andrey
from andrey.temporal.granger import granger_lasso
from tests.recovery.helpers import datagen

LAGS = 2
FOLDS = 5


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
    """The row is ``LassoCV`` on the design without the forbidden columns, and 0 on them."""
    d = series.shape[1]
    forbidden = marks(d, (1, 0, 1), (2, 0, 2), (3, 2, 1))
    design = design_of(series)
    with threadpool_limits(limits=1):
        fitted = granger_lasso(series, maxlag=LAGS, forbidden=forbidden)
        for effect, dropped in ((0, [d * 0 + 1, d * 1 + 2]), (2, [3])):
            keep = np.setdiff1d(np.arange(d * LAGS), dropped)
            expected = np.zeros(d * LAGS)
            expected[keep] = LassoCV(cv=FOLDS).fit(design[:, keep], series[LAGS:, effect]).coef_
            assert np.array_equal(fitted[effect], expected)
            assert not fitted[effect, dropped].any()


def test_a_regression_that_knowledge_leaves_alone_is_fitted_as_without_knowledge(series):
    d = series.shape[1]
    plain = granger_lasso(series, maxlag=LAGS)
    known = granger_lasso(
        series, maxlag=LAGS, forbidden=marks(d, (1, 0, 1)), required=marks(d, (2, 0, 2))
    )
    assert np.array_equal(known[1:], plain[1:])
    assert not np.array_equal(known[0], plain[0])


def test_a_required_regressor_is_not_penalized(series):
    """The lasso runs on what the required columns leave of the target and of the other columns,
    and the required weight is the least-squares weight of what the lasso's fit leaves."""
    d = series.shape[1]
    effect, required = 0, [d * 1 + 3]  # variable 3 at lag 2 is not a cause of variable 0
    design, y = design_of(series), series[LAGS:, effect]
    free = np.setdiff1d(np.arange(d * LAGS), required)
    with threadpool_limits(limits=1):
        fitted = granger_lasso(series, maxlag=LAGS, required=marks(d, (3, 0, 2)))[effect]
        both = np.column_stack([y, design[:, free]])
        partial = LinearRegression().fit(design[:, required], both)
        residuals = both - partial.predict(design[:, required])
        lasso = LassoCV(cv=FOLDS).fit(residuals[:, 1:], residuals[:, 0]).coef_
        weight = LinearRegression().fit(design[:, required], y - design[:, free] @ lasso).coef_
    np.testing.assert_allclose(fitted[free], lasso, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(fitted[required], weight, rtol=1e-8)
    assert fitted[required][0] != 0


def test_every_regressor_forbidden_leaves_an_empty_row(series):
    d = series.shape[1]
    forbidden = np.zeros((LAGS, d, d), dtype=bool)
    forbidden[:, :, 1] = True
    assert not granger_lasso(series, maxlag=LAGS, forbidden=forbidden)[1].any()


def test_the_facade_reads_a_link_as_cause_effect_lag(series):
    """``(cause, effect, lag)``: forbidding x1 -> x0 at lag 1 empties that weight only."""
    plain = andrey.granger(series, tau_max=LAGS).structure.lag_weights
    knowledge = andrey.Knowledge(forbidden=[(1, 0, 1)])
    known = andrey.granger(series, tau_max=LAGS, knowledge=knowledge).structure.lag_weights
    # lag_weights[k - 1][cause, effect]
    assert plain[0][1, 0] != 0 and known[0][1, 0] == 0
    assert np.array_equal(known[:, :, 1:], plain[:, :, 1:])
