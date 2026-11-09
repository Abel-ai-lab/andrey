"""Recovery by structural EM, ``missing="em"``, on values missing at random or completely at random.

The fixed case is ``x -> w <- y`` with ``y`` missing when ``w`` is low, ``w`` always observed: the
values are missing at random. The complete rows are a sample selected on the collider ``w``, so
the search on them keeps an edge between ``x`` and ``y``, and so does the search on the data with
each missing value replaced by its column's mean; EM learns the collider. ``hc`` is left out of
the fixed case: on the tie-break order of these data its climb stops at the complete graph, on the
complete data's covariance too. On seeded ground truth, 10 variables with 500 rows, the searches
over the BIC and ``notears`` make fewer errors in total by EM than on the complete rows, than with
mean imputation, and than the empty graph. ``dagma``, on 8 variables, makes no more errors by EM
than on the complete rows and fewer than with mean imputation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
from andrey import data
from andrey.data import missingness
from andrey.metrics import shd

COLLIDER = [(0, 2, "directed"), (1, 2, "directed")]
BIC_SEARCHES = ("ges", "xges", "hc", "tabu", "exact_search", "boss", "grasp")


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        warnings.simplefilter("ignore", andrey.SampleSizeWarning)
        yield


def selected_collider(seed: int = 0, n: int = 1000, rate: float = 0.3) -> np.ndarray:
    """``x -> w <- y`` in columns 0, 2, 1, with ``y`` missing below a quantile of ``w``."""
    rng = np.random.default_rng(seed)
    x, y = rng.standard_normal(n), rng.standard_normal(n)
    X = np.column_stack([x, y, x + y + 0.5 * rng.standard_normal(n)])
    mask = missingness.draw(X, missingness.mar(rate, form="quantile", parents={1: [2]}), seed=seed)
    return np.where(mask.mask, np.nan, X)


def complete_rows(X: np.ndarray) -> np.ndarray:
    return X[~np.isnan(X).any(axis=1)]


def mean_imputed(X: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(X), np.nanmean(X, axis=0), X)


@pytest.mark.parametrize("name", ["ges", "xges", "tabu", "exact_search", "boss", "grasp"])
@pytest.mark.parametrize("seed", range(3))
def test_em_learns_the_collider_the_complete_rows_and_mean_imputation_miss(name, seed):
    fit = getattr(andrey, name)
    X = selected_collider(seed)
    assert fit(complete_rows(X)).structure.to_numpy()[0, 1] != 0
    assert fit(mean_imputed(X)).structure.to_numpy()[0, 1] != 0
    assert fit(X, missing="em").structure.oriented_edges() == COLLIDER


@pytest.mark.parametrize("name", ["notears", "dagma"])
def test_the_continuous_searches_learn_the_collider_by_em(name):
    assert getattr(andrey, name)(selected_collider(0), missing="em").structure.oriented_edges() == (
        COLLIDER
    )


def test_a_fit_of_the_fixed_case_records_its_output():
    out = andrey.ges(selected_collider(0), missing="em")
    record = out.metadata["em"]
    assert {k: v for k, v in out.metadata.items() if k not in ("em", "score")} == {
        "algorithm": "ges",
        "score_func": "bic",
        "penalty_discount": 1.0,
        "ebic_gamma": 1.0,
        "tie_break": "rank",
        "missing": "em",
    }
    assert out.metadata["score"] == pytest.approx(-1423.1982633853625, rel=1e-9)
    assert record["iterations"] == 10
    assert record["log_likelihood"] == pytest.approx(
        [
            *(-3600.823643, -3480.586309, -3415.745201, -3381.961972, -3367.441575),
            *(-3361.188721, -3358.426326, -3357.179858, -3356.61037, -3356.348455),
        ],
        abs=1e-5,
    )


def _errors(fit, mechanism, *, d: int = 10, seeds: int = 8) -> np.ndarray:
    """Each seed's SHD of EM, the complete rows, mean imputation, and the empty graph."""
    rows = []
    for seed in range(seeds):
        sample = data.sample_scm(d=d, n=500, density=2.0, seed=seed, missing=mechanism)
        X, truth = np.asarray(sample.data), sample.graph
        rows.append(
            (
                shd(fit(X, missing="em").structure, truth),
                shd(fit(complete_rows(X)).structure, truth),
                shd(fit(mean_imputed(X)).structure, truth),
                int(np.count_nonzero(truth.to_numpy())) // 2,
            )
        )
    return np.array(rows)


@pytest.mark.parametrize("mechanism", ["mar", "mcar"])
@pytest.mark.parametrize("name", BIC_SEARCHES)
def test_em_makes_fewer_errors_than_the_complete_rows_or_mean_imputation(name, mechanism):
    errors = _errors(
        getattr(andrey, name),
        missingness.mar(0.3) if mechanism == "mar" else missingness.mcar(0.2),
    )
    em, rows, mean, empty = errors.sum(axis=0)
    assert em < rows and em < mean and em < empty


def test_notears_by_em_makes_fewer_errors_than_the_complete_rows_or_mean_imputation_under_mar():
    em, rows, mean, empty = _errors(andrey.notears, missingness.mar(0.3)).sum(axis=0)
    assert em < rows and em < mean < empty


def test_dagma_by_em_makes_no_more_errors_than_the_complete_rows_under_mar():
    em, rows, mean, empty = _errors(andrey.dagma, missingness.mar(0.3), d=8, seeds=4).sum(axis=0)
    assert em <= rows and em < mean and em < empty
