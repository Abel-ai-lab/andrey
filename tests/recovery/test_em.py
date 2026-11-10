"""Recovery by structural EM, ``missing="em"``, on values missing at random or completely at random.

The fixed case is ``x -> w <- y`` with ``y`` missing when ``w`` is low, ``w`` always observed: the
values are missing at random. The complete rows are a sample selected on the collider ``w``, so the
search on them keeps an edge between ``x`` and ``y``, and so does the search on the data with each
missing value replaced by its column's mean; EM learns the collider. ``hc`` is left out of the fixed
case: on the tie-break order of these data its climb stops at the complete graph, on the complete
data's covariance too. A second fixed case, 10 variables of mean degree 4 with 30% of the values
missing completely at random, shows what the parameter steps between searches are for (Friedman
1997): with them ``ges`` keeps no extra edge and settles; with one parameter fit per round it keeps
7 and its graph still changes in the last round. On seeded ground truth, 10 variables with 500 rows,
EM gives the complete-data fit when nothing is missing, and with values missing the searches over
the BIC make fewer errors in total by EM than on the complete rows, than with mean imputation, and
than the empty graph; ``boss`` with values missing completely at random makes as many as on the
complete rows. ``notears`` under values missing at random makes fewer by EM than on the complete
rows and than the empty graph, and no more than mean imputation makes plus the spread of the
difference, which a change of 1e-12 in the data already moves: EM and mean imputation are not
separable on these eight datasets. ``dagma``, on 8 variables, makes no more errors by EM than on
the complete rows and fewer than with mean imputation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
import andrey.search.em as em_module
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


def test_calm_learns_the_collider_by_em():
    pytest.importorskip("torch")
    assert andrey.calm(selected_collider(0), missing="em").structure.oriented_edges() == COLLIDER


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
    assert out.metadata["score"] == pytest.approx(-1425.6637157634295, rel=1e-9)
    assert record["iterations"] == 10
    assert record["parameter_steps"] == 5
    assert record["log_likelihood"] == pytest.approx(
        [
            *(-3600.823643, -3358.588938, -3356.147345, -3356.123735, -3356.123501),
            *(-3356.123499, -3356.123499, -3356.123499, -3356.123499, -3356.123499),
        ],
        abs=1e-5,
    )
    assert record["graph_changed"] == [True, True, *([False] * 8)]


def _extra_edges(structure, truth) -> int:
    """The adjacencies of ``structure`` that ``truth`` does not have."""
    found, true = structure.to_numpy() != 0, truth.to_numpy() != 0
    return int(np.triu((found | found.T) & ~(true | true.T), 1).sum())


def test_parameter_steps_keep_out_the_extra_edges_that_one_fit_per_round_leaves(monkeypatch):
    sample = data.sample_scm(d=10, n=1000, seed=1, density=4.0, missing=missingness.mcar(0.3))
    X, truth = np.asarray(sample.data), sample.graph
    alternating = andrey.ges(X, missing="em")
    monkeypatch.setattr(em_module, "PARAMETER_STEPS", 0)
    single = andrey.ges(X, missing="em")
    assert _extra_edges(alternating.structure, truth) == 0
    assert alternating.metadata["em"]["graph_changed"] == [True] * 4 + [False] * 6
    assert _extra_edges(single.structure, truth) == 7
    assert all(single.metadata["em"]["graph_changed"])


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


@pytest.mark.parametrize("name", [*BIC_SEARCHES, "notears", "dagma"])
def test_with_nothing_missing_em_gives_the_complete_data_fit(name):
    fit = getattr(andrey, name)
    for seed in range(2):
        sample = data.sample_scm(d=10, n=500, density=2.0, seed=seed)
        X = np.asarray(sample.data)
        assert fit(X, missing="em") == fit(X)


@pytest.mark.parametrize("mechanism", ["mar", "mcar"])
@pytest.mark.parametrize("name", BIC_SEARCHES)
def test_em_makes_fewer_errors_than_the_complete_rows_or_mean_imputation(name, mechanism):
    errors = _errors(
        getattr(andrey, name),
        missingness.mar(0.3) if mechanism == "mar" else missingness.mcar(0.2),
    )
    em, rows, mean, empty = errors.sum(axis=0)
    # boss under MCAR: 40 errors by EM and 40 on the complete rows.
    assert em <= rows if (name, mechanism) == ("boss", "mcar") else em < rows
    assert em < mean and em < empty


def test_notears_by_em_beats_the_complete_rows_and_is_within_spread_of_mean_imputation_under_mar():
    em, rows, mean, empty = _errors(andrey.notears, missingness.mar(0.3)).sum(axis=0)
    # NOTEARS's graph moves with the last bits of its input: a relative change of 1e-12 in the data
    # changed 16 of 80 fits by EM and 13 of 80 on the complete rows, none with mean imputation. Over
    # ten such draws EM's sum ran from 59 to 63 (61 undisturbed, 64 on another machine), the
    # complete rows' from 67 to 69; mean imputation's stayed 62 and the empty graph's is 80. EM's
    # lead over the complete rows ran from 4 to 9 errors; EM minus mean imputation, from -3 to 2, so
    # the two are not separable here, and EM may make up to 4 more.
    assert em < rows and em < empty and em <= mean + 4 and mean < empty


def test_dagma_by_em_makes_no_more_errors_than_the_complete_rows_under_mar():
    em, rows, mean, empty = _errors(andrey.dagma, missingness.mar(0.3), d=8, seeds=4).sum(axis=0)
    assert em <= rows and em < mean and em < empty
