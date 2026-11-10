"""Structural EM, ``missing="em"``: the E-step, the parameter fit, and the facades that take it.

The E-step's expected mean and covariance on a fixed dataset and model are recorded from the E-step
of the MissDAG authors' code (Gao et al. 2022), run once outside the tests. Every method that takes
``missing="em"`` gives its complete-data result on complete data, refuses ``NaN`` under ``None``,
and records the handling and the EM record on data with missing values.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey import data
from andrey.core import GraphStructure
from andrey.core.ci_missing import EM_METHODS, EM_SUPPORTED
from andrey.core.score import CopiedColumnsError
from andrey.data import missingness
from andrey.search.em import (
    ITERATIONS,
    expected_moments,
    fit_dag,
    observed_copies,
    structural_em,
)
from andrey.spec import REGISTRY
from tests.unit.methods import SHORT

nan = np.nan
#: The methods that need no optional extra.
CORE = [m for m in EM_METHODS if m != "calm"]


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        warnings.simplefilter("ignore", andrey.DataTypeWarning)
        warnings.simplefilter("ignore", andrey.SampleSizeWarning)
        warnings.simplefilter("ignore", andrey.SearchLimitWarning)  # the short schedule
        yield


def _method(name: str):
    """The facade ``name`` with the short schedule of a slow fit (``SHORT``)."""
    if name == "calm":
        pytest.importorskip("torch")
    return functools.partial(getattr(andrey, name), **SHORT.get(name, {}))


def _chain(n: int = 400, seed: int = 0) -> np.ndarray:
    """A chain 0 -> 2 -> 3 with independent columns 1 and 4."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 5))
    x[:, 2] += x[:, 0]
    x[:, 3] += x[:, 2]
    return x


def _gaps(x: np.ndarray, rate: float, seed: int = 1) -> np.ndarray:
    out = np.array(x, dtype=np.float64)
    out[np.random.default_rng(seed).random(out.shape) < rate] = nan
    return out


def test_the_expected_moments_match_the_authors_e_step_on_a_fixed_case():
    X = np.array(
        [
            [0.5, -1.0, 2.0],
            [1.5, nan, 0.25],
            [nan, 0.75, nan],
            [-0.5, 1.25, nan],
            [nan, nan, nan],
            [2.0, -0.5, 1.0],
        ]
    )
    mean = np.array([0.25, -0.125, 0.5])
    precision = np.array([[2.0, -0.5, 0.25], [-0.5, 1.5, -0.75], [0.25, -0.75, 1.25]])
    expected_mean, cov, _ = expected_moments(X, mean, precision)
    np.testing.assert_allclose(
        expected_mean, [0.692841880342, 0.090277777778, 1.036431623932], atol=1e-10
    )
    np.testing.assert_allclose(
        cov,
        [
            [0.859381096260, -0.299072118083, -0.207353589104],
            [-0.299072118083, 0.841525999078, 0.013022847137],
            [-0.207353589104, 0.013022847137, 0.798928745904],
        ],
        atol=1e-10,
    )


def test_the_log_likelihood_is_the_density_of_each_rows_observed_values():
    rng = np.random.default_rng(3)
    X = _gaps(rng.standard_normal((80, 4)) @ rng.standard_normal((4, 4)), 0.3)
    X[0] = nan
    mean = rng.standard_normal(4)
    A = rng.standard_normal((4, 4))
    precision = A @ A.T + 4 * np.eye(4)
    sigma = np.linalg.inv(precision)
    want = 0.0
    for row in X:
        o = ~np.isnan(row)
        if o.any():
            r, block = row[o] - mean[o], sigma[np.ix_(o, o)]
            want -= 0.5 * (r @ np.linalg.solve(block, r) + o.sum() * np.log(2 * np.pi))
            want -= 0.5 * np.linalg.slogdet(block)[1]
    assert expected_moments(X, mean, precision)[2] == pytest.approx(want, rel=1e-12)


def test_on_complete_data_the_expected_moments_are_the_sample_moments():
    X = _chain(200)
    mean, cov, _ = expected_moments(X, np.zeros(5), np.eye(5))
    np.testing.assert_allclose(mean, X.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(cov, np.cov(X, rowvar=False, bias=True), atol=1e-12)


def test_the_parameter_fit_recovers_a_dag_from_its_covariance():
    B = np.array([[0.0, 0.8, 0.0], [0.0, 0.0, -1.5], [0.0, 0.0, 0.0]])
    omega = np.array([1.0, 0.5, 2.0])
    lower = np.linalg.inv(np.eye(3) - B)
    cov = lower.T @ np.diag(omega) @ lower
    fitted, variances = fit_dag(B != 0, cov)
    np.testing.assert_allclose(fitted, B, atol=1e-12)
    np.testing.assert_allclose(variances, omega, atol=1e-12)


def test_with_a_fixed_graph_each_iteration_raises_the_observed_data_log_likelihood():
    sample = data.sample_scm(d=6, n=300, seed=4, density=2.0, missing=missingness.mcar(0.3))
    truth = sample.graph

    _, record = structural_em(np.asarray(sample.data), lambda cov: (None, truth))
    loglik = record["log_likelihood"]
    assert ITERATIONS == 10
    assert record["iterations"] == 10 == len(loglik)
    assert all(b >= a - 1e-9 for a, b in zip(loglik, loglik[1:], strict=False))
    assert loglik[-1] > loglik[0] + 100.0  # 423 on these data


def test_with_only_the_child_missing_em_reaches_the_complete_case_regression():
    # With x always observed and y missing completely at random, the maximum-likelihood
    # regression of y on x is the least-squares fit on the rows that observe y (Little and Rubin
    # 2019, Section 7.2), the fixed point the refit of each round converges to.
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    X = np.column_stack([x, 0.8 * x + 0.6 * rng.standard_normal(200)])
    X[rng.random(200) < 0.3, 1] = nan
    graph = GraphStructure.from_numpy(np.array([[0, 1], [2, 0]]), kind="dag")  # x -> y
    covs = []
    structural_em(X, lambda cov: (covs.append(cov), graph))
    rows = ~np.isnan(X[:, 1])
    slope, intercept = np.polyfit(X[rows, 0], X[rows, 1], 1)
    residual = X[rows, 1] - slope * X[rows, 0] - intercept
    B, omega = fit_dag(np.array([[False, True], [False, False]]), covs[-1])
    assert B[0, 1] == pytest.approx(slope, rel=1e-4)  # 0.80; 0.58 on the first round's covariance
    assert omega[1] == pytest.approx(residual.var(), rel=1e-4)


@pytest.mark.parametrize("name", EM_METHODS)
def test_complete_data_give_the_complete_data_result(name):
    fit = _method(name)
    x = _chain()
    plain, em = fit(x), fit(x, missing="em")
    assert em == plain
    assert "em" not in em.metadata and "missing" not in em.metadata


@pytest.mark.parametrize("name", EM_METHODS)
def test_a_fit_on_missing_values_records_its_handling_and_repeats(name):
    fit = _method(name)
    x = _gaps(_chain(), 0.1)
    first, again = fit(x, missing="em"), fit(x, missing="em")
    assert first == again
    assert first.metadata["missing"] == "em"
    record = first.metadata["em"]
    assert record["iterations"] == 10 == len(record["log_likelihood"])


@pytest.mark.parametrize("name", EM_METHODS)
def test_without_em_a_missing_value_raises_naming_em(name):
    fit = _method(name)
    with pytest.raises(ValueError, match=rf'^{name} does not accept .*pass missing="em"') as error:
        fit(_gaps(_chain(), 0.1))
    # The message and the schema mark "em" experimental exactly where it is not supported.
    marked = name not in EM_SUPPORTED
    assert ("experimental" in str(error.value)) is marked
    doc = next(p.doc for p in REGISTRY[name].params if p.name == "missing")
    assert ("experimental" in doc) is marked


@pytest.mark.parametrize("name", EM_METHODS)
def test_a_handling_the_method_does_not_list_raises_naming_the_argument(name):
    fit = _method(name)
    with pytest.raises(ValueError, match=rf"^{name}: missing must be one of 'em'"):
        fit(_gaps(_chain(), 0.1), missing="testwise")


@pytest.mark.parametrize("name", CORE)
def test_a_row_that_observes_no_value_changes_no_fit(name):
    fit = _method(name)
    x = _gaps(_chain(), 0.1)
    padded = np.insert(x, [0, 150, 150], nan, axis=0)
    assert fit(padded, missing="em") == fit(x, missing="em")


def test_the_refusal_names_em_only_where_em_would_run():
    x = _gaps(_chain(), 0.1)
    codes = np.round(np.abs(x))
    frame = pd.DataFrame(codes, columns=list("abcde")).astype("category")
    for call in (
        lambda: andrey.ges(codes, score_func="bdeu"),
        lambda: andrey.hc(frame),
        lambda: andrey.ges(x, env=np.repeat([0, 1], 200)),
        lambda: andrey.direct_lingam(x),
    ):
        with pytest.raises(ValueError, match=r"does not accept missing values") as error:
            call()
        assert 'missing="em"' not in str(error.value)


@pytest.mark.parametrize("name", ["ges", "xges", "hc", "tabu", "exact_search", "boss", "grasp"])
def test_em_takes_the_linear_gaussian_bic_only(name):
    x = np.round(np.abs(_gaps(_chain(), 0.1)))
    with pytest.raises(ValueError, match=rf'^{name}: missing="em" .*score_func must be "bic"'):
        getattr(andrey, name)(x, score_func="bdeu", missing="em")


@pytest.mark.parametrize("name", ["ges", "hc", "tabu", "boss", "grasp"])
def test_em_takes_no_environments(name):
    with pytest.raises(ValueError, match=rf'^{name}: missing="em" takes no env'):
        getattr(andrey, name)(_gaps(_chain(), 0.1), env=np.repeat([0, 1], 200), missing="em")


def test_a_copy_among_the_observed_values_raises_as_on_complete_data():
    x = _gaps(_chain(), 0.1)
    x[:, 4] = 3.0 * x[:, 2] - 1.0
    x[np.random.default_rng(5).random(len(x)) < 0.2, 4] = nan  # its own gaps hide it from EM
    assert observed_copies(x) == ((2, 4),)
    assert observed_copies(_gaps(_chain(), 0.1)) == ()
    for name in ("ges", "xges", "exact_search", "boss"):
        with pytest.raises(CopiedColumnsError, match=r"\b4 repeats 2\b"):
            getattr(andrey, name)(x, missing="em")


def _labeled_marks(out) -> dict:
    m, labels = out.structure.to_numpy(), out.structure.labels
    return {(a, b): int(m[i, j]) for i, a in enumerate(labels) for j, b in enumerate(labels)}


@pytest.mark.parametrize("name", EM_METHODS)
def test_a_column_order_or_a_change_of_units_changes_no_graph(name):
    fit = _method(name)
    sample = data.sample_scm(d=6, n=300, seed=2, density=2.0, missing=missingness.mcar(0.15))
    frame = pd.DataFrame(np.asarray(sample.data), columns=list("abcdef"))
    want = _labeled_marks(fit(frame, missing="em"))
    order = list(np.random.default_rng(0).permutation(list(frame.columns)))
    assert _labeled_marks(fit(frame[order], missing="em")) == want
    if name not in ("notears", "dagma", "calm"):  # these read units
        assert _labeled_marks(fit(frame * [3.0, 1, 1, 0.5, 1, 2] + 7.0, missing="em")) == want


def test_the_methods_that_take_em_list_it_as_their_only_handling():
    for name in EM_METHODS:
        param = next(p for p in REGISTRY[name].params if p.name == "missing")
        assert param.choices == ("em",) and param.default is None
        assert {"missing", "em"} <= set(REGISTRY[name].output.metadata)
