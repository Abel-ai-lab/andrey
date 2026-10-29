"""PCMCI, PCMCI+, and LPCMCI with the categorical, mixed-data, and nonlinear tests: small VARs with
planted lagged links, each of a kind the test was built for."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.core import ARROW, ci
from tests.recovery.helpers.ci_tests import PValueOnly

METHODS = ("pcmci", "pcmciplus", "lpcmci")


def _links(out: andrey.StructureOutput) -> set[tuple[int, int, int]]:
    """Every lagged ``(cause, effect, lag)`` link: an arrowhead at the effect, or for LPCMCI any
    link of its partial ancestral graph, whose lagged ends can stay circles."""
    if "links" in out.metadata:
        return {
            (i, j, lag)
            for lag, block in enumerate(out.metadata["links"])
            for i, row in enumerate(block)
            for j, mark in enumerate(row)
            if lag > 0 and mark
        }
    return {
        (int(i), int(j), lag)
        for lag in out.structure.lags
        if lag > 0
        for j, i in np.argwhere(out.structure.lag(lag).to_numpy() == ARROW)
    }


def _fit(method: str, data, **kwargs) -> andrey.StructureOutput:
    """``method`` at one lag; PCMCI at the 0.01 level PCMCI+ takes by default."""
    level = {"alpha": 0.01} if method == "pcmci" else {}
    return getattr(andrey, method)(data, tau_max=1, **level, **kwargs)


def _nonlinear_var(n: int = 300) -> np.ndarray:
    """``x0 -> x1`` through a square and ``x1 -> x2`` through an absolute value, both at lag 1, with
    ``x0`` autoregressive: links whose correlation is near zero."""
    rng = np.random.default_rng(0)
    x = np.zeros((n + 100, 3))
    e = rng.standard_normal((n + 100, 3))
    for t in range(1, n + 100):
        x[t, 0] = 0.5 * x[t - 1, 0] + e[t, 0]
        x[t, 1] = 0.9 * (x[t - 1, 0] ** 2 - 1.3) + 0.5 * e[t, 1]
        x[t, 2] = 1.2 * (np.abs(x[t - 1, 1]) - 0.9) + 0.5 * e[t, 2]
    return x[100:]


NONLINEAR_TRUTH = {(0, 0, 1), (0, 1, 1), (1, 2, 1)}


@pytest.mark.parametrize("indep_test", ["cmiknn", "gpdc", "kci", "rcot"])
def test_pcmci_with_a_nonlinear_test_recovers_nonlinear_links(indep_test):
    out = _fit("pcmci", _nonlinear_var(), indep_test=indep_test)
    assert _links(out) == NONLINEAR_TRUTH
    assert ("seed" in out.metadata) == (indep_test in ci.STOCHASTIC_INDEP_TESTS)


@pytest.mark.parametrize("method", ["pcmciplus", "lpcmci"])
def test_the_same_step_methods_take_a_nonlinear_test(method):
    assert _links(_fit(method, _nonlinear_var(), indep_test="rcot")) == NONLINEAR_TRUTH


def _discrete_var(seed: int, n: int = 500) -> pd.DataFrame:
    """Three categorical series of three categories: ``a`` repeats its last value with
    probability 0.7, ``b`` copies ``a`` and ``c`` shifts ``b`` one step later, each with
    probability 0.8."""
    rng = np.random.default_rng(seed)
    c = np.zeros((n, 3), dtype=int)
    for t in range(1, n):
        c[t, 0] = rng.integers(0, 3) if rng.random() < 0.3 else c[t - 1, 0]
        c[t, 1] = c[t - 1, 0] if rng.random() < 0.8 else rng.integers(0, 3)
        c[t, 2] = (c[t - 1, 1] + 1) % 3 if rng.random() < 0.8 else rng.integers(0, 3)
    return pd.DataFrame(c, columns=["a", "b", "c"]).astype("category")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("indep_test", [None, "gsq", "chisq"])
def test_a_categorical_test_recovers_a_discrete_var(method, indep_test):
    """A frame of pandas category columns declares every column categorical, so ``None`` runs
    the G-square test."""
    data = _discrete_var(1)
    out = _fit(method, data, indep_test=indep_test)
    assert _links(out) == {(0, 0, 1), (0, 1, 1), (1, 2, 1)}
    if indep_test is None:
        assert out.metadata["p_values"] == _fit(method, data, indep_test="gsq").metadata["p_values"]


def _mixed_var(seed: int, n: int = 500) -> pd.DataFrame:
    """A categorical series ``a`` that shifts the mean of the continuous ``b`` a step later, which
    drives the continuous ``c`` a step later; ``b`` and ``c`` depend on their own past."""
    rng = np.random.default_rng(seed)
    a, b, c = np.zeros(n, dtype=int), np.zeros(n), np.zeros(n)
    for t in range(1, n):
        a[t] = rng.integers(0, 3) if rng.random() < 0.3 else a[t - 1]
        b[t] = 0.3 * b[t - 1] + 1.5 * (a[t - 1] - 1) + rng.standard_normal()
        c[t] = 0.3 * c[t - 1] + 0.8 * b[t - 1] + rng.standard_normal()
    return pd.DataFrame({"a": pd.Categorical(a), "b": b, "c": c})


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("indep_test", [None, "cond_gauss", "degen_gauss"])
def test_a_mixed_data_test_recovers_a_mixed_var(method, indep_test):
    """The category column alone is declared, so ``None`` runs the conditional Gaussian test."""
    out = _fit(method, _mixed_var(0), indep_test=indep_test)
    assert _links(out) == {(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 2, 1), (2, 2, 1)}
    assert out.metadata["categorical"] == ["a"]


@pytest.mark.parametrize("method", METHODS)
def test_a_categorical_test_deletes_rows_per_test(method):
    """Missing values completely at random: each test reads the rows its lagged columns observe."""
    data = _discrete_var(1).astype(float)
    data[np.random.default_rng(2).random(data.shape) < 0.05] = np.nan
    out = _fit(method, data.astype("category"), indep_test="gsq")
    assert _links(out) == {(0, 0, 1), (0, 1, 1), (1, 2, 1)}
    assert out.metadata["rows_tested"]["min"] < len(data) - 2


@pytest.mark.parametrize("method", METHODS)
def test_a_randomized_test_moves_with_the_seed_only(method):
    data = _nonlinear_var(150)
    first, again, other = (_fit(method, data, indep_test="rcot", seed=s) for s in (3, 3, 4))
    assert first.metadata["p_values"] == again.metadata["p_values"]
    assert first.metadata["p_values"] != other.metadata["p_values"]
    assert (first.metadata["seed"], other.metadata["seed"]) == (3, 4)


def _linear_var(n: int = 400) -> np.ndarray:
    """A VAR(1) with the links 0 -> 0, 0 -> 1, and 1 -> 2 at lag 1."""
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((n, 3))
    x = np.zeros((n, 3))
    for t in range(1, n):
        x[t] = noise[t] + np.array([0.5 * x[t - 1, 0], 0.6 * x[t - 1, 0], 0.6 * x[t - 1, 1]])
    return x


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("method", METHODS)
def test_a_registered_p_value_only_test_gives_the_graph_of_fisherz(method, missing, monkeypatch):
    """A registered test with the single-query call alone runs the family, under test-wise deletion
    too, and ranks candidates by its p-value: the p-values of Fisher's z-test give its graph."""
    monkeypatch.setitem(ci._CI_REGISTRY, "p_value_only", PValueOnly)
    data = _linear_var()
    if missing:
        data[np.random.default_rng(1).random(data.shape) < 0.05] = np.nan
    expected = _fit(method, data)
    out = _fit(method, data, indep_test="p_value_only")
    assert {(0, 0, 1), (0, 1, 1), (1, 2, 1)} <= _links(out)
    assert _links(out) == _links(expected)
    assert [out.structure.lag(k).to_numpy().tolist() for k in out.structure.lags] == [
        expected.structure.lag(k).to_numpy().tolist() for k in expected.structure.lags
    ]
    if missing:
        assert out.metadata["rows_tested"]["min"] < len(data) - 2
