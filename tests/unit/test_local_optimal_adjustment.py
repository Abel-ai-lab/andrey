"""Checks of the ``local_optimal_adjustment`` facade and its column view.

The result holds two records, the first with the first of ``variables`` as the treatment, named
by label or position, and the edges at the two variables alone. The answer does not depend on the
column order or the columns' units, a rerun repeats it, and ``n_tests`` counts the tests the fit
evaluated. Arguments that do not name two columns raise. A column view answers each query as the
full test answers the same columns.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.constraint.local_optimal_adjustment import ColumnView
from andrey.core.ci import counting_tests, make_indep_test


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        yield


def _frame(seed: int = 0, n: int = 3000) -> pd.DataFrame:
    """i -> x <- c, c -> y <- p, x -> y, x -> m -> y, and w -> c: the effect of x on y is
    identifiable, with {c, p} the optimal set. Every partial correlation of two adjacent variables
    is at least 0.13 in absolute value, whatever the conditioning set."""
    rng = np.random.default_rng(seed)
    w, i, p = rng.standard_normal((3, n))
    c = 0.5 * w + rng.standard_normal(n)
    x = c + i + rng.standard_normal(n)
    m = 0.5 * x + rng.standard_normal(n)
    y = 0.5 * x + 0.5 * m + 0.5 * c + p + rng.standard_normal(n)
    return pd.DataFrame({"w": w, "i": i, "p": p, "c": c, "x": x, "m": m, "y": y})


def _answer(out) -> list[tuple]:
    return [
        (
            r["treatment"],
            r["outcome"],
            r["nonzero"],
            r["identifiable"],
            sorted(sorted(map(str, s)) for s in r["adjustment_sets"]),
        )
        for r in out.metadata["effects"]
    ]


def test_two_records_the_first_with_the_first_variable_as_treatment():
    df = _frame()
    out = andrey.local_optimal_adjustment(df, variables=["y", "x"])
    first, second = out.metadata["effects"]
    assert (first["treatment"], first["outcome"]) == ("y", "x")
    assert (second["treatment"], second["outcome"]) == ("x", "y")
    assert (second["nonzero"], second["identifiable"]) == (True, True)
    assert second["adjustment_sets"] == [["p", "c"]] and second["set_kind"] == "optimal"
    assert (first["nonzero"], first["identifiable"]) == (False, False)
    assert out.metadata["algorithm"] == "local_optimal_adjustment"
    assert out.metadata["blankets_learned"][:2] in (["x", "y"], ["y", "x"])


def test_positions_name_the_records_without_labels():
    df = _frame()
    out = andrey.local_optimal_adjustment(df.to_numpy(), variables=[4, 6])
    assert out.metadata["effects"][0]["adjustment_sets"] == [[2, 3]]


def test_the_graph_holds_the_edges_at_the_two_variables_alone():
    df = _frame()
    out = andrey.local_optimal_adjustment(df, variables=["x", "y"])
    marks = out.structure.to_numpy()
    x, y = 4, 6
    away = np.ones_like(marks, dtype=bool)
    away[[x, y], :] = away[:, [x, y]] = False
    assert not marks[away].any()
    assert marks[x].any() and marks[y].any()


def test_the_answer_does_not_depend_on_the_column_order():
    df = _frame(1)
    out = andrey.local_optimal_adjustment(df, variables=["x", "y"])
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(df.columns)
        shuffled = andrey.local_optimal_adjustment(df[order], variables=["x", "y"])
        assert _answer(shuffled) == _answer(out)
        assert shuffled.metadata["n_tests"] == out.metadata["n_tests"]


def test_the_answer_does_not_depend_on_the_columns_units():
    df = _frame(2)
    scales = np.random.default_rng(0).uniform(0.01, 100, df.shape[1])
    out = andrey.local_optimal_adjustment(df, variables=["x", "y"])
    rescaled = andrey.local_optimal_adjustment(df * scales, variables=["x", "y"])
    assert _answer(rescaled) == _answer(out)


def test_a_rerun_repeats_the_answer_and_counts_the_tests():
    df = _frame(3)
    with counting_tests() as tests:
        out = andrey.local_optimal_adjustment(df, variables=["x", "y"])
    assert out.metadata["n_tests"] == tests.n > 0
    again = andrey.local_optimal_adjustment(df, variables=["x", "y"])
    assert _answer(again) == _answer(out)
    assert again.metadata["n_tests"] == out.metadata["n_tests"]


@pytest.mark.parametrize(
    ("variables", "error"),
    [
        (None, ValueError),
        (["x"], ValueError),
        (["x", "y", "c"], ValueError),
        (["x", "x"], ValueError),
        (["x", "nope"], ValueError),
        ("xy", TypeError),
    ],
)
def test_variables_must_name_two_columns(variables, error):
    with pytest.raises(error):
        andrey.local_optimal_adjustment(_frame(n=200), variables=variables)


@pytest.mark.parametrize("indep_test", ["fisherz", "gsq"])
def test_a_column_view_answers_as_the_full_test(indep_test):
    rng = np.random.default_rng(0)
    if indep_test == "gsq":
        data = rng.integers(0, 3, (500, 6)).astype(float)
        data[:, 2] = (data[:, 0] + data[:, 1]) % 3
    else:
        data = _frame(n=500).to_numpy()[:, :6]
    full = make_indep_test(indep_test, data)
    columns = [5, 0, 2, 3]
    view = ColumnView(full, columns)
    queries = [(0, 1, ()), (2, 1, (0,)), (3, 2, (0, 1)), (1, 3, (2,))]
    for a, b, s in queries:
        want = full(columns[a], columns[b], [columns[v] for v in s])
        assert view(a, b, s) == want
    if view.statistic_pvalues is not None:
        r, p = view.statistic_pvalues(np.array([0, 2]), np.array([1, 1]), np.array([[3], [0]]))
        want = full.statistic_pvalues(np.array([5, 2]), np.array([0, 0]), np.array([[3], [5]]))
        assert np.array_equal(p, want[1])
