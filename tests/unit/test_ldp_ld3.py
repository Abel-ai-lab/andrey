"""``ldp`` and ``ld3``: the answer's shape, the arguments, and the properties a fit must keep (the
column order, the columns' units, a rerun, one test per query)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from andrey import ld3, ldp
from andrey.constraint import partition
from andrey.constraint.blanket import local_search
from andrey.core.ci import counting_tests
from andrey.core.effects import check_effects

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")

METHODS = (ldp, ld3)


def frame(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """A confounder ``c``, an instrument ``i``, a cause of the outcome ``p``, a mediator ``m``, a
    child of the treatment ``k``, and an unrelated ``u``, around ``x -> y``."""
    rng = np.random.default_rng(seed)
    c, i, p, u = rng.standard_normal((4, n))
    x = c + i + rng.standard_normal(n)
    m = x + rng.standard_normal(n)
    k = x + rng.standard_normal(n)
    y = x + m + c + p + rng.standard_normal(n)
    return pd.DataFrame({"c": c, "i": i, "p": p, "u": u, "x": x, "m": m, "k": k, "y": y})


def answer(out) -> tuple:
    """The record and the labels' members as sets, so the column order does not show."""
    record = out.metadata["effects"][0]
    sets = [frozenset(s) for s in record["adjustment_sets"]]
    return (
        record["nonzero"],
        record["identifiable"],
        frozenset(sets),
        record["set_kind"],
        {k: frozenset(v) for k, v in out.metadata["partition"].items()},
        frozenset(out.structure.oriented_edges()),
    )


@pytest.mark.parametrize("method", METHODS)
def test_one_record_for_the_pair_and_every_column_labeled(method):
    df = frame()
    out = method(df, treatment="x", outcome="y")
    records = out.metadata["effects"]
    check_effects(records, list(df.columns))
    assert [(r["treatment"], r["outcome"]) for r in records] == [("x", "y")]
    labeled = set().union(*out.metadata["partition"].values())
    assert labeled == set(df.columns) - {"x", "y"}
    assert isinstance(out.metadata["n_tests"], int) and out.metadata["n_tests"] > 0


def test_ldp_returns_the_confounder_and_no_edges():
    out = ldp(frame(), treatment="x", outcome="y")
    record = out.metadata["effects"][0]
    assert record["identifiable"] is True and record["set_kind"] == "valid"
    assert record["adjustment_sets"] == [["c"]]
    assert out.metadata["partition"]["Z5"] == ["i"]
    assert out.metadata["partition"]["Z4"] == ["p"]
    assert out.metadata["partition"]["Z8"] == ["u"]
    assert out.structure.oriented_edges() == []


def test_ld3_returns_the_parents_as_edges_into_the_outcome():
    out = ld3(frame(), treatment="x", outcome="y")
    record = out.metadata["effects"][0]
    assert record["nonzero"] is True and record["identifiable"] is None
    assert record["adjustment_sets"] == [["c", "p", "m"]]
    edges = {(a, b, kind) for a, b, kind in out.structure.oriented_edges()}
    assert edges == {(v, "y", "directed") for v in ("c", "p", "x", "m")}
    assert out.metadata["partition"]["Z5,7"] == ["k"]


def test_ldp_gives_no_answer_without_a_confounder():
    rng = np.random.default_rng(1)
    i, p = rng.standard_normal((2, 3000))
    x = i + rng.standard_normal(3000)
    y = x + p + rng.standard_normal(3000)
    record = ldp(np.column_stack([i, p, x, y]), treatment=2, outcome=3).metadata["effects"][0]
    assert record["identifiable"] is None
    assert record["adjustment_sets"] == [] and record["set_kind"] is None


@pytest.mark.parametrize("method", METHODS)
def test_positions_name_the_columns_of_an_array(method):
    df = frame()
    out = method(df.to_numpy(), treatment=4, outcome=7)
    record = out.metadata["effects"][0]
    assert (record["treatment"], record["outcome"]) == (4, 7)
    named = method(df, treatment="x", outcome="y").metadata["effects"][0]
    columns = list(df.columns)
    assert [[columns[j] for j in s] for s in record["adjustment_sets"]] == named["adjustment_sets"]


@pytest.mark.parametrize("method", METHODS)
def test_the_answer_does_not_depend_on_the_column_order(method):
    df = frame()
    base = answer(method(df, treatment="x", outcome="y"))
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(df.columns)
        assert answer(method(df[order], treatment="x", outcome="y")) == base


@pytest.mark.parametrize("method", METHODS)
def test_the_answer_does_not_depend_on_the_units(method):
    df = frame()
    factors = np.geomspace(1e-6, 1e6, df.shape[1])
    base = method(df, treatment="x", outcome="y").metadata
    scaled = method(df * factors + 3.0, treatment="x", outcome="y").metadata
    assert scaled["effects"] == base["effects"]
    assert scaled["partition"] == base["partition"]


@pytest.mark.parametrize("method", METHODS)
def test_a_rerun_repeats_the_fit(method):
    df = frame(seed=2)
    first = method(df, treatment="x", outcome="y").metadata
    assert method(df, treatment="x", outcome="y").metadata == first


@pytest.mark.parametrize("method", METHODS)
def test_the_pair_is_required_and_two_columns(method):
    df = frame(n=200)
    with pytest.raises(ValueError, match="pass treatment and outcome"):
        method(df, treatment="x")
    with pytest.raises(ValueError, match="same column"):
        method(df, treatment="x", outcome=4)
    with pytest.raises(ValueError):
        method(df, treatment="x", outcome="nope")


@pytest.mark.parametrize("engine", ("ldp", "ld3"))
def test_each_query_is_tested_once(engine):
    X = frame().to_numpy()
    with counting_tests() as tests:
        search = local_search(X, alpha=0.005, indep_test="fisherz", ranked=False, trace=True)
        getattr(partition, engine)(search, 4, 7, [0, 1, 2, 3, 5, 6])
    assert search.counts.queries is not None
    assert tests.n == search.counts.issued == len(search.counts.queries)


def test_missing_values_are_deleted_test_by_test():
    df = frame()
    df.iloc[::7, 0] = np.nan
    out = ldp(df, treatment="x", outcome="y")
    assert out.metadata["missing"] == "testwise"
    assert out.metadata["rows_tested"]["min"] < len(df)
