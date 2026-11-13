"""``covariate_rules``: the answer's shape, the arguments, and the properties a fit must keep (the
column order, the columns' units, a rerun, one test per query, the conditioning-set cap)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from andrey import covariate_rules
from andrey.constraint.blanket import local_search
from andrey.constraint.covariates import covariate_rules as engine
from andrey.core.ci import counting_tests
from andrey.core.effects import check_effects

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")


def frame(n: int = 3000, seed: int = 0, effect: float = 1.0) -> pd.DataFrame:
    """Pretreatment covariates around ``x -> y``: ``a`` causes the treatment only, ``c`` both, and
    ``b`` the outcome only; ``u`` is unrelated."""
    rng = np.random.default_rng(seed)
    a, b, c, u = rng.standard_normal((4, n))
    x = a + c + rng.standard_normal(n)
    y = effect * x + c + b + rng.standard_normal(n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "u": u, "x": x, "y": y})


def answer(out) -> tuple:
    """The record, with its sets as sets, and the counts, so the column order does not show."""
    record = out.metadata["effects"][0]
    return (
        record["nonzero"],
        record["identifiable"],
        frozenset(frozenset(z) for z in record["adjustment_sets"]),
        record["set_kind"],
        out.metadata["rule_counts"],
        out.metadata["n_tests"],
    )


def test_an_effect_comes_with_valid_sets_and_no_edges():
    out = covariate_rules(frame(), treatment="x", outcome="y")
    check_effects(out.metadata["effects"], columns=list(frame().columns))
    (record,) = out.metadata["effects"]
    assert (record["treatment"], record["outcome"], record["estimand"]) == ("x", "y", "total")
    assert record["nonzero"] is True and record["identifiable"] is True
    assert record["set_kind"] == "valid"
    # w = a: a and y depend given Z and are independent given Z and x exactly when c is in Z.
    assert record["adjustment_sets"] and all("c" in z for z in record["adjustment_sets"])
    assert out.structure.oriented_edges() == []
    assert out.metadata["rule_counts"]["R1"] > 0 and out.metadata["rule_counts"]["R2"] == 0


def test_no_effect_is_found_and_has_no_set():
    out = covariate_rules(frame(effect=0.0), treatment="x", outcome="y")
    (record,) = out.metadata["effects"]
    assert record["nonzero"] is False and record["identifiable"] is None
    assert record["adjustment_sets"] == [] and record["set_kind"] is None


def test_undecided_when_no_independence_tells():
    # A confounder alone: no independence among the three, as in Figure 1 (a).
    df = frame()[["c", "x", "y"]]
    (record,) = covariate_rules(df, treatment="x", outcome="y").metadata["effects"]
    assert record["nonzero"] is None and record["adjustment_sets"] == []


def test_counts_cover_every_candidate():
    out = covariate_rules(frame(), treatment="x", outcome="y")
    counts = out.metadata["rule_counts"]
    m = 4  # covariates
    assert counts["R1_checked"] == m * 2 ** (m - 1)
    assert counts["R2_checked"] == 2**m + m * 2 ** (m - 1)
    assert out.metadata["n_tests"] == 2 ** (m - 1) * (3 * m + 2)


def test_positions_name_the_columns_of_an_array():
    out = covariate_rules(frame().to_numpy(), treatment=4, outcome=5)
    (record,) = out.metadata["effects"]
    assert (record["treatment"], record["outcome"]) == (4, 5)
    assert all(2 in z for z in record["adjustment_sets"])


def test_the_answer_does_not_depend_on_the_column_order():
    df = frame()
    base = answer(covariate_rules(df, treatment="x", outcome="y"))
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(df.columns)
        assert answer(covariate_rules(df[order], treatment="x", outcome="y")) == base


def test_the_answer_does_not_depend_on_the_units():
    df = frame()
    factors = np.geomspace(1e-6, 1e6, df.shape[1])
    base = covariate_rules(df, treatment="x", outcome="y").metadata
    scaled = covariate_rules(df * factors + 3.0, treatment="x", outcome="y").metadata
    assert scaled["effects"] == base["effects"]
    assert scaled["rule_counts"] == base["rule_counts"]


def test_a_rerun_repeats_the_fit():
    df = frame(seed=2)
    first = covariate_rules(df, treatment="x", outcome="y").metadata
    assert covariate_rules(df, treatment="x", outcome="y").metadata == first


def test_the_pair_is_required_and_two_columns():
    df = frame(n=200)
    with pytest.raises(ValueError, match="pass treatment and outcome"):
        covariate_rules(df, treatment="x")
    with pytest.raises(ValueError, match="same column"):
        covariate_rules(df, treatment="x", outcome=4)
    with pytest.raises(ValueError):
        covariate_rules(df, treatment="x", outcome="nope")


def test_each_query_is_tested_once():
    X = frame().to_numpy()
    with counting_tests() as tests:
        search = local_search(X, alpha=0.05, indep_test="fisherz", ranked=False, trace=True)
        engine(search, 4, 5, [0, 1, 2, 3])
    assert search.counts.queries is not None
    assert tests.n == search.counts.issued == len(search.counts.queries)


@pytest.mark.parametrize("cap", [0, 1, 2, 3])
def test_a_cap_bounds_r1_one_below_r2(cap):
    X = frame().to_numpy()
    search = local_search(X, alpha=0.05, indep_test="fisherz", ranked=False)
    result = engine(search, 4, 5, [0, 1, 2, 3], max_depth=cap)
    # R1 checks (w, Z) with |Z| <= cap - 1, R2 (i) sets with |Z| <= cap, R2 (ii) (w, Z), |Z| <= cap.
    sizes = range(cap + 1)
    r1 = sum(4 * len(_subsets(3, k)) for k in range(cap))
    r2 = sum(len(_subsets(4, k)) for k in sizes) + sum(4 * len(_subsets(3, k)) for k in sizes)
    assert (result.counts["R1_checked"], result.counts["R2_checked"]) == (r1, r2)
    assert all(len(z) < cap for z in result.sets)


def _subsets(n: int, k: int) -> list:
    from itertools import combinations

    return list(combinations(range(n), k))


def test_no_covariates_leaves_only_the_marginal_test():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    out = covariate_rules(np.column_stack([x, rng.standard_normal(500)]), treatment=0, outcome=1)
    assert out.metadata["effects"][0]["nonzero"] is False  # x and y independent: R2 (i)
    assert out.metadata["n_tests"] == 1


def test_missing_values_are_deleted_test_by_test():
    df = frame()
    df.iloc[::7, 0] = np.nan
    out = covariate_rules(df, treatment="x", outcome="y")
    assert out.metadata["missing"] == "testwise"
    assert out.metadata["rows_tested"]["min"] < len(df)
