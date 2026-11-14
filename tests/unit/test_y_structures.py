"""``y_structures``: the claim records, ``variables``, column order, per-column rescaling, reruns,
and the test count."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from andrey import y_structures
from andrey.constraint.blanket import LocalSearch
from andrey.constraint.ystructures import y_structures as engine
from andrey.core.effects import check_effects
from andrey.data.latent import DSepOracle

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")


def selected(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """The paper's fixed graph (Figure 5a): c -> x1, c -> x5 <- x4, x5 -> x6, x3 -> x2, with the
    rows kept when x1 + x2 > 1, a selection both depend on."""
    rng = np.random.default_rng(seed)
    m = 6 * n
    c, x3, x4 = rng.standard_normal((3, m))
    x1 = c + rng.standard_normal(m)
    x2 = x3 + rng.standard_normal(m)
    x5 = c + x4 + rng.standard_normal(m)
    x6 = x5 + rng.standard_normal(m)
    data = pd.DataFrame({"c": c, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5, "x6": x6})
    return data[x1 + x2 > 1].head(n).reset_index(drop=True)


def claims(out, *, ordered: bool = True) -> dict:
    """``{(treatment, outcome): (score, patterns)}``, patterns as a set of pairs."""
    return {
        (e["treatment"], e["outcome"]): (
            e["score"],
            {tuple(p) if ordered else frozenset(p) for p in e["patterns"]},
        )
        for e in out.metadata["effects"]
    }


@pytest.mark.parametrize("extended", [False, True])
def test_each_claim_is_an_effects_record_and_the_graph_has_no_edges(extended):
    data = selected()
    out = y_structures(data, extended=extended)
    records = out.metadata["effects"]
    check_effects(records, list(data.columns))
    assert ("x5", "x6") in claims(out)
    assert not out.structure.to_numpy().any()
    for e in records:
        assert (e["estimand"], e["nonzero"], e["identifiable"]) == ("total", True, True)
        assert (e["adjustment_sets"], e["set_kind"]) == ([[]], "valid")
        assert e["patterns"] and all(len(p) == 2 for p in e["patterns"])
    scores = [e["score"] for e in records]
    assert scores == sorted(scores, reverse=True)
    assert isinstance(out.metadata["n_tests"], int) and out.metadata["n_tests"] >= 21


def test_variables_keep_the_claims_about_their_pairs():
    """``variables`` answers only for pairs of them, and a pair gets the same claim either way."""
    data = selected()
    every = claims(y_structures(data))
    some = claims(y_structures(data, variables=["x6", "x5", "x1"]))
    assert some == {k: v for k, v in every.items() if set(k) <= {"x1", "x5", "x6"}}
    assert some


def test_the_y_structure_lists_each_pattern_once_and_the_extended_one_in_order():
    """The Y-structure's two variables play one role, so a pattern is one pair; the extended
    Y-structure's are ordered, the first separated from the effect by the cause."""
    found = claims(y_structures(selected()))["x5", "x6"][1]
    wider = claims(y_structures(selected(), extended=True))["x5", "x6"][1]
    assert all(v < w for v, w in found)
    assert {frozenset(p) for p in found} <= {frozenset(p) for p in wider}
    assert ("c", "x4") in wider and ("x4", "c") in wider


@pytest.mark.parametrize("extended", [False, True])
def test_the_claims_do_not_depend_on_the_column_order(extended):
    data = selected()
    want = claims(y_structures(data, extended=extended), ordered=extended)
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(data.columns)
        got = claims(y_structures(data[order], extended=extended), ordered=extended)
        assert got.keys() == want.keys()
        for key, (score, patterns) in got.items():
            assert patterns == want[key][1]
            assert score == pytest.approx(want[key][0], rel=1e-9)


def test_the_claims_do_not_depend_on_the_columns_units():
    data = selected()
    want = claims(y_structures(data))
    factors = 10.0 ** np.array([-6, 3, 0, 9, -2, 5, -8])
    got = claims(y_structures(data * factors))
    assert got.keys() == want.keys()
    for key, (score, patterns) in got.items():
        assert patterns == want[key][1]
        assert score == pytest.approx(want[key][0], rel=1e-9)


def same_claims(got: dict, want: dict) -> None:
    assert want
    assert got.keys() == want.keys()
    for key, (score, patterns) in got.items():
        assert patterns == want[key][1]
        assert score == pytest.approx(want[key][0], rel=1e-9)


@pytest.mark.parametrize("extended", [False, True])
def test_a_monotone_transform_of_each_column_keeps_the_claims_under_normal_scores(extended):
    """The rank-based test reads each column only through its ranks, so a strictly increasing
    transform of each column, each its own, changes no claim."""
    data = selected()
    transforms = [np.exp, lambda x: x**3, np.sinh, lambda x: x + x**3, np.arctan, np.cbrt, np.tanh]
    changed = pd.DataFrame({c: f(data[c]) for c, f in zip(data.columns, transforms, strict=True)})
    same_claims(
        claims(y_structures(changed, indep_test="normal_scores", extended=extended)),
        claims(y_structures(data, indep_test="normal_scores", extended=extended)),
    )


@pytest.mark.parametrize("indep_test", ["chisq", "gsq"])
def test_recoding_each_columns_categories_in_order_keeps_the_claims(indep_test):
    """A categorical test reads a column only through which rows share a category, so other whole
    numbers in the same order for each column's categories change no claim."""
    data = selected()
    edges = data.quantile([1 / 3, 2 / 3])
    levels = pd.DataFrame({c: np.digitize(data[c], edges[c]) for c in data.columns})
    codes = [[-4, 1, 7], [10, 20, 30], [0, 5, 6], [-9, -8, 40], [2, 3, 100], [-1, 0, 1], [5, 6, 7]]
    recoded = pd.DataFrame(
        {c: np.array(k)[levels[c]] for c, k in zip(levels.columns, codes, strict=True)}
    )
    same_claims(
        claims(y_structures(recoded, indep_test=indep_test, extended=True)),
        claims(y_structures(levels, indep_test=indep_test, extended=True)),
    )


def test_a_rerun_gives_the_same_result():
    data = selected()
    assert y_structures(data).metadata == y_structures(data).metadata


def test_fewer_than_four_columns_claim_nothing():
    out = y_structures(selected()[["c", "x5", "x6"]])
    assert out.metadata["effects"] == []


@pytest.mark.parametrize(
    ("variables", "error"),
    [
        ("x5", TypeError),
        (["x5"], ValueError),
        (["x5", "x5"], ValueError),
        (["x9", "x5"], ValueError),
    ],
)
def test_variables_are_checked(variables, error):
    with pytest.raises(error):
        y_structures(selected(n=200), variables=variables)


def test_each_test_given_x_runs_once_for_a_pattern_it_can_serve():
    """Under d-separation on v -> x <- w, x -> y: the 6 marginal tests, then given each of the 4
    variables only the pairs a pattern there could use, each once."""
    dag = nx.DiGraph([(0, 2), (1, 2), (2, 3)])
    oracle = DSepOracle(dag)
    search = LocalSearch(oracle, [0, 1, 2, 3], 0.5, trace=True)
    found = engine(search, range(4))
    assert [(c.cause, c.effect, c.patterns) for c in found] == [(2, 3, ((0, 1),))]
    asked = search.counts.issued
    # Given 2: (0, 3) and (1, 3) separated, then (0, 1) connected. Given 3: (0, 2) and (1, 2), the
    # dependent pairs of a variable with an independent partner, neither separated. Given 0 or 1
    # no pair is independent, so no pattern there can use a test.
    assert asked == 6 + 3 + 2


class _Table:
    """A CI test answering from a table of p-values: ``{(a, b, given): p}``, ``0`` otherwise."""

    def __init__(self, table):
        self.table = {(min(a, b), max(a, b), tuple(z)): p for (a, b, z), p in table.items()}

    def __call__(self, x, y, condition_set=()):
        return self.table.get((min(x, y), max(x, y), tuple(sorted(condition_set))), 0.0)


@pytest.mark.parametrize("extended", [False, True])
@pytest.mark.parametrize("p_xy", [0.0, 1.0])
def test_a_pair_is_claimed_only_when_it_tests_dependent(extended, p_xy):
    """v, w = 0, 1 around x = 2 and y = 3 make both patterns; x and y testing independent, which
    no pattern allows, leaves the pair unclaimed."""
    table = {(0, 1, ()): 1.0, (0, 3, (2,)): 1.0, (1, 3, (2,)): 1.0, (2, 3, ()): p_xy}
    search = LocalSearch(_Table(table), [0, 1, 2, 3], 0.01)
    claimed = {(c.cause, c.effect) for c in engine(search, range(4), extended=extended)}
    assert ((2, 3) in claimed) == (p_xy == 0.0)
