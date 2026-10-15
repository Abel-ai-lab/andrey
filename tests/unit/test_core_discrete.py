"""Discrete tests and scores: reference values, invariances, and the search-facing surfaces.

The reference values were computed on the fixed data below by causal-learn 0.1.4 (``CIT`` with
``"chisq"`` and ``"gsq"``; ``local_score_BDeu``, structure prior included, times ``-2``) and by
pgmpy 1.0 (``BIC.local_score`` times ``-2``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.api._adapt import data_matrix, discrete_matrix, fit_matrix
from andrey.core.ci import make_indep_test
from andrey.core.discrete import BDeuScore, ChiSquare, DiscreteBICScore, GSquare, encode_discrete
from andrey.core.score import make_discrete_score
from andrey.core.score_delta import ScoreDifference, make_score_engine


def _fixed_data() -> np.ndarray:
    """Four categorical columns with 2, 3, 2, and 4 levels: a -> b -> d <- c, and b -> c."""
    rng = np.random.default_rng(7)
    n = 200
    a = rng.integers(0, 2, n)
    b = (a + rng.integers(0, 3, n) * (rng.random(n) < 0.4)) % 3
    c = rng.integers(0, 2, n) ^ (b == 2)
    d = (b + 2 * c + (rng.random(n) < 0.3)) % 4
    return np.column_stack([a, b, c, d]).astype(np.float64)


_TESTS = [(0, 1, ()), (0, 2, (1,)), (0, 3, (1, 2)), (1, 3, (2,)), (2, 3, ())]
_CHISQ = [5.570270172544039e-18, 0.6946324469306543, 0.45249901975116247, 4.782080556141617e-37,
          8.133317252256037e-13]  # fmt: skip
_GSQ = [7.268003537936198e-20, 0.6936440487566553, 0.4695380964444543, 5.178677721066172e-43,
        9.998537786635298e-15]  # fmt: skip
_FAMILIES = [(0, ()), (1, (0,)), (3, (1, 2)), (3, (0, 1, 2))]
_BDEU = [284.9438585690648, 345.8688129007728, 315.62260494976687, 369.97498750730057]
_BIC = [282.05698104858, 339.56452655709927, 346.4207481756152, 436.19116897316906]


@pytest.mark.parametrize(("test", "expected"), [(ChiSquare, _CHISQ), (GSquare, _GSQ)])
def test_tests_match_reference_pvalues(test, expected):
    ci = test(_fixed_data())
    got = [ci(x, y, s) for x, y, s in _TESTS]
    np.testing.assert_allclose(got, expected, rtol=1e-9)


@pytest.mark.parametrize(("score", "expected"), [(BDeuScore, _BDEU), (DiscreteBICScore, _BIC)])
def test_scores_match_reference_values(score, expected):
    s = score(_fixed_data())
    np.testing.assert_allclose([s.score(i, pa) for i, pa in _FAMILIES], expected, rtol=1e-12)


@pytest.mark.parametrize("test", [ChiSquare, GSquare])
def test_pvalue_ignores_argument_order_and_level_labels(test):
    x = _fixed_data()
    relabeled = x * 10.0 - 3.0  # the same categories under other numbers
    a, b = test(x), test(relabeled)
    assert a(0, 3, (2, 1)) == a(3, 0, (1, 2)) == b(0, 3, (1, 2))


@pytest.mark.parametrize("test", [ChiSquare, GSquare])
def test_no_degrees_of_freedom_gives_one(test):
    # y takes one level in every stratum of z, so no test of x and y given z is possible.
    z = np.repeat([0.0, 1.0], 50)
    x = np.tile([0.0, 1.0], 50)
    assert test(np.column_stack([x, z, z]))(0, 2, (1,)) == 1.0


def test_unobserved_configurations_add_nothing():
    # Level 2 of column 0 never meets level 1 of column 1, so their configuration is unobserved.
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 0]] * 20, dtype=np.float64)
    n_j = np.array([40.0, 40.0, 40.0])  # child 1's counts per level of its parent, column 0
    n_jk = np.array([[20.0, 20.0], [20.0, 20.0], [40.0, 0.0]])
    log_lik = sum(c * np.log(c / n) for row, n in zip(n_jk, n_j) for c in row if c > 0)
    penalty = np.log(len(x)) * 3 * (2 - 1)
    assert DiscreteBICScore(x).score(1, (0,)) == pytest.approx(-2 * log_lik + penalty, rel=1e-12)


def test_add_scores_equal_score_and_fill_its_cache():
    x = _fixed_data()
    batched, single = BDeuScore(x), BDeuScore(x)
    got = batched.add_scores(3, [1], [0, 2])
    assert got.tolist() == [single.score(3, (0, 1)), single.score(3, (1, 2))]
    assert (3, (0, 1)) in batched._cache and (3, (1, 2)) in batched._cache


def test_registered_names_build_the_tests():
    x = _fixed_data()
    assert isinstance(make_indep_test("chisq", x), ChiSquare)
    assert isinstance(make_indep_test("gsq", x), GSquare)


def test_encode_rejects_non_finite():
    with pytest.raises(ValueError, match="NaN"):
        encode_discrete(np.array([[0.0, np.nan], [1.0, 0.0]]))
    codes, levels = encode_discrete(np.array([[5, 1], [7, 1], [5, 3]]))
    assert codes.tolist() == [[0, 0], [1, 0], [0, 1]] and levels.tolist() == [2, 2]


def test_structure_prior_penalizes_each_parent():
    x = _fixed_data()
    plain = BDeuScore(x, structure_prior=1.5)  # p = 0.5: the same prior for every parent set
    default = BDeuScore(x)  # p = 1/3
    for pa in [(), (0,), (0, 1, 2)]:
        k = len(pa)
        prior = k * np.log(1 / 3) + (3 - k) * np.log(2 / 3) - 3 * np.log(0.5)
        assert default.score(3, pa) - plain.score(3, pa) == pytest.approx(-2 * prior)


def test_structure_prior_is_finite_for_two_variables():
    # One other variable makes p = structure_prior / 1 reach 1; the cap at 0.5 keeps it finite.
    x = _fixed_data()[:, :2]
    s = BDeuScore(x)
    assert np.isfinite([s.score(1, ()), s.score(1, (0,))]).all()
    one = BDeuScore(x[:, :1])
    assert np.isfinite(one.score(0, ()))


def test_score_settings_are_checked():
    x = _fixed_data()
    with pytest.raises(ValueError, match="ess"):
        BDeuScore(x, ess=0.0)
    with pytest.raises(ValueError, match="structure_prior"):
        BDeuScore(x, structure_prior=0.0)
    with pytest.raises(ValueError, match="lambda_value"):
        DiscreteBICScore(x, lambda_value=-1.0)
    with pytest.raises(ValueError, match="BDeu"):
        make_discrete_score("local_score_BDeu", x, lambda_value=2.0)
    with pytest.raises(NotImplementedError, match="local_score_BIC_from_cov"):
        make_score_engine(
            "local_score_CV", x, lambda_value=1.0, bic_name="local_score_BIC_from_cov"
        )


def test_lambda_scales_the_discrete_penalty():
    x = _fixed_data()
    plain, double = DiscreteBICScore(x), DiscreteBICScore(x, lambda_value=2.0)
    penalty = np.log(len(x)) * 3 * 2 * (4 - 1)  # q = 3 * 2 configurations of (b, c), r = 4
    assert double.score(3, (1, 2)) - plain.score(3, (1, 2)) == pytest.approx(penalty)


def test_score_difference_equals_two_full_scores():
    score = BDeuScore(_fixed_data())
    engine = ScoreDifference(score, 4)
    adds = engine.add_deltas(3, [1])
    assert adds[3] == adds[1] == np.inf
    assert adds[2] == pytest.approx(score.score(3, [1, 2]) - score.score(3, [1]))
    removes = engine.remove_deltas(3, [1, 2])
    assert removes[0] == np.inf
    assert removes[1] == pytest.approx(score.score(3, [2]) - score.score(3, [1, 2]))
    assert engine.score_many_with_base(3, [1], [("add", 2), ("remove", 1)]) == [
        score.score(3, [1, 2]),
        score.score(3, []),
    ]
    assert engine.delta(3, (1,), 2, "add") == engine.delta_many_with_base(3, [1], [("add", 2)])[0]


# ---- input handling ------------------------------------------------------------------------------


def test_discrete_matrix_codes_every_categorical_kind():
    frame = pd.DataFrame(
        {
            "grade": pd.Categorical(["lo", "hi", "lo", "mid"], categories=["lo", "mid", "hi"]),
            "flag": [True, False, True, True],
            "word": ["x", "y", "y", "x"],
            "count": [1, 3, 3, 1],
            "whole": [2.0, 0.0, 2.0, 0.0],
        }
    )
    assert discrete_matrix(frame).tolist() == [
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0, 0.0],
    ]


@pytest.mark.parametrize(
    ("column", "message"),
    [
        ([0.5, 1.0, 2.0, 1.0], "non-integer"),
        (pd.Categorical(["x", None, "y", "x"]), "missing"),
        (["x", None, "y", "x"], "missing"),
        ([1, 1, 1, 1], "constant"),
    ],
)
def test_discrete_matrix_names_the_bad_column(column, message):
    frame = pd.DataFrame({"bad": column, "ok": [0, 1, 1, 0]})
    with pytest.raises(ValueError, match=f"'bad'.*{message}|{message}.*'bad'"):
        discrete_matrix(frame)


def test_continuous_setting_rejects_categorical_columns():
    frame = pd.DataFrame({"a": pd.Categorical(["x", "y", "x"]), "b": [0.1, 0.4, 0.2]})
    with pytest.raises(ValueError, match="'a' is categorical.*chisq"):
        data_matrix(frame)


def test_fit_matrix_refuses_a_discrete_test_with_a_continuous_score():
    with pytest.raises(ValueError, match="disagree"):
        fit_matrix(_fixed_data(), indep_test="chisq", score_func="local_score_BIC")
    assert fit_matrix(_fixed_data(), indep_test="gsq", score_func="local_score_BDeu").shape == (
        200,
        4,
    )


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
def test_gfci_names_the_mismatch():
    with pytest.raises(ValueError, match="disagree"):
        andrey.gfci(_fixed_data(), indep_test="chisq")


def test_a_dataframe_keeps_its_labels_under_a_discrete_score():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 1000)
    c = rng.integers(0, 2, 1000)
    b = np.where(rng.random(1000) < 0.9, a | c, 1 - (a | c))
    frame = pd.DataFrame(
        {
            "a": pd.Categorical(np.where(a == 1, "on", "off")),
            "b": b.astype(bool),
            "c": np.where(c == 1, "on", "off"),
        }
    )
    edges = andrey.ges(frame, score_func="local_score_BDeu").structure.oriented_edges()
    assert edges == [("a", "b", "directed"), ("c", "b", "directed")]


@pytest.mark.parametrize("test", [ChiSquare, GSquare])
def test_a_batch_gives_each_set_its_single_p_value(test):
    ci = test(_fixed_data())
    sets = [(), (1,), (2,), (1, 2), (2, 1)]
    assert ci.batched_call(3, 0, sets) == [ci(0, 3, s) for s in sets]


@pytest.mark.parametrize("test", [ChiSquare, GSquare])
def test_counting_only_the_observed_strata_gives_the_same_p_values(test, monkeypatch):
    from andrey.core import discrete

    x = _fixed_data()
    dense = [test(x)(x_, y_, s) for x_, y_, s in _TESTS]
    monkeypatch.setattr(discrete, "_DENSE_MIN", 0)
    monkeypatch.setattr(discrete, "_DENSE_ROWS", 0)
    np.testing.assert_allclose([test(x)(x_, y_, s) for x_, y_, s in _TESTS], dense, rtol=1e-12)


@pytest.mark.parametrize("score", [BDeuScore, DiscreteBICScore])
def test_scores_counting_only_the_observed_strata_agree(score, monkeypatch):
    from andrey.core import discrete

    x = _fixed_data()
    dense = [score(x).score(i, pa) for i, pa in _FAMILIES]
    monkeypatch.setattr(discrete, "_DENSE_MIN", 0)
    monkeypatch.setattr(discrete, "_DENSE_ROWS", 0)
    np.testing.assert_allclose([score(x).score(i, pa) for i, pa in _FAMILIES], dense, rtol=1e-12)
