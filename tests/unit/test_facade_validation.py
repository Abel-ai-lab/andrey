"""Facade input-validation guards: unusable inputs raise the documented error, naming the argument.

Direct Python calls (not the CLI, which validates inputs of its own) must not leak a raw
``IndexError`` / ``LinAlgError`` / ``StopIteration`` or return a result from data no method can use.
"""

from __future__ import annotations

import warnings
from functools import partial

import numpy as np
import pytest

import andrey
from andrey.data import sample_scm


def test_direct_lingam_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        andrey.direct_lingam(np.random.default_rng(0).standard_normal(100))


@pytest.mark.parametrize("shape", [(100,), (100, 1), (100, 3)])
def test_pnl_rejects_anything_but_a_pair(shape):
    with pytest.raises(ValueError, match=r"\(n_samples, 2\)"):
        andrey.pnl(np.random.default_rng(0).standard_normal(shape))


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_pnl_rejects_an_alpha_outside_the_unit_interval(alpha):
    with pytest.raises(ValueError, match="alpha"):
        andrey.pnl(np.random.default_rng(0).standard_normal((100, 2)), alpha=alpha)


def _data(n=60, d=3):
    return np.random.default_rng(0).uniform(-1, 1, (n, d))


def _with(x, cell, value):
    x = x.copy()
    x[cell] = value
    return x


# Every method that takes one data matrix, called with its other required arguments.
SINGLE = {
    "pc": andrey.pc,
    "fci": andrey.fci,
    "gfci": andrey.gfci,
    "cdnod": lambda data: andrey.cdnod(data, c_indx=np.arange(len(data)) % 2),
    "ges": andrey.ges,
    "gies": andrey.gies,
    "xges": andrey.xges,
    "hc": andrey.hc,
    "tabu": andrey.tabu,
    "exact_search": andrey.exact_search,
    "calm": andrey.calm,
    "boss": andrey.boss,
    "grasp": andrey.grasp,
    "direct_lingam": andrey.direct_lingam,
    "ica_lingam": andrey.ica_lingam,
    "varma_lingam": andrey.varma_lingam,
    "gin": andrey.gin,
    "rcd": andrey.rcd,
    "camuv": andrey.camuv,
    "bottom_up_parce_lingam": andrey.bottom_up_parce_lingam,
}

# Data no method can fit, and what the error must say about it.
UNUSABLE = {
    "1-D": (lambda x: x[:, 0], r"data must be a 2-D"),
    "one row": (lambda x: x[:1], r"data needs at least \d+ rows"),
    "no rows": (lambda x: x[:0], r"data needs at least \d+ rows"),
    "NaN": (lambda x: _with(x, (3, 2), np.nan), r"data .*row 3, column 2"),
    "inf": (lambda x: _with(x, (5, 0), np.inf), r"data .*row 5, column 0"),
    "constant column": (lambda x: _with(x, (slice(None), 1), 3.0), r"data column 1 is constant"),
    "text": (lambda x: np.full(x.shape, "high"), r"data must hold numbers"),
}


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize("case", list(UNUSABLE))
@pytest.mark.parametrize("method", list(SINGLE))
def test_unusable_data_raises_a_value_error_naming_it(method, case):
    if method == "calm":
        pytest.importorskip("torch")  # without the extra, calm raises ImportError first
    make, message = UNUSABLE[case]
    with pytest.raises(ValueError, match=message):
        SINGLE[method](make(_data()))


def test_a_dataframe_error_names_the_constant_column():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_with(_data(), (slice(None), 1), 3.0), columns=["a", "b", "c"])
    with pytest.raises(ValueError, match="column 'b' is constant"):
        andrey.pc(frame)


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize("method", [m for m in SINGLE if m not in ("calm", "varma_lingam")])
def test_one_column_gives_a_one_node_graph(method):
    structure = SINGLE[method](_data(d=1)).structure
    assert structure.n_nodes == 1
    assert structure.oriented_edges() == []


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize("method", ["pc", "ges", "hc", "xges"])
def test_no_columns_give_an_empty_graph(method):
    assert SINGLE[method](_data(d=0)).structure.n_nodes == 0


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
def test_varma_lingam_needs_two_variables():
    with pytest.raises(ValueError, match="data needs at least 2 columns"):
        andrey.varma_lingam(_data(d=1))


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
def test_pnl_rejects_unusable_data():
    with pytest.raises(ValueError, match=r"data .*row 2, column 0"):
        andrey.pnl(_with(_data(d=2), (2, 0), np.nan))


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize(
    ("fit", "name", "least"),
    [
        (andrey.multi_group_direct_lingam, "data_groups", "two datasets"),
        (andrey.longitudinal_lingam, "data_list", "two time points"),
    ],
)
def test_multi_dataset_methods_name_the_dataset_at_fault(fit, name, least):
    with pytest.raises(ValueError, match=rf"{name}\[1\] .*row 4, column 1"):
        fit([_data(), _with(_data(), (4, 1), np.nan)])
    with pytest.raises(ValueError, match=rf"{name} must hold at least {least}"):
        fit([_data()])


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
def test_multi_group_direct_lingam_needs_the_same_columns_in_every_group():
    with pytest.raises(ValueError, match="data_groups needs the same columns"):
        andrey.multi_group_direct_lingam([_data(d=3), _data(d=4)])


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize(("n_lags", "error"), [(0, ValueError), (3, ValueError), (1.5, TypeError)])
def test_longitudinal_lingam_takes_n_lags_up_to_the_time_points_minus_one(n_lags, error):
    data = [_data() for _ in range(3)]
    with pytest.raises(error, match="n_lags"):
        andrey.longitudinal_lingam(data, n_lags=n_lags)
    assert andrey.longitudinal_lingam(data, n_lags=2).structure.n_lags == 3  # lags 0, 1, 2


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize(
    ("order", "error"),
    [((-1, 1), ValueError), ((0, 0), ValueError), ((1,), TypeError), ((1.0, 1), TypeError)],
)
def test_varma_lingam_checks_order(order, error):
    with pytest.raises(error, match="order"):
        andrey.varma_lingam(_data(), order=order)


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
def test_gin_rejects_an_alpha_outside_the_unit_interval(alpha):
    with pytest.raises(ValueError, match="alpha"):
        andrey.gin(_data(d=4), alpha=alpha)


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize(
    ("fit", "setting", "value", "error"),
    [
        *[(andrey.rcd, "max_explanatory_vars", bad, ValueError) for bad in (0, -1)],
        *[(andrey.camuv, "max_explanatory_vars", bad, ValueError) for bad in (1, 0, -1)],
        (andrey.rcd, "max_explanatory_vars", 1.5, TypeError),
        (andrey.camuv, "max_explanatory_vars", 1.5, TypeError),
        *[
            (andrey.rcd, level, bad, ValueError)
            for level in ("cor_alpha", "ind_alpha", "shapiro_alpha")
            for bad in (0.0, 1.0, 2.0)
        ],
        *[(andrey.camuv, "alpha", bad, ValueError) for bad in (0.0, 1.0, 2.0)],
        *[(andrey.bottom_up_parce_lingam, "alpha", bad, ValueError) for bad in (0.0, 1.0, 2.0)],
    ],
)
def test_hidden_confounder_methods_reject_a_bad_setting(fit, setting, value, error):
    with pytest.raises(error, match=setting):
        fit(_data(), **{setting: value})


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
def test_gin_labels_name_each_column():
    with pytest.raises(ValueError, match="labels must name each of the 4 columns"):
        andrey.gin(_data(d=4), labels=("a", "b"))


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize(
    ("fit", "max_iter", "error"),
    [
        (andrey.hc, 0, ValueError),
        (andrey.hc, 2.5, TypeError),
        (andrey.hc, True, TypeError),
        (andrey.tabu, 0, ValueError),
        (andrey.tabu, 2.5, TypeError),
        (andrey.ica_lingam, 0, ValueError),
        (andrey.ica_lingam, 1.5, TypeError),
    ],
)
def test_max_iter_is_a_positive_int(fit, max_iter, error):
    with pytest.raises(error, match="max_iter"):
        fit(_data(), max_iter=max_iter)


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
def test_hc_stops_at_max_iter_and_warns():
    chain = np.cumsum(np.random.default_rng(0).standard_normal((500, 8)), axis=1)
    with pytest.warns(andrey.SearchLimitWarning, match="max_iter=3"):
        out = andrey.hc(chain, max_iter=3)
    assert len(out.structure.oriented_edges()) == 3
    assert len(andrey.hc(chain).structure.oriented_edges()) == 7  # runs to a local optimum


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
def test_hc_sets_no_move_limit_by_default():
    # This dense graph takes over 200 moves: a limit of 200 stops the search, the default does not.
    x = sample_scm(d=30, n=800, seed=0, density=16.0).data
    with pytest.warns(andrey.SearchLimitWarning, match="max_iter=200"):
        andrey.hc(x, max_iter=200)
    with warnings.catch_warnings():
        warnings.simplefilter("error", andrey.SearchLimitWarning)
        out = andrey.hc(x)
    assert len(out.structure.oriented_edges()) > 200


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize(
    ("settings", "named", "error"),
    [
        ({"tabu_length": 0}, "tabu_length", ValueError),
        ({"tabu_length": 2.5}, "tabu_length", TypeError),
        ({"max_tabu": 0}, "max_tabu", ValueError),
        ({"max_tabu": 11}, "max_tabu", ValueError),  # longer than the default tabu list
        ({"tabu_length": 3, "max_tabu": 4}, "max_tabu", ValueError),
        ({"max_tabu": True}, "max_tabu", TypeError),
        ({"lambda_value": -1.0}, "lambda_value", ValueError),
        ({"score_func": "local_score_BDeu"}, "score_func", NotImplementedError),
    ],
)
def test_tabu_rejects_a_bad_setting(settings, named, error):
    with pytest.raises(error, match=named):
        andrey.tabu(_data(), **settings)


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
def test_tabu_stops_at_max_iter_and_warns():
    chain = np.cumsum(np.random.default_rng(0).standard_normal((500, 8)), axis=1)
    with pytest.warns(andrey.SearchLimitWarning, match="max_iter=3"):
        out = andrey.tabu(chain, max_iter=3)
    assert len(out.structure.oriented_edges()) == 3


def test_cdnod_with_one_domain_gives_the_pc_graph():
    rng = np.random.default_rng(1)
    w = np.triu(rng.uniform(0.5, 1.5, (6, 6)) * (rng.random((6, 6)) < 0.4), 1)
    x = rng.standard_normal((500, 6)) @ np.linalg.inv(np.eye(6) - w)
    one_domain = andrey.cdnod(x, c_indx=np.full(500, 3.0)).structure
    assert np.array_equal(one_domain.to_numpy(), andrey.pc(x).structure.to_numpy())


def test_cdnod_rejects_nan_in_c_indx():
    with pytest.raises(ValueError, match="c_indx contains NaN"):
        andrey.cdnod(_data(), c_indx=np.r_[np.nan, np.zeros(59)])


# Every categorical setting of each facade that takes one, called on one data matrix.
CATEGORICAL = {
    ("pc", "indep_test", "chisq"): lambda x: andrey.pc(x, indep_test="chisq"),
    ("pc", "indep_test", "gsq"): lambda x: andrey.pc(x, indep_test="gsq"),
    ("fci", "indep_test", "chisq"): lambda x: andrey.fci(x, indep_test="chisq"),
    ("fci", "indep_test", "gsq"): lambda x: andrey.fci(x, indep_test="gsq"),
    ("cdnod", "indep_test", "gsq"): lambda x: andrey.cdnod(
        x, np.arange(len(x)) % 2, indep_test="gsq"
    ),
    ("gfci", "indep_test", "chisq"): lambda x: andrey.gfci(
        x, indep_test="chisq", score_func="local_score_BDeu"
    ),
    ("ges", "score_func", "local_score_BDeu"): lambda x: andrey.ges(
        x, score_func="local_score_BDeu"
    ),
    ("hc", "score_func", "local_score_BDeu"): lambda x: andrey.hc(x, score_func="local_score_BDeu"),
}


def _categories(n=200, d=3):
    return np.random.default_rng(0).integers(0, 3, (n, d)).astype(np.float64)


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize("setting", list(CATEGORICAL), ids="-".join)
def test_a_categorical_setting_rejects_a_fractional_value(setting):
    _, option, value = setting
    with pytest.raises(ValueError, match=rf"{option}='{value}' reads data .* 0\.5 at row 7, col"):
        CATEGORICAL[setting](_with(_categories(), (7, 2), 0.5))
    assert CATEGORICAL[setting](_categories()).structure.n_nodes == 3


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize(
    ("score_func", "indep_test"), [("local_score_BDeu", "fisherz"), ("local_score_BIC", "chisq")]
)
def test_gfci_rejects_a_score_and_test_that_read_the_data_differently(score_func, indep_test):
    with pytest.raises(ValueError, match="score and test must read the data alike"):
        andrey.gfci(_categories(), score_func=score_func, indep_test=indep_test)


@pytest.mark.parametrize("code", [2.0**53, -(2.0**60)])
def test_a_categorical_setting_rejects_codes_too_large_to_stay_distinct(code):
    with pytest.raises(ValueError, match=r"2\*\*53 or more"):
        andrey.pc(_with(_categories(), (5, 0), code), indep_test="chisq")


def test_a_categorical_error_names_the_dataframe_column():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_with(_categories(), (4, 1), 2.25), columns=["a", "b", "c"])
    with pytest.raises(ValueError, match=r"2\.25 at row 4, column 'b'"):
        andrey.pc(frame, indep_test="chisq")


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize("fit", [andrey.ges, andrey.hc, partial(andrey.gfci, indep_test="chisq")])
def test_bdeu_still_checks_lambda_value(fit):
    """BDeu has no penalty weight, but a setting outside its range is an error under any score."""
    with pytest.raises(ValueError, match="lambda_value"):
        fit(_categories(), score_func="local_score_BDeu", lambda_value=-1.0)


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
def test_cdnod_with_a_categorical_test_rejects_a_fractional_index():
    c_indx = np.r_[0.5, np.zeros(99), np.ones(100)]
    with pytest.raises(ValueError, match=r"c_indx .* 0\.5 at row 0"):
        andrey.cdnod(_categories(), c_indx, indep_test="chisq")


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize(
    ("setting", "value", "error"),
    [
        ("extended_search", 1, TypeError),
        ("extended_search", "yes", TypeError),
        ("lambda_value", -1.0, ValueError),
        ("lambda_value", float("inf"), ValueError),
        ("score_func", "local_score_BDeu", NotImplementedError),
    ],
)
def test_xges_rejects_a_bad_setting(setting, value, error):
    with pytest.raises(error, match=setting):
        andrey.xges(_data(), **{setting: value})


# Every mixed-data setting of each facade that takes one, called on one data matrix.
MIXED = {
    ("pc", "dg"): lambda x, **kw: andrey.pc(x, indep_test="dg", **kw),
    ("fci", "cg"): lambda x, **kw: andrey.fci(x, indep_test="cg", **kw),
    ("gfci", "dg"): lambda x, **kw: andrey.gfci(
        x, score_func="local_score_DG", indep_test="dg", **kw
    ),
    ("ges", "local_score_CG"): lambda x, **kw: andrey.ges(x, score_func="local_score_CG", **kw),
    ("hc", "local_score_DG"): lambda x, **kw: andrey.hc(x, score_func="local_score_DG", **kw),
    ("boss", "local_score_CG"): lambda x, **kw: andrey.boss(x, score_func="local_score_CG", **kw),
    ("grasp", "local_score_DG"): lambda x, **kw: andrey.grasp(x, score_func="local_score_DG", **kw),
    ("exact_search", "local_score_CG"): lambda x, **kw: andrey.exact_search(
        x, score_func="local_score_CG", **kw
    ),
}


def _mixed_data(n=200):
    """Columns: 3 categories, continuous, 2 categories."""
    rng = np.random.default_rng(0)
    return np.column_stack(
        [rng.integers(0, 3, n), rng.standard_normal(n), rng.integers(0, 2, n)]
    ).astype(np.float64)


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize("setting", list(MIXED), ids="-".join)
def test_a_mixed_setting_needs_a_categorical_column(setting):
    with pytest.raises(ValueError, match="no column of data is categorical"):
        MIXED[setting](_mixed_data())
    assert MIXED[setting](_mixed_data(), categorical=[0, 2]).structure.n_nodes == 3


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize("setting", list(MIXED), ids="-".join)
def test_a_mixed_setting_rejects_a_fractional_category(setting):
    with pytest.raises(ValueError, match=r"reads data .* 0\.5 at row 7, column 2"):
        MIXED[setting](_with(_mixed_data(), (7, 2), 0.5), categorical=[0, 2])


@pytest.mark.parametrize(
    ("data", "categorical", "error", "message"),
    [
        ("frame", ["a", "zz"], ValueError, "names column 'zz', which data does not have"),
        ("array", ["a"], ValueError, "data has no column names"),
        ("array", [3], ValueError, "lists column 3, but data has 3 columns"),
        ("array", "a", TypeError, "list of column names or positions"),
        ("array", [1.5], TypeError, "names or 0-based positions, got float"),
        ("array", [True], TypeError, "names or 0-based positions, got bool"),
    ],
)
def test_categorical_lists_columns_the_data_has(data, categorical, error, message):
    pd = pytest.importorskip("pandas")
    x = _mixed_data()
    x = pd.DataFrame(x, columns=["a", "b", "c"]) if data == "frame" else x
    with pytest.raises(error, match=message):
        andrey.pc(x, indep_test="dg", categorical=categorical)


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize(
    "fit",
    [
        lambda x: andrey.pc(x, categorical=[0]),
        lambda x: andrey.pc(x, indep_test="chisq", categorical=[0]),
        lambda x: andrey.ges(x, categorical=[0]),
        lambda x: andrey.boss(x, categorical=[0]),
        lambda x: andrey.exact_search(x, categorical=[0]),
    ],
)
def test_a_setting_that_reads_no_column_types_rejects_categorical(fit):
    with pytest.raises(ValueError, match="leave categorical as None"):
        fit(np.round(_mixed_data()))


def test_a_column_that_does_not_hold_numbers_must_be_categorical():
    pd = pytest.importorskip("pandas")
    x = _mixed_data()
    frame = pd.DataFrame({"a": np.array(["p", "q", "r"])[x[:, 0].astype(int)], "b": x[:, 1]})
    frame["c"] = x[:, 2]
    with pytest.raises(ValueError, match="column 'a' does not hold numbers; list it"):
        andrey.pc(frame, indep_test="dg", categorical=["c"])
    assert andrey.pc(frame, indep_test="dg").structure.n_nodes == 3  # an object column
    frame.loc[5, "a"] = None
    with pytest.raises(ValueError, match="column 'a' has a missing value at row 5"):
        andrey.pc(frame, indep_test="dg")


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize(
    ("score_func", "indep_test"),
    [("local_score_DG", "fisherz"), ("local_score_BIC", "cg"), ("local_score_BDeu", "dg")],
)
def test_gfci_rejects_a_mixed_score_or_test_without_its_pair(score_func, indep_test):
    with pytest.raises(ValueError, match="score and test must read the data alike"):
        andrey.gfci(_mixed_data(), score_func=score_func, indep_test=indep_test, categorical=[0])


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
def test_cdnod_with_a_mixed_test_takes_continuous_data_and_rejects_a_fractional_index():
    """The index is the categorical variable, so the data may have no categorical column."""
    x = _mixed_data()[:, [1]]
    c_indx = np.repeat([0.0, 1.0], 100)
    assert andrey.cdnod(x, c_indx, indep_test="dg").structure.n_nodes == 1
    with pytest.raises(ValueError, match=r"c_indx .* 0\.5 at row 0"):
        andrey.cdnod(x, np.r_[0.5, c_indx[1:]], indep_test="cg")
