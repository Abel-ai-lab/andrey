"""Facade input-validation guards: unusable inputs raise the documented error, naming the argument.

Direct Python calls (not the CLI, which validates inputs of its own) must not leak a raw
``IndexError`` / ``LinAlgError`` / ``StopIteration`` or return a result from data no method can use.
"""

from __future__ import annotations

import importlib
import re
import warnings
from functools import partial

import numpy as np
import pytest

import andrey
from andrey.core.orient import dag2cpdag, from_structure, pdag2dag
from andrey.core.score import LINEAR_SCORES
from andrey.data import sample_scm


def test_direct_lingam_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        andrey.direct_lingam(np.random.default_rng(0).standard_normal(100))


#: The pairwise direction tests, which take a two-column table.
PAIRWISE = [andrey.pnl, andrey.anm]


@pytest.mark.parametrize("fit", PAIRWISE, ids=lambda f: f.__name__)
@pytest.mark.parametrize("shape", [(100,), (100, 1), (100, 3)])
def test_pairwise_tests_reject_anything_but_a_pair(fit, shape):
    with pytest.raises(ValueError, match=r"\(n_samples, 2\)"):
        fit(np.random.default_rng(0).standard_normal(shape))


@pytest.mark.parametrize("fit", PAIRWISE, ids=lambda f: f.__name__)
@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_pairwise_tests_reject_an_alpha_outside_the_unit_interval(fit, alpha):
    with pytest.raises(ValueError, match="alpha"):
        fit(np.random.default_rng(0).standard_normal((100, 2)), alpha=alpha)


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
    "cdnod": lambda data, **kw: andrey.cdnod(data, domain_index=np.arange(len(data)) % 2, **kw),
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
    "pcmci": andrey.pcmci,
    "pcmciplus": andrey.pcmciplus,
    "lpcmci": andrey.lpcmci,
    "granger": andrey.granger,
    "gin": andrey.gin,
    "rcd": andrey.rcd,
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


@pytest.mark.parametrize("case", list(UNUSABLE))
@pytest.mark.parametrize("method", list(SINGLE))
def test_unusable_data_raises_a_value_error_naming_it(method, case):
    if method == "calm":
        pytest.importorskip("torch")  # without the extra, calm raises ImportError first
    make, message = UNUSABLE[case]
    with pytest.raises(ValueError, match=message):
        SINGLE[method](make(_data()))


# One method from each family, plus a mixed-data setting, which reads its data on its own path.
FAMILIES = {
    "pc": andrey.pc,
    "pc mixed": partial(andrey.pc, indep_test="degen_gauss", categorical=["a"]),
    "ges": andrey.ges,
    "boss": andrey.boss,
    "direct_lingam": andrey.direct_lingam,
    "pcmci": andrey.pcmci,
    "gin": andrey.gin,
    "rcd": andrey.rcd,
}


@pytest.mark.parametrize("method", list(FAMILIES))
def test_every_constant_column_is_named_with_the_fix(method):
    pd = pytest.importorskip("pandas")
    x = _data(200, 5)
    x[:, 0] = np.arange(200) % 3
    x[:, 1] = 3.0
    x[:, 3] = -1.0
    frame = pd.DataFrame(x, columns=["a", "b", "c", "d", "e"])
    with pytest.raises(ValueError, match=r"data columns 'b' and 'd' are constant; .*drop them"):
        FAMILIES[method](frame)


def test_a_dataframe_error_names_the_constant_column():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_with(_data(), (slice(None), 1), 3.0), columns=["a", "b", "c"])
    with pytest.raises(ValueError, match="column 'b' is constant"):
        andrey.pc(frame)


@pytest.mark.parametrize(
    "method",
    [
        m
        for m in SINGLE
        if m not in ("calm", "varma_lingam", "pcmci", "pcmciplus", "lpcmci", "granger")
    ],
)
def test_one_column_gives_a_one_node_graph(method):
    structure = SINGLE[method](_data(d=1)).structure
    assert structure.n_nodes == 1
    assert structure.oriented_edges() == []


@pytest.mark.parametrize("method", ["pc", "ges", "hc", "xges"])
def test_no_columns_give_an_empty_graph(method):
    assert SINGLE[method](_data(d=0)).structure.n_nodes == 0


BIC = ["ges", "gies", "gfci", "xges", "hc", "tabu", "boss", "grasp", "exact_search"]


def _sum_of_two_columns():
    x = _data(200, 6)
    x[:, 5] = x[:, 2] + x[:, 3]
    return x


# Data whose BIC score is meaningless, and what the error must say about it.
BROKEN = {
    "scaled copy": (
        lambda: np.column_stack([_data(), 2.0 * _data()[:, 0] + 1.0]),
        r"\b3 repeats 0\b",
    ),
    "sum of two columns": (_sum_of_two_columns, r"data columns \[2, 3, 5\] are linearly dependent"),
}


# Every BIC search runs the check; one of them shows the pairwise message.
@pytest.mark.parametrize(
    ("method", "case"), [(m, "sum of two columns") for m in BIC] + [("ges", "scaled copy")]
)
def test_bic_searches_reject_dependent_columns(method, case):
    make, message = BROKEN[case]
    with pytest.raises(ValueError, match=message):
        SINGLE[method](make())


def _two_groups_of_copies():
    """x2 and x7 repeat x1, and x5 repeats x3, up to scale and shift."""
    pd = pytest.importorskip("pandas")
    x = _data(200, 8)
    x[:, 2] = 2.0 * x[:, 1] + 1.0
    x[:, 7] = -x[:, 1]
    x[:, 5] = 3.0 * x[:, 3] - 2.0
    return pd.DataFrame(x, columns=[f"x{j}" for j in range(8)])


# The searches that reject copies before the fit: the BIC searches, the partial-correlation
# constraint searches, and the lagged partial-correlation searches.
COPY_CHECKED = [*BIC, "pc", "fci", "cdnod", "pcmci", "pcmciplus", "lpcmci"]


@pytest.mark.parametrize("method", COPY_CHECKED)
def test_copy_checked_searches_name_every_group_of_copies_by_column_name(method):
    with pytest.raises(ValueError) as error:
        SINGLE[method](_two_groups_of_copies())
    message = str(error.value)
    assert re.search(r"'x2' and 'x7' repeat 'x1'", message)
    assert re.search(r"'x5' repeats 'x3'", message)
    assert "keep one column" in message


@pytest.mark.parametrize("indep_test", ["fisherz", "student_t"])
@pytest.mark.parametrize("method", ["pc", "fci", "cdnod"])
def test_constraint_searches_reject_copies_no_test_meets(method, indep_test):
    # The copy is unrelated to every other column, so no test conditions on it with its original.
    x = _data(200, 4)
    x[:, 3] = 5.0 - 2.0 * x[:, 0]
    with pytest.raises(ValueError, match=r"\b3 repeats 0\b"):
        SINGLE[method](x, indep_test=indep_test)


def test_cdnod_names_a_column_that_repeats_the_domain_index():
    pd = pytest.importorskip("pandas")
    x = _data(200, 3)
    index = np.arange(200) % 2
    frame = pd.DataFrame(np.column_stack([x, 0.5 * index + 0.25]), columns=["a", "b", "c", "d"])
    with pytest.raises(ValueError, match=r"domain_index repeats 'd'"):
        andrey.cdnod(frame, domain_index=index)


@pytest.mark.parametrize("method", [m for m in BIC if m != "exact_search"])
def test_bic_searches_fit_fewer_rows_than_columns(method):
    out = SINGLE[method](np.random.default_rng(0).standard_normal((6, 12)))
    assert out.structure.n_nodes == 12
    if method == "gfci":  # a PAG, with no score in its metadata
        assert out.structure.kind == "pag"
    else:
        assert np.isfinite(out.metadata["score"])


def test_exact_search_gives_no_variable_more_parents_than_rows_minus_2():
    dag = andrey.exact_search(np.random.default_rng(0).standard_normal((4, 6))).structure
    heads = [v for _, v, _ in dag.oriented_edges(index=True)]
    assert dag.n_nodes == 6
    assert all(heads.count(v) <= 2 for v in range(6))


@pytest.mark.parametrize("depth", [1, 2])
def test_grasp_returns_a_valid_cpdag_at_shallower_depths(depth):
    data = sample_scm(d=8, n=400, seed=1, density=2.0, scale="standardize").data
    out = andrey.grasp(data, depth=depth)
    cpdag = from_structure(out.structure)
    assert out.structure.kind == "cpdag" and out.structure.n_nodes == 8
    np.testing.assert_array_equal(dag2cpdag(pdag2dag(cpdag)), cpdag)
    assert np.isfinite(out.metadata["score"])


@pytest.mark.parametrize("depth", [0, -1])
def test_grasp_rejects_a_depth_below_1(depth):
    with pytest.raises(ValueError, match="depth must be an int at least 1"):
        andrey.grasp(_data(), depth=depth)


@pytest.mark.parametrize("fit", [andrey.varma_lingam, andrey.granger], ids=lambda f: f.__name__)
def test_time_series_methods_need_two_variables(fit):
    with pytest.raises(ValueError, match="data needs at least 2 columns"):
        fit(_data(d=1))


@pytest.mark.parametrize(
    ("setting", "value", "error"),
    [
        ("maxlag", 0, ValueError),
        ("maxlag", 1.5, TypeError),
        ("cv", 1, ValueError),
        ("cv", 2.5, TypeError),
    ],
)
def test_granger_checks_maxlag_and_cv(setting, value, error):
    with pytest.raises(error, match=setting):
        andrey.granger(_data(), **{setting: value})


def test_granger_needs_a_time_step_in_every_fold():
    with pytest.raises(ValueError, match="data needs at least 8 rows"):
        andrey.granger(_data()[:7], maxlag=3, cv=5)
    assert andrey.granger(_data()[:8], maxlag=3, cv=5).structure.lags == (1, 2, 3)


LINGAM_PRUNED = {
    "direct_lingam": andrey.direct_lingam,
    "ica_lingam": partial(andrey.ica_lingam, random_state=0),
    "multi_group_direct_lingam": lambda x: andrey.multi_group_direct_lingam([x, x]),
}


@pytest.mark.parametrize("method", list(LINGAM_PRUNED))
def test_lingam_methods_need_one_more_row_than_variables(method):
    # The pruning step fits the last variable of the order on the other 5 with an intercept.
    with pytest.raises(
        ValueError, match=r"needs at least 7 rows \(samples\) over 6 variables, got 6"
    ):
        LINGAM_PRUNED[method](_data(6, 6))
    assert LINGAM_PRUNED[method](_data(7, 6)) is not None


@pytest.mark.parametrize("fit", PAIRWISE, ids=lambda f: f.__name__)
def test_pairwise_tests_reject_unusable_data(fit):
    with pytest.raises(ValueError, match=r"data .*row 2, column 0"):
        fit(_with(_data(d=2), (2, 0), np.nan))
    with pytest.raises(ValueError, match=r"data column 1 is constant"):
        fit(_with(_data(d=2), (slice(None), 1), 3.0))


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


def test_multi_group_direct_lingam_needs_the_same_columns_in_every_group():
    with pytest.raises(ValueError, match="data_groups needs the same columns"):
        andrey.multi_group_direct_lingam([_data(d=3), _data(d=4)])


@pytest.mark.parametrize(("n_lags", "error"), [(0, ValueError), (3, ValueError), (1.5, TypeError)])
def test_longitudinal_lingam_takes_n_lags_up_to_the_time_points_minus_one(n_lags, error):
    data = [_data() for _ in range(3)]
    with pytest.raises(error, match="n_lags"):
        andrey.longitudinal_lingam(data, n_lags=n_lags)
    assert andrey.longitudinal_lingam(data, n_lags=2).structure.n_lags == 3  # lags 0, 1, 2


@pytest.mark.parametrize(
    ("order", "error"),
    [((-1, 1), ValueError), ((0, 0), ValueError), ((1,), TypeError), ((1.0, 1), TypeError)],
)
def test_varma_lingam_checks_order(order, error):
    with pytest.raises(error, match="order"):
        andrey.varma_lingam(_data(), order=order)


@pytest.mark.parametrize(
    ("params", "error", "name"),
    [
        ({"tau_max": 0}, ValueError, "tau_max"),
        ({"tau_max": 1.5}, TypeError, "tau_max"),
        ({"pc_alpha": 0.0}, ValueError, "pc_alpha"),
        ({"pc_alpha": 1.0}, ValueError, "pc_alpha"),
        ({"alpha_level": 0.0}, ValueError, "alpha_level"),
        ({"alpha_level": 1.5}, ValueError, "alpha_level"),
        ({"indep_test": "spearman"}, ValueError, "indep_test"),
    ],
)
def test_pcmci_checks_its_parameters(params, error, name):
    with pytest.raises(error, match=name):
        andrey.pcmci(_data(), **params)


def test_pcmci_needs_enough_time_steps_for_every_test():
    # 2 * tau_max * (n_variables + 1) + 3 = 27 rows for tau_max=3 over 3 variables.
    with pytest.raises(ValueError, match="data needs at least 27 rows"):
        andrey.pcmci(_data(n=26), tau_max=3, pc_alpha=0.99)
    assert andrey.pcmci(_data(n=27), tau_max=3, pc_alpha=0.99).structure.lags == (1, 2, 3)


@pytest.mark.parametrize(
    ("params", "error", "name"),
    [
        ({"tau_max": 0}, ValueError, "tau_max"),
        ({"tau_max": 1.5}, TypeError, "tau_max"),
        ({"pc_alpha": 0.0}, ValueError, "pc_alpha"),
        ({"pc_alpha": 1.0}, ValueError, "pc_alpha"),
        ({"collider_rule": "vote"}, ValueError, "collider_rule"),
        ({"indep_test": "spearman"}, ValueError, "indep_test"),
    ],
)
def test_pcmciplus_checks_its_parameters(params, error, name):
    with pytest.raises(error, match=name):
        andrey.pcmciplus(_data(), **params)


def test_pcmciplus_needs_enough_time_steps_for_every_test():
    # 2 * tau_max * (n_variables + 1) + n_variables + 1 = 28 rows for tau_max=3 over 3 variables.
    with pytest.raises(ValueError, match="data needs at least 28 rows"):
        andrey.pcmciplus(_data(n=27), tau_max=3, pc_alpha=0.99)
    out = andrey.pcmciplus(_data(n=28), tau_max=3, pc_alpha=0.99)
    assert out.structure.lags == (0, 1, 2, 3)
    assert len(out.structure.lag(0).oriented_edges()) == 3  # nothing removed: the largest tests ran


def test_pcmciplus_takes_one_variable():
    out = andrey.pcmciplus(_data(d=1))
    assert out.structure.n_nodes == 1
    assert out.structure.lag(0).oriented_edges() == []


@pytest.mark.parametrize(
    ("params", "error", "name"),
    [
        ({"tau_max": 0}, ValueError, "tau_max"),
        ({"tau_max": 1.5}, TypeError, "tau_max"),
        ({"pc_alpha": 0.0}, ValueError, "pc_alpha"),
        ({"pc_alpha": 1.0}, ValueError, "pc_alpha"),
        ({"n_preliminary_iterations": -1}, ValueError, "n_preliminary_iterations"),
        ({"n_preliminary_iterations": 1.5}, TypeError, "n_preliminary_iterations"),
        ({"indep_test": "spearman"}, ValueError, "indep_test"),
    ],
)
def test_lpcmci_checks_its_parameters(params, error, name):
    with pytest.raises(error, match=name):
        andrey.lpcmci(_data(), **params)


def test_lpcmci_needs_enough_time_steps_for_every_test():
    # n_variables * (tau_max + 1) + 2 * tau_max + 1 = 19 rows for tau_max=3 over 3 variables.
    with pytest.raises(ValueError, match="data needs at least 19 rows"):
        andrey.lpcmci(_data(n=18), tau_max=3, pc_alpha=0.99)
    out = andrey.lpcmci(_data(n=19), tau_max=3, pc_alpha=0.99)  # keeps links: large tests run
    assert out.structure.lags == (0, 1, 2, 3)
    assert len(out.structure.lag(0).oriented_edges()) == 3


def test_lpcmci_takes_one_variable():
    out = andrey.lpcmci(_data(d=1))
    assert out.structure.n_nodes == 1
    assert out.metadata["parents"] == [[]]


def test_pcmci_takes_one_variable():
    out = andrey.pcmci(_data(d=1))
    assert out.structure.n_nodes == 1
    assert out.metadata["parents"] == [[]]


@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
def test_gin_rejects_an_alpha_outside_the_unit_interval(alpha):
    with pytest.raises(ValueError, match="alpha"):
        andrey.gin(_data(d=4), alpha=alpha)


@pytest.mark.parametrize(
    ("fit", "setting", "value", "error"),
    [
        *[(andrey.rcd, "max_explanatory_vars", bad, ValueError) for bad in (0, -1)],
        (andrey.rcd, "max_explanatory_vars", 1.5, TypeError),
        *[
            (andrey.rcd, level, bad, ValueError)
            for level in ("cor_alpha", "ind_alpha", "shapiro_alpha")
            for bad in (0.0, 1.0, 2.0)
        ],
        *[(andrey.bottom_up_parce_lingam, "alpha", bad, ValueError) for bad in (0.0, 1.0, 2.0)],
    ],
)
def test_hidden_confounder_methods_reject_a_bad_setting(fit, setting, value, error):
    with pytest.raises(error, match=setting):
        fit(_data(), **{setting: value})


def test_gin_labels_name_each_column():
    with pytest.raises(ValueError, match="labels must name each of the 4 columns"):
        andrey.gin(_data(d=4), labels=("a", "b"))


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


def test_hc_stops_at_max_iter_and_warns():
    chain = np.cumsum(np.random.default_rng(0).standard_normal((500, 8)), axis=1)
    with pytest.warns(andrey.SearchLimitWarning, match="max_iter=3"):
        out = andrey.hc(chain, max_iter=3)
    assert len(out.structure.oriented_edges()) == 3
    assert len(andrey.hc(chain).structure.oriented_edges()) == 7  # runs to a local optimum


def test_hc_sets_no_move_limit_by_default():
    # This dense graph takes over 200 moves: a limit of 200 stops the search, the default does not.
    x = sample_scm(d=30, n=800, seed=0, density=16.0).data
    with pytest.warns(andrey.SearchLimitWarning, match="max_iter=200"):
        andrey.hc(x, max_iter=200)
    with warnings.catch_warnings():
        warnings.simplefilter("error", andrey.SearchLimitWarning)
        out = andrey.hc(x)
    assert len(out.structure.oriented_edges()) > 200


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
        ({"score_func": "local_score_cond_gauss"}, "local_score_cond_gauss", NotImplementedError),
    ],
)
def test_tabu_rejects_a_bad_setting(settings, named, error):
    with pytest.raises(error, match=named):
        andrey.tabu(_data(), **settings)


def test_tabu_stops_at_max_iter_and_warns():
    chain = np.cumsum(np.random.default_rng(0).standard_normal((500, 8)), axis=1)
    with pytest.warns(andrey.SearchLimitWarning, match="max_iter=3"):
        out = andrey.tabu(chain, max_iter=3)
    assert len(out.structure.oriented_edges()) == 3


def test_cdnod_with_one_domain_gives_the_pc_graph():
    rng = np.random.default_rng(1)
    w = np.triu(rng.uniform(0.5, 1.5, (6, 6)) * (rng.random((6, 6)) < 0.4), 1)
    x = rng.standard_normal((500, 6)) @ np.linalg.inv(np.eye(6) - w)
    one_domain = andrey.cdnod(x, domain_index=np.full(500, 3.0)).structure
    assert np.array_equal(one_domain.to_numpy(), andrey.pc(x).structure.to_numpy())


def test_cdnod_rejects_nan_in_domain_index():
    with pytest.raises(ValueError, match="domain_index contains NaN"):
        andrey.cdnod(_data(), domain_index=np.r_[np.nan, np.zeros(59)])


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
    ("gfci", "indep_test", "gsq"): lambda x: andrey.gfci(
        x, indep_test="gsq", score_func="local_score_BIC_categorical"
    ),
    ("ges", "score_func", "local_score_BIC_categorical"): lambda x: andrey.ges(
        x, score_func="local_score_BIC_categorical"
    ),
    ("hc", "score_func", "local_score_BIC_categorical"): lambda x: andrey.hc(
        x, score_func="local_score_BIC_categorical"
    ),
    ("tabu", "score_func", "local_score_BDeu"): lambda x: andrey.tabu(
        x, score_func="local_score_BDeu"
    ),
    ("tabu", "score_func", "local_score_BIC_categorical"): lambda x: andrey.tabu(
        x, score_func="local_score_BIC_categorical"
    ),
    ("exact_search", "score_func", "local_score_BDeu"): lambda x: andrey.exact_search(
        x, score_func="local_score_BDeu"
    ),
    ("exact_search", "score_func", "local_score_BIC_categorical"): lambda x: andrey.exact_search(
        x, score_func="local_score_BIC_categorical"
    ),
}


def _categories(n=200, d=3):
    return np.random.default_rng(0).integers(0, 3, (n, d)).astype(np.float64)


@pytest.mark.parametrize("setting", list(CATEGORICAL), ids="-".join)
def test_a_categorical_setting_rejects_a_fractional_value(setting):
    _, option, value = setting
    with pytest.raises(ValueError, match=rf"{option}='{value}' reads data .* 0\.5 at row 7, col"):
        CATEGORICAL[setting](_with(_categories(), (7, 2), 0.5))
    assert CATEGORICAL[setting](_categories()).structure.n_nodes == 3


# Columns that hold categories as something other than whole numbers, built from whole codes.
_LEVELS = np.array(["low", "mid", "high"])
CATEGORY_COLUMNS = {
    "text": lambda pd, codes: _LEVELS[codes],
    "category of text": lambda pd, codes: pd.Categorical(_LEVELS[codes]),
    "string": lambda pd, codes: pd.array(_LEVELS[codes], dtype="string"),
    "bool": lambda pd, codes: codes > 0,
}


@pytest.mark.parametrize("kind", list(CATEGORY_COLUMNS))
@pytest.mark.parametrize("setting", list(CATEGORICAL), ids="-".join)
def test_a_categorical_setting_reads_a_column_of_categories_like_its_codes(setting, kind):
    """Each distinct value is one category, so a column of text, a pandas ``category`` or
    ``string`` column, or a boolean column gives the graph its codes give."""
    pd = pytest.importorskip("pandas")
    codes = _categories(n=400)
    codes[:, 1] = np.where(np.random.default_rng(1).random(400) < 0.8, codes[:, 0], codes[:, 1])
    if kind == "bool":
        codes[:, 0] = codes[:, 0] > 0
    frame = pd.DataFrame(codes, columns=["a", "b", "c"])
    typed = frame.assign(a=CATEGORY_COLUMNS[kind](pd, codes[:, 0].astype(int)))
    fit = CATEGORICAL[setting]
    assert np.array_equal(fit(typed).structure.to_numpy(), fit(frame).structure.to_numpy())


@pytest.mark.parametrize("setting", list(CATEGORICAL), ids="-".join)
def test_a_categorical_setting_names_a_missing_category(setting):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_categories(), columns=["a", "b", "c"])
    frame["b"] = np.where(frame["b"] > 0, "yes", None)
    with pytest.raises(ValueError, match=r"column 'b' has a missing value at row \d+"):
        CATEGORICAL[setting](frame)


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


@pytest.mark.parametrize(
    "fit", [andrey.ges, andrey.hc, andrey.tabu, partial(andrey.gfci, indep_test="chisq")]
)
def test_bdeu_still_checks_lambda_value(fit):
    """BDeu has no penalty weight, but a setting outside its range is an error under any score."""
    with pytest.raises(ValueError, match="lambda_value"):
        fit(_categories(), score_func="local_score_BDeu", lambda_value=-1.0)


def test_cdnod_with_a_categorical_test_rejects_a_fractional_index():
    c_indx = np.r_[0.5, np.zeros(99), np.ones(100)]
    with pytest.raises(ValueError, match=r"domain_index .* 0\.5 at row 0"):
        andrey.cdnod(_categories(), c_indx, indep_test="chisq")


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
    ("pc", "degen_gauss"): lambda x, **kw: andrey.pc(x, indep_test="degen_gauss", **kw),
    ("fci", "cond_gauss"): lambda x, **kw: andrey.fci(x, indep_test="cond_gauss", **kw),
    ("gfci", "degen_gauss"): lambda x, **kw: andrey.gfci(
        x, score_func="local_score_degen_gauss", indep_test="degen_gauss", **kw
    ),
    ("ges", "local_score_cond_gauss"): lambda x, **kw: andrey.ges(
        x, score_func="local_score_cond_gauss", **kw
    ),
    ("hc", "local_score_degen_gauss"): lambda x, **kw: andrey.hc(
        x, score_func="local_score_degen_gauss", **kw
    ),
    ("boss", "local_score_cond_gauss"): lambda x, **kw: andrey.boss(
        x, score_func="local_score_cond_gauss", **kw
    ),
    ("grasp", "local_score_degen_gauss"): lambda x, **kw: andrey.grasp(
        x, score_func="local_score_degen_gauss", **kw
    ),
    ("exact_search", "local_score_cond_gauss"): lambda x, **kw: andrey.exact_search(
        x, score_func="local_score_cond_gauss", **kw
    ),
}


def _mixed_data(n=200):
    """Columns: 3 categories, continuous, 2 categories."""
    rng = np.random.default_rng(0)
    return np.column_stack(
        [rng.integers(0, 3, n), rng.standard_normal(n), rng.integers(0, 2, n)]
    ).astype(np.float64)


@pytest.mark.parametrize("setting", list(MIXED), ids="-".join)
def test_a_mixed_setting_infers_whole_number_columns_and_says_so(setting):
    """Columns 0 and 2 hold three and two whole numbers; column 1 is continuous."""
    with pytest.warns(andrey.DataTypeWarning, match=r"columns 0 \(3 values\), 2 \(2 values\)"):
        out = MIXED[setting](_mixed_data())
    assert out.metadata["categorical"] == [0, 2]
    named = MIXED[setting](_mixed_data(), categorical=[0, 2])
    assert np.array_equal(named.structure.to_numpy(), out.structure.to_numpy())


@pytest.mark.parametrize("setting", list(MIXED), ids="-".join)
def test_a_mixed_setting_without_categorical_columns_warns_and_runs(setting):
    with pytest.warns(andrey.DataTypeWarning, match="with categorical=\\[\\], so it runs as the"):
        out = MIXED[setting](_mixed_data(), categorical=[])
    assert out.metadata["categorical"] == []
    continuous = np.random.default_rng(1).standard_normal((200, 3))
    with pytest.warns(andrey.DataTypeWarning, match="found no categorical column"):
        MIXED[setting](continuous)


@pytest.mark.parametrize(
    ("max_categories", "inferred"), [(2, [2]), (3, [0, 2]), (10, [0, 2])], ids=str
)
def test_max_categories_bounds_the_inferred_columns(max_categories, inferred):
    with pytest.warns(andrey.DataTypeWarning):
        out = andrey.pc(_mixed_data(), indep_test="cond_gauss", max_categories=max_categories)
    assert out.metadata["categorical"] == inferred


@pytest.mark.parametrize(("value", "error"), [(1, ValueError), (2.5, TypeError), (True, TypeError)])
def test_max_categories_is_an_int_of_at_least_two(value, error):
    with pytest.raises(error, match="max_categories"):
        andrey.pc(_mixed_data(), max_categories=value)


def test_inference_reads_every_row():
    """A column that is a code over the first rows and continuous after them is not categorical."""
    x = _mixed_data()
    x[150:, 0] += 0.5
    with pytest.warns(andrey.DataTypeWarning) as caught:
        out = andrey.pc(x, indep_test="degen_gauss")
    assert out.metadata["categorical"] == [2]
    assert "column 0" not in str(caught[0].message) and "0 (" not in str(caught[0].message)


def test_a_dataframe_types_mark_columns_and_the_warning_names_why():
    pd = pytest.importorskip("pandas")
    x = _mixed_data()
    frame = pd.DataFrame(
        {
            "group": pd.Categorical(x[:, 0].astype(int)),
            "u": x[:, 1],
            "flag": x[:, 2].astype(bool),
            "label": np.array(["p", "q"])[x[:, 2].astype(int)],
        }
    )
    with pytest.warns(andrey.DataTypeWarning) as caught:
        out = andrey.ges(frame, score_func="local_score_cond_gauss")
    message = str(caught[0].message)
    assert "'group' (category)" in message and "'flag' (bool)" in message
    assert "'label' (text)" in message
    assert out.metadata["categorical"] == ["group", "flag", "label"]


@pytest.mark.parametrize(
    "fit",
    [
        lambda x: andrey.pc(x),
        lambda x: andrey.ges(x, score_func="local_score_BIC"),
        lambda x: andrey.boss(x, score_func="local_score_BIC_from_cov"),
    ],
    ids=["pc", "ges", "boss"],
)
def test_a_continuous_setting_hints_at_columns_that_look_like_codes(fit):
    with pytest.warns(andrey.DataTypeWarning, match="columns 0, 2 hold at most 10 distinct whole"):
        out = fit(_mixed_data())
    assert "categorical" not in out.metadata


#: The score searches whose ``score_func=None`` picks BDeu on all-categorical data and, but for
#: tabu, the conditional Gaussian BIC on mixed data: each one's facade, the module and name of the
#: search it calls, and the linear-Gaussian BIC it picks otherwise.
SCORE_SEARCHES = {
    "ges": (andrey.ges, "andrey.search.ges", "ges", "local_score_BIC"),
    "hc": (andrey.hc, "andrey.search.hc", "hc", "local_score_BIC_from_cov"),
    "tabu": (andrey.tabu, "andrey.search.tabu", "tabu", "local_score_BIC_from_cov"),
    "exact_search": (andrey.exact_search, "andrey.search.exact", "exact_search", "local_score_BIC"),
    "boss": (andrey.boss, "andrey.search.boss", "boss", "local_score_BIC_from_cov"),
    "grasp": (andrey.grasp, "andrey.search.grasp", "grasp", "local_score_BIC_from_cov"),
}


def _scores_used(monkeypatch, name):
    """Record the ``score_func`` each call of ``name`` passes to its search."""
    _, module, attribute, _ = SCORE_SEARCHES[name]
    search = importlib.import_module(module)
    run = getattr(search, attribute)
    used = []

    def record(*args, **kwargs):
        used.append(kwargs["score_func"])
        return run(*args, **kwargs)

    monkeypatch.setattr(search, attribute, record)
    return used


def _categorical_data(n=500):
    """Columns: 2 categories, 3 categories, and 3 categories that depend on the first two."""
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 2, n), rng.integers(0, 3, n)
    c = np.where(rng.random(n) < 0.8, (a + b) % 3, rng.integers(0, 3, n))
    return np.column_stack([a, b, c]).astype(np.float64)


def _categorical_frame():
    """The columns of :func:`_categorical_data` as bool, category, and text columns."""
    pd = pytest.importorskip("pandas")
    x = _categorical_data().astype(int)
    return pd.DataFrame(
        {
            "a": x[:, 0].astype(bool),
            "b": pd.Categorical(x[:, 1]),
            "c": np.array(list("pqr"))[x[:, 2]],
        }
    )


def _same_fit(a, b):
    """Whether two results hold the same graph and the same metadata, the score included."""
    return (
        np.array_equal(a.structure.to_numpy(), b.structure.to_numpy()) and a.metadata == b.metadata
    )


def _switch_warnings(caught):
    return [w for w in caught if "the score defaults to" in str(w.message)]


def _mixed_frame():
    """The columns of :func:`_mixed_data` as a category, a float, and a bool column."""
    pd = pytest.importorskip("pandas")
    x = _mixed_data()
    return pd.DataFrame(
        {"group": pd.Categorical(x[:, 0].astype(int)), "u": x[:, 1], "flag": x[:, 2] > 0}
    )


#: The score searches with a mixed-data score, so a default that switches on mixed data.
MIXED_DEFAULTS = [name for name in SCORE_SEARCHES if name != "tabu"]


@pytest.mark.parametrize("name", [*SCORE_SEARCHES, "xges", "gfci"])
def test_every_score_search_takes_both_names_of_the_linear_gaussian_bic(name):
    fit = getattr(andrey, name)
    rng = np.random.default_rng(0)
    a = rng.standard_normal(300)
    b = a + rng.standard_normal(300)
    x = np.column_stack([a, b, b + rng.standard_normal(300), rng.standard_normal(300)])
    fits = [fit(x, score_func=score_func) for score_func in LINEAR_SCORES]
    assert fits[0].structure.to_numpy().any()
    assert _same_fit(*fits)


@pytest.mark.parametrize("data", [_categorical_data, _categorical_frame], ids=["codes", "frame"])
@pytest.mark.parametrize("name", list(SCORE_SEARCHES))
def test_all_categorical_data_defaults_to_bdeu_with_one_warning(name, data, monkeypatch):
    fit, _, _, linear = SCORE_SEARCHES[name]
    x = data()
    used = _scores_used(monkeypatch, name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fit(x)
        fit(x)
    assert used == ["local_score_BDeu"] * 2
    assert [w.category for w in caught] == [andrey.DataTypeWarning]
    assert linear in str(caught[0].message)
    assert _same_fit(out, fit(x, score_func="local_score_BDeu"))


@pytest.mark.parametrize("name", list(SCORE_SEARCHES))
def test_an_explicit_score_func_wins_on_categorical_and_mixed_data(name, monkeypatch):
    fit, _, _, linear = SCORE_SEARCHES[name]
    used = _scores_used(monkeypatch, name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit(_categorical_data(), score_func=linear)
        fit(_categorical_data(), score_func="local_score_BIC_categorical")
        fit(_mixed_data(), score_func=linear)
    assert used == [linear, "local_score_BIC_categorical", linear]
    assert _switch_warnings(caught) == []


@pytest.mark.parametrize("data", [_mixed_data, _mixed_frame], ids=["codes", "frame"])
@pytest.mark.parametrize("name", MIXED_DEFAULTS)
def test_mixed_data_defaults_to_the_conditional_gaussian_with_one_warning(name, data, monkeypatch):
    fit, _, _, linear = SCORE_SEARCHES[name]
    x = data()
    used = _scores_used(monkeypatch, name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fit(x)
        fit(x)
    assert used == ["local_score_cond_gauss"] * 2
    assert [w.category for w in caught] == [andrey.DataTypeWarning]
    message = str(caught[0].message)
    assert "local_score_cond_gauss" in message and linear in message
    named = fit(x, score_func="local_score_cond_gauss", categorical=[0, 2])
    assert _same_fit(out, named)
    assert len(out.metadata["categorical"]) == 2


def test_tabu_keeps_the_linear_bic_on_mixed_data(monkeypatch):
    """Tabu search has no mixed-data score, so mixed data keep the linear-Gaussian BIC."""
    used = _scores_used(monkeypatch, "tabu")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = andrey.tabu(_mixed_data())
        explicit = andrey.tabu(_mixed_data(), score_func="local_score_BIC_from_cov")
    assert used == ["local_score_BIC_from_cov"] * 2
    assert _same_fit(out, explicit)
    assert _switch_warnings(caught) == []


def _many_levels_data(n=500):
    """Columns: 12 categories, 3 categories, and 3 categories that depend on the first two."""
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 12, n), rng.integers(0, 3, n)
    c = np.where(rng.random(n) < 0.8, (a + b) % 3, rng.integers(0, 3, n))
    return np.column_stack([a, b, c]).astype(np.float64)


@pytest.mark.parametrize(
    ("data", "categorical"),
    [
        (_categorical_data, [0, 1, 2]),
        (_categorical_frame, ["a", "b", "c"]),
        (_many_levels_data, [0, 1, 2]),
    ],
    ids=["positions", "names", "beyond-max-categories"],
)
@pytest.mark.parametrize("name", MIXED_DEFAULTS)
def test_categorical_naming_every_column_defaults_to_bdeu_with_one_warning(
    name, data, categorical, monkeypatch
):
    fit, _, _, linear = SCORE_SEARCHES[name]
    x = data()
    used = _scores_used(monkeypatch, name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fit(x, categorical=categorical)
        fit(x, categorical=categorical)
    assert used == ["local_score_BDeu"] * 2
    assert [w.category for w in caught] == [andrey.DataTypeWarning]
    assert linear in str(caught[0].message)
    assert _same_fit(out, fit(x, score_func="local_score_BDeu"))


@pytest.mark.parametrize(
    ("data", "categorical"),
    [(_mixed_data, [0]), (_mixed_frame, ["group"]), (_mixed_data, [2])],
    ids=["positions", "names", "other-column"],
)
@pytest.mark.parametrize("name", MIXED_DEFAULTS)
def test_categorical_naming_some_columns_defaults_to_the_conditional_gaussian(
    name, data, categorical, monkeypatch
):
    """The named columns are the categorical ones, not the ones the inference would find."""
    fit, _, _, linear = SCORE_SEARCHES[name]
    x = data()
    used = _scores_used(monkeypatch, name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fit(x, categorical=categorical)
        fit(x, categorical=categorical)
    assert used == ["local_score_cond_gauss"] * 2
    assert [w.category for w in caught] == [andrey.DataTypeWarning]
    assert linear in str(caught[0].message)
    assert out.metadata["categorical"] == categorical
    named = fit(x, score_func="local_score_cond_gauss", categorical=categorical)
    assert _same_fit(out, named)


@pytest.mark.parametrize("name", MIXED_DEFAULTS)
def test_categorical_naming_no_column_keeps_the_linear_bic(name, monkeypatch):
    """The empty list overrides the inference: whole-number columns are numbers."""
    fit, _, _, linear = SCORE_SEARCHES[name]
    x = _categorical_data()
    used = _scores_used(monkeypatch, name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fit(x, categorical=[])
        explicit = fit(x, score_func=linear)
    assert used == [linear, linear]
    assert _same_fit(out, explicit)
    assert _switch_warnings(caught) == []


@pytest.mark.parametrize("name", MIXED_DEFAULTS)
def test_an_explicit_score_func_wins_over_categorical(name, monkeypatch):
    fit, _, _, linear = SCORE_SEARCHES[name]
    used = _scores_used(monkeypatch, name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit(_mixed_data(), score_func="local_score_degen_gauss", categorical=[0])
        fit(_categorical_data(), score_func="local_score_cond_gauss", categorical=[0, 1, 2])
    assert used == ["local_score_degen_gauss", "local_score_cond_gauss"]
    assert _switch_warnings(caught) == []
    with pytest.raises(ValueError, match="leave categorical as None"):
        fit(_mixed_data(), score_func=linear, categorical=[0])


@pytest.mark.parametrize(
    ("x", "categorical", "error", "message"),
    [
        (_mixed_frame, ["group", "zz"], ValueError, "names column 'zz', which data does not have"),
        (_mixed_data, [3], ValueError, "lists column 3, but data has 3 columns"),
        (_mixed_data, "a", TypeError, "list of column names or positions"),
        (lambda: np.ones(5), [0], ValueError, "must be a 2-D"),
    ],
    ids=["missing-name", "missing-position", "not-a-list", "not-a-table"],
)
@pytest.mark.parametrize("name", MIXED_DEFAULTS)
def test_categorical_must_list_columns_the_data_has_with_the_default_score(
    name, x, categorical, error, message
):
    fit, *_ = SCORE_SEARCHES[name]
    with pytest.raises(error, match=message):
        fit(x(), categorical=categorical)


@pytest.mark.parametrize("name", list(SCORE_SEARCHES))
def test_continuous_data_keep_the_linear_bic(name, monkeypatch):
    fit, _, _, linear = SCORE_SEARCHES[name]
    x = np.random.default_rng(1).standard_normal((200, 3))
    used = _scores_used(monkeypatch, name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fit(x)
        explicit = fit(x, score_func=linear)
    assert used == [linear, linear]
    assert _same_fit(out, explicit)
    assert caught == []


@pytest.mark.parametrize("name", list(SCORE_SEARCHES))
def test_the_default_score_and_graph_follow_the_columns_in_any_order(name, monkeypatch):
    fit, _, _, linear = SCORE_SEARCHES[name]
    order = [2, 0, 1]
    used = _scores_used(monkeypatch, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.DataTypeWarning)
        for x in (_mixed_data(), _categorical_data()):
            out, moved = fit(x), fit(x[:, order])
            assert np.array_equal(
                moved.structure.to_numpy(), out.structure.to_numpy()[order][:, order]
            )
    mixed = linear if name == "tabu" else "local_score_cond_gauss"
    assert used == [mixed, mixed, "local_score_BDeu", "local_score_BDeu"]


def test_whole_number_counts_read_as_categories_and_the_warning_names_the_bic(monkeypatch):
    """Counts of at most ``max_categories`` values meet the rule, so the warning names the BIC.

    With ``max_categories=9``, the third column's 10 values read as continuous: the data are mixed.
    """
    rng = np.random.default_rng(0)
    x = rng.binomial(4, 0.5, 500)
    y = x + rng.binomial(3, 0.5, 500)
    counts = np.column_stack([x, y, y + rng.binomial(2, 0.5, 500)]).astype(np.float64)
    assert np.unique(counts[:, 2]).size == 10
    used = _scores_used(monkeypatch, "hc")
    with pytest.warns(andrey.DataTypeWarning) as caught:
        andrey.hc(counts)
        andrey.hc(counts, score_func="local_score_BIC_from_cov")
        andrey.hc(counts, max_categories=9)
    assert "local_score_BIC_from_cov" in str(caught[0].message)
    assert used == ["local_score_BDeu", "local_score_BIC_from_cov", "local_score_cond_gauss"]


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
        andrey.pc(x, indep_test="degen_gauss", categorical=categorical)


@pytest.mark.parametrize("categorical", [["1", "2"], [0, 2], ["1", 2]])
@pytest.mark.parametrize(
    "fit",
    [partial(andrey.pc, indep_test="degen_gauss"), andrey.ges],  # ges picks its score from them
    ids=["pc", "ges"],
)
def test_categorical_reads_a_string_as_a_label_and_an_integer_as_a_position(fit, categorical):
    pd = pytest.importorskip("pandas")
    x = pd.DataFrame(_mixed_data(), columns=[1, 0, 2])  # categories in columns "1" and "2"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.DataTypeWarning)
        out = fit(x, categorical=categorical)
    assert out.metadata["categorical"] == ["1", "2"]


@pytest.mark.parametrize(
    "fit",
    [
        lambda x: andrey.pc(x, categorical=[0]),
        lambda x: andrey.pc(x, indep_test="chisq", categorical=[0]),
        lambda x: andrey.ges(x, score_func="local_score_BIC", categorical=[0]),
        lambda x: andrey.boss(x, score_func="local_score_BIC_from_cov", categorical=[0]),
        lambda x: andrey.exact_search(x, score_func="local_score_BIC", categorical=[0]),
    ],
)
def test_a_setting_that_reads_no_column_types_rejects_categorical(fit):
    with pytest.raises(ValueError, match="leave categorical as None"):
        fit(np.round(_mixed_data()))


def _normal_data(n=300):
    """Four continuous columns: a cause, its effect, a further effect, and an independent one."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(n)
    b = a + rng.standard_normal(n)
    return np.column_stack([a, b, b + rng.standard_normal(n), rng.standard_normal(n)])


def _cdnod(x, **kwargs):
    return andrey.cdnod(x, np.repeat([0.0, 1.0], len(x) // 2), **kwargs)


@pytest.mark.parametrize("empty", [[], (), np.array([], dtype=int)], ids=["list", "tuple", "array"])
@pytest.mark.parametrize(
    ("fit", "data"),
    [
        (partial(andrey.pc, indep_test="fisherz"), _normal_data),
        (partial(andrey.pc, indep_test="chisq"), _categorical_data),
        (partial(andrey.fci, indep_test="fisherz"), _normal_data),
        (partial(andrey.fci, indep_test="gsq"), _categorical_data),
        (partial(_cdnod, indep_test="fisherz"), _normal_data),
        (partial(andrey.gfci, score_func="local_score_BIC", indep_test="fisherz"), _normal_data),
        (
            partial(andrey.gfci, score_func="local_score_BDeu", indep_test="chisq"),
            _categorical_data,
        ),
        (partial(andrey.ges, score_func="local_score_BIC"), _normal_data),
        (partial(andrey.ges, score_func="local_score_BDeu"), _categorical_data),
        (partial(andrey.hc, score_func="local_score_BIC_from_cov"), _normal_data),
        (partial(andrey.hc, score_func="local_score_BIC_categorical"), _categorical_data),
        (partial(andrey.exact_search, score_func="local_score_BIC"), _normal_data),
        (partial(andrey.exact_search, score_func="local_score_BDeu"), _categorical_data),
        (partial(andrey.boss, score_func="local_score_BIC_from_cov", seed=0), _normal_data),
        (partial(andrey.boss, score_func="local_score_BDeu", seed=0), _categorical_data),
        (partial(andrey.grasp, score_func="local_score_BIC_from_cov", seed=0), _normal_data),
        (partial(andrey.grasp, score_func="local_score_BDeu", seed=0), _categorical_data),
    ],
    ids=lambda value: value.func.__name__ if isinstance(value, partial) else None,
)
def test_an_empty_categorical_reads_as_none_where_no_column_types_are_read(fit, data, empty):
    x = data()
    assert _same_fit(fit(x, categorical=empty), fit(x))


@pytest.mark.parametrize(
    "categorical", ["", "a", 0, [[]]], ids=["empty text", "text", "int", "nested"]
)
def test_a_categorical_that_is_not_an_empty_list_is_still_rejected(categorical):
    with pytest.raises(ValueError, match="leave categorical as None or"):
        andrey.pc(_normal_data(), indep_test="fisherz", categorical=categorical)


def test_a_column_that_does_not_hold_numbers_must_be_categorical():
    pd = pytest.importorskip("pandas")
    x = _mixed_data()
    frame = pd.DataFrame({"a": np.array(["p", "q", "r"])[x[:, 0].astype(int)], "b": x[:, 1]})
    frame["c"] = x[:, 2]
    with pytest.raises(ValueError, match="column 'a' does not hold numbers; list it"):
        andrey.pc(frame, indep_test="degen_gauss", categorical=["c"])
    assert andrey.pc(frame, indep_test="degen_gauss").structure.n_nodes == 3  # an object column
    frame.loc[5, "a"] = None
    with pytest.raises(ValueError, match="column 'a' has a missing value at row 5"):
        andrey.pc(frame, indep_test="degen_gauss")


@pytest.mark.parametrize(
    ("score_func", "indep_test"),
    [
        ("local_score_degen_gauss", "fisherz"),
        ("local_score_BIC", "cond_gauss"),
        ("local_score_BDeu", "degen_gauss"),
    ],
)
def test_gfci_rejects_a_mixed_score_or_test_without_its_pair(score_func, indep_test):
    with pytest.raises(ValueError, match="score and test must read the data alike"):
        andrey.gfci(_mixed_data(), score_func=score_func, indep_test=indep_test, categorical=[0])


def test_cdnod_with_a_mixed_test_takes_continuous_data_and_rejects_a_fractional_index():
    """The index is the categorical variable, so the data may have no categorical column."""
    x = _mixed_data()[:, [1]]
    c_indx = np.repeat([0.0, 1.0], 100)
    assert andrey.cdnod(x, c_indx, indep_test="degen_gauss").structure.n_nodes == 1
    with pytest.raises(ValueError, match=r"domain_index .* 0\.5 at row 0"):
        andrey.cdnod(x, np.r_[0.5, c_indx[1:]], indep_test="cond_gauss")


def _intervened(rows, column):
    mask = np.zeros((60, 3), dtype=bool)
    mask[rows, column] = True
    return mask


@pytest.mark.parametrize(
    ("interventions", "message"),
    [
        (np.zeros((60, 2)), "the shape of data"),
        (np.zeros((59, 3)), "the shape of data"),
        (np.full((60, 3), 0.5), "0/1"),
        (_intervened(slice(None), 1), "variable 1 needs at least two rows"),
        (_intervened(slice(1, None), 2), "variable 2 needs at least two rows"),
    ],
)
def test_gies_rejects_unusable_interventions(interventions, message):
    with pytest.raises(ValueError, match=message):
        andrey.gies(_data(), interventions)
