"""Facade input-validation guards: unusable inputs raise the documented error, naming the argument.

Direct Python calls (not the CLI, which validates inputs of its own) must not leak a raw
``IndexError`` / ``LinAlgError`` / ``StopIteration`` or return a result from data no method can use.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey


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
    "hc": andrey.hc,
    "exact_search": andrey.exact_search,
    "calm": andrey.calm,
    "boss": andrey.boss,
    "grasp": andrey.grasp,
    "direct_lingam": andrey.direct_lingam,
    "ica_lingam": andrey.ica_lingam,
    "varma_lingam": andrey.varma_lingam,
    "gin": andrey.gin,
}

# Data no method can fit, and what the error must say about it.
UNUSABLE = {
    "1-D": (lambda x: x[:, 0], r"data must be a 2-D"),
    "one row": (lambda x: x[:1], r"data needs at least 2 rows"),
    "no rows": (lambda x: x[:0], r"data needs at least 2 rows"),
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
    assert len(andrey.hc(chain).structure.oriented_edges()) == 7  # finished before 200 moves


def test_cdnod_with_one_domain_gives_the_pc_graph():
    rng = np.random.default_rng(1)
    w = np.triu(rng.uniform(0.5, 1.5, (6, 6)) * (rng.random((6, 6)) < 0.4), 1)
    x = rng.standard_normal((500, 6)) @ np.linalg.inv(np.eye(6) - w)
    one_domain = andrey.cdnod(x, c_indx=np.full(500, 3.0)).structure
    assert np.array_equal(one_domain.to_numpy(), andrey.pc(x).structure.to_numpy())


def test_cdnod_rejects_nan_in_c_indx():
    with pytest.raises(ValueError, match="c_indx contains NaN"):
        andrey.cdnod(_data(), c_indx=np.r_[np.nan, np.zeros(59)])
