"""``normalize_env_targets``: every form of ``env`` and ``targets`` reaches one normal form.

Each accepted form returns the same variables, codes, labels, and targets as the form it
describes; each form that cannot say which environment a row is in, or which variables an
environment intervened on, raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from andrey.core.environments import env_codes, normalize_env_targets, pool_environments

N = 6
#: Each row's environment, in an order that is not sorted.
ENV = np.array(["set y", "observed", "set y", "observed", "set z", "observed"])
TARGETS = {"observed": [], "set y": ["y"], "set z": ["z"]}
#: The normal form of ``ENV`` and ``TARGETS``: codes in sorted label order, positions of x, y, z.
CODES = [1, 0, 1, 0, 2, 0]
LABELS = ["observed", "set y", "set z"]
BY_CODE = {0: frozenset(), 1: frozenset({1}), 2: frozenset({2})}
MASK = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
DATA = pd.DataFrame(np.arange(18.0).reshape(N, 3), columns=["x", "y", "z"])


def _normal(data=DATA, env=ENV, targets=TARGETS, *, known=True):
    return normalize_env_targets(data, env, targets, require_known_targets=known)


def _same_variables(X) -> bool:
    return list(X.columns) == ["x", "y", "z"] and np.array_equal(X.to_numpy(), DATA.to_numpy())


FORMS = {
    "labels and a mapping by name": lambda: _normal(),
    "a mapping by position": lambda: _normal(targets={"observed": [], "set y": [1], "set z": 2}),
    "an env column": lambda: _normal(DATA.assign(e=ENV), "e"),
    "a list of labels": lambda: _normal(env=list(ENV)),
    "a (n, 1) column of labels": lambda: _normal(env=ENV.reshape(-1, 1)),
    "a matrix with env": lambda: _normal(targets=MASK),
    "a matrix as a table with env": lambda: _normal(targets=pd.DataFrame(MASK.astype(int))),
    "a cell per row with env": lambda: _normal(targets=np.array(["y", "", "y", None, "z", np.nan])),
    "a targets column with env": lambda: _normal(
        DATA.assign(t=[["y"], [], ("y",), np.nan, {"z"}, ""]), targets="t"
    ),
    "an env column and a targets column": lambda: _normal(
        DATA.assign(e=ENV, t=["y", "", "y", "", 2, ""]), "e", "t"
    ),
}


@pytest.mark.parametrize("form", list(FORMS))
def test_each_form_reaches_the_same_normal_form(form):
    X, codes, labels, by_code = FORMS[form]()
    assert _same_variables(X)
    assert codes.dtype.kind == "i" and codes.shape == (N,) and codes.tolist() == CODES
    assert labels == LABELS and by_code == BY_CODE


def test_number_labels_come_back_as_given_in_sorted_order():
    env = np.array([2.5, 0.25, 2.5, 7.0, 0.25, 7.0])
    _, codes, labels, by_code = _normal(env=env, targets=None, known=False)
    assert codes.tolist() == [1, 0, 1, 2, 0, 2] and labels == [0.25, 2.5, 7.0]
    assert np.array_equal(np.asarray(labels)[codes], env)
    assert by_code == {0: None, 1: None, 2: None}


def test_codes_do_not_depend_on_the_row_order():
    order = np.array([4, 2, 0, 5, 1, 3])
    _, codes, labels, by_code = _normal(DATA.iloc[order], ENV[order])
    assert codes.tolist() == [CODES[i] for i in order]
    assert labels == LABELS and by_code == BY_CODE


def test_without_env_the_environments_are_the_distinct_targets():
    _, codes, labels, by_code = _normal(env=None, targets=MASK)
    assert labels == [frozenset(), frozenset({2}), frozenset({1})]  # np.unique's row order
    assert by_code == dict(enumerate(labels))
    assert [by_code[c] for c in codes] == [BY_CODE[c] for c in CODES]


def test_without_env_or_targets_the_rows_are_one_observational_environment():
    X, codes, labels, by_code = _normal(env=None, targets=None)
    assert _same_variables(X) and codes.tolist() == [0] * N
    assert labels == [frozenset()] and by_code == {0: frozenset()}


def test_an_array_comes_back_as_an_array_numbered_by_position():
    X, _, _, by_code = _normal(DATA.to_numpy(), targets={"observed": [], "set y": 1, "set z": [2]})
    assert isinstance(X, np.ndarray) and np.array_equal(X, DATA.to_numpy())
    assert by_code == BY_CODE


def test_unknown_targets_come_back_as_none_unless_they_are_required():
    partial = {"observed": [], "set y": None}
    _, _, _, by_code = _normal(targets=partial, known=False)
    assert by_code == {0: frozenset(), 1: None, 2: None}
    with pytest.raises(ValueError, match="set z"):
        _normal(targets=partial)
    with pytest.raises(ValueError):
        _normal(targets=None)


#: ``DATA``'s rows by environment, in sorted label order, and the table they stack into.
APART = {label: DATA[ENV == label] for label in LABELS}
STACKED = pd.concat(APART.values(), ignore_index=True)
BY_POSITION = {0: [], 1: ["y"], 2: [2]}

#: Each form of the environments apart, and the labels it gives them.
APART_FORMS = {
    "a mapping of DataFrames": (lambda: _normal(APART, None), LABELS),
    "a list of DataFrames": (lambda: _normal(list(APART.values()), None, BY_POSITION), [0, 1, 2]),
    "DataFrames with their columns in another order": (
        lambda: _normal(
            [t[["z", "x", "y"]] if i else t for i, t in enumerate(APART.values())],
            None,
            BY_POSITION,
        ),
        [0, 1, 2],
    ),
    "a tuple of arrays": (
        lambda: _normal(tuple(t.to_numpy() for t in APART.values()), None, {0: [], 1: 1, 2: [2]}),
        [0, 1, 2],
    ),
}


@pytest.mark.parametrize("form", list(APART_FORMS))
def test_environments_apart_reach_the_normal_form_of_their_stacked_rows(form):
    call, labels = APART_FORMS[form]
    X, codes, env_labels, by_code = call()
    assert np.array_equal(np.asarray(X), STACKED.to_numpy())
    assert list(getattr(X, "columns", ["x", "y", "z"])) == ["x", "y", "z"]
    assert codes.tolist() == sorted(CODES) and env_labels == labels and by_code == BY_CODE


def test_data_that_is_not_apart_comes_back_unchanged():
    rows = [[0.0, 1.0], [2.0, 3.0]]  # a list of rows is one table
    for data in (rows, DATA, DATA.to_numpy()):
        pooled, env = pool_environments(data, ENV)
        assert pooled is data and env is ENV


ERRORS = {
    "env with the environments apart": lambda: _normal(APART, ENV),
    "a targets column with the environments apart": lambda: _normal(APART, None, "t"),
    "a targets matrix with the environments apart": lambda: _normal(APART, None, MASK),
    "no environment": lambda: _normal([], None, None),
    "an environment without rows": lambda: _normal({**APART, "set z": DATA.iloc[:0]}, None),
    "DataFrames and arrays": lambda: _normal([DATA, DATA.to_numpy()], None, None),
    "DataFrames with other columns": lambda: _normal([DATA, DATA[["x", "y"]]], None, None),
    "arrays with other numbers of columns": lambda: _normal(
        [DATA.to_numpy(), DATA.to_numpy()[:, :2]], None, None
    ),
    "keys that mix strings and numbers": lambda: _normal({0: DATA, "a": DATA}, None, None),
    "a key that is not a string or a number": lambda: _normal({("a",): DATA}, None, None),
    "a NaN key": lambda: _normal({np.nan: DATA}, None, None),
    "an environment that is not a table": lambda: _normal({"a": DATA, "b": [0.0]}, None, None),
    "env of another length": lambda: _normal(env=ENV[1:]),
    "env of two columns": lambda: _normal(env=np.column_stack([ENV, ENV])),
    "env with None": lambda: _normal(env=np.r_[[None], ENV[1:]]),
    "env with NaN": lambda: _normal(env=[np.nan, 0, 0, 1, 1, 1], targets={0: [], 1: []}),
    "env that mixes strings and numbers": lambda: _normal(env=np.r_[[0], ENV[1:]].astype(object)),
    "env with a label that is not a string or a number": lambda: _normal(env=[b"a"] * N),
    "env that names a column data lacks": lambda: _normal(env="e"),
    "env that names a column of an array": lambda: _normal(DATA.to_numpy(), "e"),
    "env and targets that name one column": lambda: _normal(DATA.assign(e=ENV), "e", "e"),
    "a mapping without env": lambda: _normal(env=None),
    "a key env lacks": lambda: _normal(targets={**TARGETS, "set w": []}),
    "a variable data lacks": lambda: _normal(targets={**TARGETS, "set z": ["w"]}),
    "a position data lacks": lambda: _normal(targets={**TARGETS, "set z": [3]}),
    "a float position": lambda: _normal(targets={**TARGETS, "set z": [2.0]}),
    "a bool position": lambda: _normal(targets={**TARGETS, "set z": [True]}),
    "a name with no column names": lambda: _normal(DATA.to_numpy()),
    "a matrix of another shape": lambda: _normal(targets=MASK[:, :2]),
    "a matrix that is not 0/1": lambda: _normal(targets=MASK * 0.5),
    "a matrix that splits an environment": lambda: _normal(env=np.zeros(N), targets=MASK),
    "cells of another length": lambda: _normal(targets=np.array(["y", ""])),
    "cells that split an environment": lambda: _normal(targets=np.array(["y"] * 5 + [""])),
}


@pytest.mark.parametrize("case", list(ERRORS))
def test_each_unreadable_form_raises(case):
    with pytest.raises(ValueError):
        ERRORS[case]()


def test_env_codes_follow_the_sorted_labels():
    codes, labels = env_codes(["b", "a", "c", "a"], 4)
    assert codes.tolist() == [1, 0, 2, 0] and labels == ["a", "b", "c"]


@pytest.mark.parametrize(
    "env",
    [
        np.array(["b", "a", "c", "a"]),
        np.array([3, -1, 7, -1]),
        np.array([2**63 + 1, 5, 2**64 - 1, 5], dtype=np.uint64),
    ],
    ids=["text", "int", "uint"],
)
def test_a_typed_array_gives_the_codes_and_labels_of_its_list(env):
    expected_codes, expected_labels = env_codes(env.tolist(), 4)
    for shaped in (env, env.reshape(4, 1), np.asmatrix(env).T):
        codes, labels = env_codes(shaped, 4)
        assert codes.dtype == np.intp and codes.tolist() == expected_codes.tolist()
        assert labels == expected_labels
        assert [type(v) for v in labels] == [type(v) for v in expected_labels]


def test_the_docstring_example_runs():
    import doctest

    from andrey.core import environments

    assert doctest.testmod(environments).failed == 0
