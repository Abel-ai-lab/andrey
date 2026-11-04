"""``normalize_env_targets``: every form of ``env`` and ``targets`` reaches one normal form.

Each accepted form returns the same variables, codes, labels, and targets as the form it
describes; each form that cannot say which environment a row is in, or which variables an
environment intervened on, raises.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from andrey.core.environments import env_codes, normalize_env_targets

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


ERRORS = {
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


#: Labels of each kind, and a mapping keyed by them: an observational environment and two others.
KINDS = {
    "int": ([3, 7, 12], {3: [], 7: ["y"], 12: ["z"]}),
    "float": ([0.5, 1.0, 2.25], {0.5: [], 1.0: ["y"], 2.25: ["z"]}),
    "text": (["obs", "1", "b"], {"obs": [], "1": ["y"], "b": ["z"]}),
    "bool": ([False, True], {False: [], True: ["y"]}),
}


@pytest.mark.parametrize("kind", list(KINDS))
def test_a_mapping_written_to_json_names_the_same_environments(kind):
    """JSON writes every key as text; ``targets`` reads it back as the label it spells."""
    import json

    labels, mapping = KINDS[kind]
    env = np.resize(np.array(labels, dtype=object), N)
    text = json.loads(json.dumps(mapping))
    assert all(isinstance(key, str) for key in text)
    assert _normal(env=env, targets=text)[3] == _normal(env=env, targets=mapping)[3]


@pytest.mark.parametrize("key", ["1", "1.0", "01", "+1", "1.", "1e0", " 1"])
def test_a_text_key_names_the_number_label_it_spells(key):
    env = np.resize([0, 1], N)
    assert _normal(env=env, targets={0: [], key: ["y"]})[3] == {0: frozenset(), 1: frozenset({1})}


@pytest.mark.parametrize("key", ["nan", "inf", "true", "[1]", "one", "2"])
def test_a_text_key_that_spells_no_number_label_raises_and_names_it(key):
    with pytest.raises(ValueError, match=re.escape(f"targets names environments [{key!r}")):
        _normal(env=np.resize([0, 1], N), targets={0: [], key: ["y"]})


def test_boolean_labels_read_only_true_and_false():
    env = np.resize([False, True], N)
    assert _normal(env=env, targets={"false": [], "true": ["y"]})[3] == {
        0: frozenset(),
        1: frozenset({1}),
    }
    with pytest.raises(ValueError, match="that env does not have"):
        _normal(env=env, targets={"0": [], "1": ["y"]})


def test_a_label_beyond_2_to_the_53_matches_its_text_exactly():
    big = 2**53 + 1
    env = np.resize(np.array([big, big + 2], dtype=object), N)
    assert _normal(env=env, targets={str(big): [], str(big + 2): ["y"]})[3] == {
        0: frozenset(),
        1: frozenset({1}),
    }


def test_text_labels_are_never_read_as_numbers():
    env = np.resize(["1", "b"], N)
    assert _normal(env=env, targets={"1": [], "b": ["y"]})[3] == {
        0: frozenset(),
        1: frozenset({1}),
    }
    with pytest.raises(ValueError, match="the labels are text, so write the key '1'"):
        _normal(env=env, targets={1: [], "b": ["y"]})


@pytest.mark.parametrize(
    ("env", "targets", "group"),
    [
        ([0, 1], {0: [], 1: ["y"], "1": ["z"]}, r"\[1, '1'\] name 1"),
        ([0, 1], {0: [], "1": ["y"], "1.0": ["z"]}, r"\['1', '1.0'\] name 1"),
        ([0.0, 1.0], {0: [], 1: ["y"], "1.0": ["z"]}, r"\[1, '1.0'\] name 1.0"),
    ],
)
def test_two_keys_that_name_one_label_raise_and_name_each_group(env, targets, group):
    with pytest.raises(ValueError, match=f"more than one key: {group}"):
        _normal(env=np.resize(env, N), targets=targets)


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
