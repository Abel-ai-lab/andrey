"""ICP recovers the causal parents of a target from environments with known truth.

The cases follow the identifiability results of Peters, Buhlmann and Meinshausen (2016), Section
4.3, on a linear Gaussian SEM over ``a, b, y, c, d`` with ``a -> b``, ``a -> y``, ``b -> y``,
``y -> c``, and ``d`` alone, the target ``y``: a single do- or noise intervention on every other
variable (Theorem 2 i and ii) and noise interventions on every other variable at once (Theorem 2
iii) each give the parents ``a`` and ``b``. A target without parents, and observational data split
in two, give no parent. An intervention on the target leaves no set invariant, unless ``targets``
names it, which leaves its rows out. The changes are strong and every environment has 500 or more
rows, so every other set is rejected; the true parents' set is accepted with probability at least
``1 - alpha``, so each case is pinned to its seed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import andrey
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output

COLUMNS = ["a", "b", "y", "c", "d"]
#: ``WEIGHTS[j][i]`` is the weight of ``COLUMNS[i] -> COLUMNS[j]``, in causal order.
WEIGHTS = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.0, 0.0, 0.0],
        [0.7, -0.9, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.8, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)
PARENTS = ["a", "b"]


def sample(rng: np.random.Generator, n: int, changes: dict[str, tuple[str, float]]) -> np.ndarray:
    """``n`` rows of the SEM, with each named variable changed.

    ``("do", m)`` sets it to ``N(m, 1)`` draws, ``("scale", f)`` multiplies its noise by ``f``, and
    ``("shift", s)`` adds ``s``.
    """
    X = np.zeros((n, len(COLUMNS)))
    for j, column in enumerate(COLUMNS):
        kind, value = changes.get(column, ("none", 0.0))
        noise = rng.standard_normal(n) * (value if kind == "scale" else 1.0)
        X[:, j] = X @ WEIGHTS[j] + noise + (value if kind == "shift" else 0.0)
        if kind == "do":
            X[:, j] = value + rng.standard_normal(n)
    return X


def environments(
    seed: int, n: int, settings: list[dict[str, tuple[str, float]]]
) -> tuple[pd.DataFrame, np.ndarray]:
    """The data of one environment per setting, ``n`` rows each, and each row's environment."""
    rng = np.random.default_rng(seed)
    X = np.vstack([sample(rng, n, changes) for changes in settings])
    return pd.DataFrame(X, columns=COLUMNS), np.repeat(np.arange(len(settings)), n)


def fit(data, env, *, target: str | int = "y", **kwargs) -> andrey.StructureOutput:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        return andrey.icp(data, env, target=target, **kwargs)


SINGLE = [{}, *({column: ("do", 2.0)} for column in ("a", "b", "c", "d"))]
NOISE = [{}, *({column: ("scale", 3.0)} for column in ("a", "b", "c", "d"))]
SIMULTANEOUS = [{}, {column: ("scale", 2.5) for column in ("a", "b", "c", "d")}]


@pytest.mark.parametrize("invariance_test", ["approximate", "exact"])
@pytest.mark.parametrize(
    ("settings", "n"),
    [
        pytest.param(SINGLE, 500, id="do on each variable"),
        pytest.param(NOISE, 500, id="noise on each variable"),
        pytest.param(SIMULTANEOUS, 1000, id="simultaneous noise"),
    ],
)
def test_identifying_interventions_recover_the_parents(settings, n, invariance_test):
    data, env = environments(0, n, settings)
    out = fit(data, env, invariance_test=invariance_test)
    assert out.metadata["parents"] == PARENTS
    assert out.structure.oriented_edges() == [("a", "y", "directed"), ("b", "y", "directed")]
    assert not out.metadata["model_rejected"]


def test_a_target_without_parents_gets_none():
    data, env = environments(0, 500, NOISE)
    data["y"] = np.random.default_rng(0).standard_normal(len(data))
    out = fit(data, env)
    assert out.metadata["parents"] == []
    assert [] in out.metadata["accepted_sets"]


def test_observational_data_split_in_two_give_no_parents():
    data, _ = environments(2, 1000, [{}])
    out = fit(data, np.arange(1000) % 2)
    assert out.metadata["parents"] == []
    assert [] in out.metadata["accepted_sets"]


def test_an_intervention_on_the_target_is_left_out_when_targets_name_it():
    data, env = environments(3, 500, [{}, {"y": ("shift", 3.0)}, *SINGLE[1:4]])
    rejected = fit(data, env)
    assert rejected.metadata["model_rejected"] and rejected.metadata["parents"] == []
    # An environment without an entry has unknown targets and stays in.
    out = fit(data, env, targets={0: [], 1: ["y"], 2: ["a"], 3: None})
    assert out.metadata["environments_left_out"] == [1]
    assert out.metadata["parents"] == PARENTS


def test_a_fit_repeats():
    data, env = environments(4, 500, SINGLE)
    first, second = fit(data, env), fit(data, env)
    assert first.metadata == second.metadata


def test_icp_keeps_the_recorded_output():
    """The observational case ``gauss_5v`` split in two blocks: x3 keeps no parent, and the record
    pins the accepted sets and the p-values."""
    target = int(datagen.column_of("gauss_5v")[3])
    out = fit(datagen.generate("gauss_5v"), datagen.domain_index("gauss_5v"), target=target)
    assert_recorded_output(out, "ICP", case="gauss_5v")
