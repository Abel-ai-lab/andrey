"""``andrey.ut_igsp``: its inputs, outputs, errors, and what the result does not depend on."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import andrey
import andrey.data as data
from andrey.data.interventions import hard, soft


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        yield


def sample(seed: int = 1, n: int = 500):
    """Linear-Gaussian data from an observational environment and three settings: shift ``x1``
    (known), shift ``x3`` and ``x4`` (``x3`` known), and a hard intervention on ``x5`` (unknown).

    Returns the DataFrame, the label of each row's environment, the targets mapping, and the
    true targets by column name. At the default seed the fit finds the true graph and targets.
    """
    p = 6
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=p, avg_degree=2.0),
        functional=data.functional.linear(weight_range=(0.5, 1.0)),
        noise=data.noise.gaussian(),
    )
    envs = [{}, {1: soft(shift=2.0)}, {3: soft(shift=2.0), 4: soft(shift=2.0)}, {5: hard(2.0)}]
    ds = scm.sample(n=n, seed=seed, scale="raw", envs=envs)
    names = [f"x{j}" for j in range(p)]
    frame = pd.DataFrame(ds.data, columns=names)
    labels = np.array(["observed", "one", "two", "three"])[ds.env]
    targets = {"observed": [], "one": ["x1"], "two": ["x3"], "three": None}
    truth = {"observed": [], "one": ["x1"], "two": ["x3", "x4"], "three": ["x5"]}
    return frame, labels, targets, truth


def marks(out) -> dict:
    """The result as ``{(label, label): mark}``, which no column order changes."""
    m, labels = out.structure.to_numpy(), out.structure.labels
    return {(a, b): int(m[i, j]) for i, a in enumerate(labels) for j, b in enumerate(labels)}


def test_the_targets_are_known_and_learned_keyed_by_the_labels_of_env():
    frame, env, targets, truth = sample()
    out = andrey.ut_igsp(frame, env, targets=targets)
    assert out.structure.kind == "cpdag"
    assert out.metadata["targets"] == truth
    assert out.metadata["invariance_test"] == "gaussian"
    assert out.metadata["indep_test"] == "fisherz"
    edges = int((out.structure.to_numpy() != 0).sum() // 2)
    assert out.metadata["score"] == edges + sum(len(t) for t in truth.values())
    assert out.metadata["seed"] == 0


def test_with_no_known_targets_the_mapping_names_unknown_settings():
    frame, env, _, truth = sample()
    out = andrey.ut_igsp(frame, env, targets={"observed": []})
    assert out.metadata["targets"] == truth


def test_every_form_of_env_and_targets_gives_one_result_when_the_known_targets_differ():
    frame, env, _, _ = sample()
    known = {"observed": [], "one": ["x1"], "two": ["x3"], "three": ["x5"]}
    mapping = andrey.ut_igsp(frame, env, targets=known)
    cells = [known[label] for label in env]
    column = andrey.ut_igsp(frame.assign(t=cells, e=env), "e", targets="t")
    mask = np.zeros(frame.shape, dtype=int)
    for i, variables in enumerate(cells):
        mask[i, [int(v[1:]) for v in variables]] = 1
    alone = andrey.ut_igsp(frame, targets=mask)
    with_env = andrey.ut_igsp(frame, env, targets=mask)
    for out in (column, alone, with_env):
        assert marks(out) == marks(mapping)
    assert with_env.metadata["targets"] == mapping.metadata["targets"]
    # Without env, each environment is named by its known targets.
    assert alone.metadata["targets"] == {"": [], "x1": ["x1"], "x3": ["x3", "x4"], "x5": ["x5"]}


def test_a_numbered_array_names_targets_by_position():
    frame, env, _, truth = sample()
    out = andrey.ut_igsp(frame.to_numpy(), env, targets={"observed": [], "one": [1], "two": [3]})
    assert out.metadata["targets"] == {
        k: [int(v[1:]) for v in variables] for k, variables in truth.items()
    }


@pytest.mark.parametrize(
    ("env", "targets", "message"),
    [
        (None, None, "at least one interventional environment"),
        ("one", {"a": []}, "at least one interventional environment"),
        ("one", None, "needs an observational environment"),
        ("one", {"a": ["x0"]}, "needs an observational environment"),
        ("one", {"zz": []}, "that env does not have"),
        ("small", {"a": []}, "at least 3 rows in every environment"),
    ],
)
def test_bad_environments_raise(env, targets, message):
    frame, _, _, _ = sample(n=50)
    n = len(frame)
    rows = {"one": np.full(n, "a"), "small": np.array(["a"] * (n - 2) + ["b", "b"]), None: None}
    with pytest.raises(ValueError, match=message):
        andrey.ut_igsp(frame, rows[env], targets=targets)


def test_a_column_constant_on_the_observational_rows_raises():
    frame, env, targets, _ = sample(n=100)
    frame.loc[env == "observed", "x2"] = 1.0
    with pytest.raises(ValueError, match=r"columns \['x2'\] are constant on the observational"):
        andrey.ut_igsp(frame, env, targets=targets)


def test_a_column_constant_within_a_setting_is_read():
    frame, env, targets, _ = sample()
    frame.loc[env == "three", "x5"] = 2.0  # a hard intervention without noise
    out = andrey.ut_igsp(frame, env, targets=targets)
    assert "x5" in out.metadata["targets"]["three"]


def test_invariance_alpha_reaches_the_invariance_test_and_alpha_does_not(monkeypatch):
    """Under an invariance test whose p-value is always 0.01, every variable is a target at a
    level above it and only the known ones below it."""
    from andrey.search import igsp

    monkeypatch.setattr(igsp, "make_invariance_test", lambda *args, **kwargs: lambda y, S: 0.01)
    frame, env, targets, _ = sample()
    every = {"observed": [], **dict.fromkeys(("one", "two", "three"), list(frame.columns))}
    known = {"observed": [], "one": ["x1"], "two": ["x3"], "three": []}
    for levels, want in (
        ({"invariance_alpha": 0.05}, every),
        ({"invariance_alpha": 0.001}, known),
        ({"alpha": 0.05}, every),
        ({"alpha": 0.001}, known),
        ({"alpha": 0.05, "invariance_alpha": 0.001}, known),
    ):
        assert andrey.ut_igsp(frame, env, targets=targets, **levels).metadata["targets"] == want


@pytest.mark.parametrize("indep_test", ["fisherz", "kci"])
def test_row_order_environment_labels_and_split_observational_rows_change_nothing(indep_test):
    frame, env, targets, _ = sample(n=150 if indep_test == "kci" else 500)
    kwargs = {"indep_test": indep_test, "n_starts": 2}
    want = andrey.ut_igsp(frame, env, targets=targets, **kwargs)
    order = np.random.default_rng(1).permutation(len(frame))
    shuffled = andrey.ut_igsp(frame.iloc[order], env[order], targets=targets, **kwargs)
    assert marks(shuffled) == marks(want)
    assert shuffled.metadata["targets"] == want.metadata["targets"]
    renamed = {"observed": "z", "one": "y", "two": "x", "three": "w"}  # reverses the sorted order
    relabeled = andrey.ut_igsp(
        frame,
        np.array([renamed[label] for label in env]),
        targets={renamed[k]: v for k, v in targets.items()},
        **kwargs,
    )
    assert marks(relabeled) == marks(want)
    assert relabeled.metadata["targets"] == {
        renamed[k]: v for k, v in want.metadata["targets"].items()
    }
    split = np.where((env == "observed") & (np.arange(len(env)) % 2 == 1), "observed 2", env)
    two = andrey.ut_igsp(frame, split, targets={**targets, "observed 2": []}, **kwargs)
    assert marks(two) == marks(want)


@pytest.mark.parametrize("n_starts", [1, 5])
def test_reordering_the_columns_relabels_the_graph_and_the_targets(n_starts):
    frame, env, targets, _ = sample(seed=3)
    want = andrey.ut_igsp(frame, env, targets=targets, n_starts=n_starts, seed=4)
    for order in ([5, 4, 3, 2, 1, 0], [2, 0, 5, 1, 3, 4]):
        columns = [frame.columns[k] for k in order]
        out = andrey.ut_igsp(frame[columns], env, targets=targets, n_starts=n_starts, seed=4)
        assert marks(out) == marks(want)
        assert {k: sorted(v) for k, v in out.metadata["targets"].items()} == {
            k: sorted(v) for k, v in want.metadata["targets"].items()
        }


def test_a_seed_repeats_and_one_start_does_not_depend_on_it(monkeypatch):
    frame, env, targets, _ = sample(seed=5)
    first = andrey.ut_igsp(frame, env, targets=targets, seed=3)
    assert marks(andrey.ut_igsp(frame, env, targets=targets, seed=3)) == marks(first)
    monkeypatch.setenv("ANDREY_SEED", "3")
    assert andrey.ut_igsp(frame, env, targets=targets).metadata["seed"] == 3
    single = [andrey.ut_igsp(frame, env, targets=targets, n_starts=1, seed=s) for s in range(5)]
    assert all(marks(out) == marks(single[0]) for out in single)
    assert "seed" not in single[0].metadata


def test_knowledge_keeps_required_edges_drops_forbidden_ones_and_orients_by_it():
    frame, env, targets, _ = sample()
    base = andrey.ut_igsp(frame, env, targets=targets)
    present = next(e for e in base.structure.oriented_edges())
    absent = next(
        (a, b)
        for a in frame.columns
        for b in frame.columns
        if a < b and not any({a, b} == {e[0], e[1]} for e in base.structure.oriented_edges())
    )
    knowledge = andrey.Knowledge(required=[absent], forbidden=[present[:2]])
    out = andrey.ut_igsp(frame, env, targets=targets, knowledge=knowledge)
    edges = {(a, b): kind for a, b, kind in out.structure.oriented_edges()}
    assert edges.get(absent) == "directed"
    assert present[:2] not in edges


def test_the_indicator_form_with_a_categorical_test_reads_categories():
    rng = np.random.default_rng(0)
    n = 1500
    env = np.repeat(["observed", "set b"], n)
    a = rng.integers(0, 3, 2 * n)
    b = np.where(rng.random(2 * n) < 0.8, a, rng.integers(0, 3, 2 * n))
    b[n:] = rng.choice(3, size=n, p=[0.7, 0.2, 0.1])  # set from outside, with a new marginal
    c = np.where(rng.random(2 * n) < 0.8, b, rng.integers(0, 3, 2 * n))
    table = pd.DataFrame({k: pd.Categorical(v) for k, v in zip("abc", (a, b, c), strict=True)})
    out = andrey.ut_igsp(table, env, targets={"observed": [], "set b": None})
    assert out.metadata["indep_test"] == "gsq"
    assert out.metadata["invariance_test"] == "indicator"
    assert out.metadata["targets"]["set b"] == ["b"]
    assert ("a", "b", "directed") in out.structure.oriented_edges()
