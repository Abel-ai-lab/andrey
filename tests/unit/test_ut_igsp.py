"""``andrey.ut_igsp``: its inputs, outputs, errors, and what the result does not depend on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import andrey
import andrey.data as data
from andrey.data.interventions import hard, soft


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


NUMBER = {"observed": 0, "one": 1, "two": 2, "three": 3}


def test_a_numbered_array_names_targets_by_position_and_environments_by_their_labels():
    frame, env, _, truth = sample()
    codes = np.array([NUMBER[label] for label in env])
    out = andrey.ut_igsp(frame.to_numpy(), codes, targets={0: [], 1: [1], 2: [3]})
    assert out.metadata["targets"] == {
        NUMBER[k]: [int(v[1:]) for v in variables] for k, variables in truth.items()
    }
    assert [type(k) for k in out.metadata["targets"]] == [int] * 4


def test_the_targets_pass_back_as_given_after_save_load_and_json_alike(tmp_path):
    """Passed back, the targets are known: the refit keeps them and may learn more, so it need not
    repeat the fit. The Python keys, the keys ``load`` gives, and the keys of a JSON pass name the
    same environments, so their refits agree; ``gies`` takes the mapping too."""
    import json

    frame, env, _, _ = sample()
    X, codes = frame.to_numpy(), np.array([NUMBER[label] for label in env])
    out = andrey.ut_igsp(X, codes, targets={0: [], 1: [1], 2: [3]})
    assert all(out.metadata["targets"][k] for k in (1, 2, 3))  # every setting has a target
    out.save(tmp_path / "out.json")
    loaded = andrey.StructureOutput.load(tmp_path / "out.json").metadata["targets"]
    passed = json.loads(json.dumps(dict(out.metadata)))["targets"]
    assert set(loaded) == set(passed) == {"0", "1", "2", "3"}
    refits = [
        andrey.ut_igsp(X, codes, targets=given)
        for given in (out.metadata["targets"], loaded, passed)
    ]
    for refit in refits:
        assert np.array_equal(refit.structure.to_numpy(), refits[0].structure.to_numpy())
        assert refit.metadata["targets"] == refits[0].metadata["targets"]
        for k, variables in out.metadata["targets"].items():
            assert set(variables) <= set(refit.metadata["targets"][k])
    by_gies = [andrey.gies(X, codes, targets=given) for given in (out.metadata["targets"], loaded)]
    assert np.array_equal(by_gies[0].structure.to_numpy(), by_gies[1].structure.to_numpy())


def test_a_setting_with_no_target_found_passes_back_as_observational():
    """A setting identical to the observational one maps to ``[]``, which ``targets`` reads as
    observational; with every setting so, the refit has no interventional environment."""
    frame, env, _, _ = sample()
    X = frame.to_numpy()[env == "observed"]  # setting 1: the second half of the observational rows
    codes = np.repeat([0, 1], [len(X) // 2, len(X) - len(X) // 2])
    out = andrey.ut_igsp(X, codes, targets={0: [], 1: None})
    assert out.metadata["targets"] == {0: [], 1: []}
    with pytest.raises(ValueError, match="at least one interventional environment"):
        andrey.ut_igsp(X, codes, targets=out.metadata["targets"])


def test_bootstrap_fits_kept_hold_the_labels_as_keys():
    frame, env, _, _ = sample(n=200)
    X, codes = frame.to_numpy(), np.array([NUMBER[label] for label in env])
    out = andrey.bootstrap(
        andrey.ut_igsp, X, env=codes, targets={0: [], 1: [1]}, n_resamples=2, keep_fits=True
    )
    for fit in out.fits:
        assert set(fit.metadata["targets"]) == {0, 1, 2, 3}


def test_without_env_two_environments_with_one_key_raise():
    frame, env, _, _ = sample(n=50)
    frame = frame.rename(columns={"x1": "a,b", "x3": "a", "x4": "b"})
    known = {"observed": [], "one": ["a,b"], "two": ["a", "b"], "three": ["x5"]}
    with pytest.raises(ValueError, match="names more than one"):
        andrey.ut_igsp(frame.assign(t=[known[label] for label in env]), targets="t")


@pytest.mark.parametrize("alpha", [None, 0.05])
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
def test_bad_environments_raise(env, targets, message, alpha):
    frame, _, _, _ = sample(n=50)
    n = len(frame)
    rows = {"one": np.full(n, "a"), "small": np.array(["a"] * (n - 2) + ["b", "b"]), None: None}
    with pytest.raises(ValueError, match=message):
        andrey.ut_igsp(frame, rows[env], targets=targets, alpha=alpha)


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


def test_alpha_none_is_one_over_the_observational_rows_and_a_float_is_kept():
    frame, env, targets, _ = sample()
    rows = int((env == "observed").sum())
    default = andrey.ut_igsp(frame, env, targets=targets)
    explicit = andrey.ut_igsp(frame, env, targets=targets, alpha=1 / rows)
    assert default.metadata["alpha"] == explicit.metadata["alpha"] == 1 / rows
    assert marks(default) == marks(explicit)
    assert default.metadata["targets"] == explicit.metadata["targets"]
    assert andrey.ut_igsp(frame, env, targets=targets, alpha=0.01).metadata["alpha"] == 0.01
    # Every environment mapped to [] is observational: their rows are pooled and all counted.
    split = np.where((env == "observed") & (np.arange(len(env)) % 2 == 0), "again", env)
    pooled = andrey.ut_igsp(frame, split, targets={**targets, "again": []})
    assert pooled.metadata["alpha"] == 1 / rows


def test_alpha_none_sets_the_invariance_level_by_the_observational_rows(monkeypatch):
    """Under an invariance test whose p-value is always 0.01, every variable is a target at 50
    observational rows, a level of 0.02, and only the known ones at 500, a level of 0.002."""
    from andrey.search import igsp

    monkeypatch.setattr(igsp, "make_invariance_test", lambda *args, **kwargs: lambda y, S: 0.01)
    for n, every in ((50, True), (500, False)):
        frame, env, targets, _ = sample(n=n)
        out = andrey.ut_igsp(frame, env, targets=targets)
        assert out.metadata["alpha"] == 1 / n
        found = out.metadata["targets"]
        if every:
            assert all(found[k] == list(frame.columns) for k in ("one", "two", "three"))
        else:
            assert found == {"observed": [], "one": ["x1"], "two": ["x3"], "three": []}


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


def test_without_env_reordering_the_columns_keeps_the_keys():
    frame, env, _, _ = sample(seed=3)
    known = {"observed": [], "one": ["x1"], "two": ["x3", "x4"], "three": ["x5"]}
    cells = [known[label] for label in env]
    want = andrey.ut_igsp(frame.assign(t=cells), targets="t", n_starts=1)
    assert set(want.metadata["targets"]) == {"", "x1", "x3,x4", "x5"}
    for order in ([5, 4, 3, 2, 1, 0], [2, 0, 5, 1, 3, 4]):
        columns = [frame.columns[k] for k in order]
        out = andrey.ut_igsp(frame[columns].assign(t=cells), targets="t", n_starts=1)
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


def test_the_indicator_form_with_a_categorical_test_reads_categories_whatever_their_codes():
    rng = np.random.default_rng(0)
    n = 1500
    env = np.repeat(["observed", "set b"], n)
    targets = {"observed": [], "set b": None}
    a = rng.integers(0, 3, 2 * n)
    b = np.where(rng.random(2 * n) < 0.8, a, rng.integers(0, 3, 2 * n))
    b[n:] = rng.choice(3, size=n, p=[0.7, 0.2, 0.1])  # set from outside, with a new marginal
    c = np.where(rng.random(2 * n) < 0.8, b, rng.integers(0, 3, 2 * n))

    def table(codes: list[int]) -> pd.DataFrame:
        """The columns as categories, category ``k`` coded ``codes[k]``."""
        return pd.DataFrame(
            {k: pd.Categorical(np.asarray(codes)[v]) for k, v in zip("abc", (a, b, c), strict=True)}
        )

    out = andrey.ut_igsp(table([0, 1, 2]), env, targets=targets)
    assert out.metadata["indep_test"] == "gsq"
    assert out.metadata["invariance_test"] == "indicator"
    assert out.metadata["targets"]["set b"] == ["b"]
    assert ("a", "b", "directed") in out.structure.oriented_edges()
    for codes in ([2, 0, 1], [1, 2, 0]):
        recoded = andrey.ut_igsp(table(codes), env, targets=targets)
        assert marks(recoded) == marks(out)
        assert recoded.metadata["targets"] == out.metadata["targets"]
    # The Gaussian form would regress on the codes, so it refuses the categorical tests.
    for indep_test in ("gsq", "cond_gauss"):
        with pytest.raises(ValueError, match="invariance_test='gaussian'"):
            andrey.ut_igsp(
                table([0, 1, 2]),
                env,
                targets=targets,
                indep_test=indep_test,
                invariance_test="gaussian",
            )
