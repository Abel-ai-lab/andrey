"""``andrey.igsp`` and ``andrey.ut_igsp``: their inputs, outputs, errors, and what the result does
not depend on.

The two facades read environments, levels, and seeds by the same rules, so one test covers both
where they share a rule; each case observes its own method, IGSP through the answers of its
invariance test and UT-IGSP through the targets it learns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import andrey
import andrey.data as data
from andrey.core.environments import normalize_env_targets
from andrey.data.interventions import hard, soft
from andrey.search import igsp
from andrey.search.gies import cut_pairs, replace_unprotected

METHODS = ["igsp", "ut_igsp"]
#: The variables each setting of :func:`sample` intervenes on.
TRUE_TARGETS = {"observed": [], "one": ["x1"], "two": ["x3", "x4"], "three": ["x5"]}
#: The targets each method is given: IGSP every one; UT-IGSP ``x1``, ``x3`` of ``x3`` and ``x4``,
#: and none of the third setting's.
TARGETS = {
    "igsp": TRUE_TARGETS,
    "ut_igsp": {"observed": [], "one": ["x1"], "two": ["x3"], "three": None},
}
#: The seed at which each method's fit finds the true graph, and UT-IGSP the true targets.
SEED = {"igsp": 2, "ut_igsp": 1}


def sample(method: str, seed: int | None = None, n: int = 500):
    """Linear-Gaussian data from an observational environment and three settings: shift ``x1``,
    shift ``x3`` and ``x4``, and a hard intervention on ``x5``.

    Returns the DataFrame, the label of each row's environment, the method's targets mapping, and
    the dataset.
    """
    p = 6
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=p, avg_degree=2.0),
        functional=data.functional.linear(weight_range=(0.5, 1.0)),
        noise=data.noise.gaussian(),
    )
    envs = [{}, {1: soft(shift=2.0)}, {3: soft(shift=2.0), 4: soft(shift=2.0)}, {5: hard(2.0)}]
    ds = scm.sample(n=n, seed=SEED[method] if seed is None else seed, scale="raw", envs=envs)
    frame = pd.DataFrame(ds.data, columns=[f"x{j}" for j in range(p)])
    labels = np.array(["observed", "one", "two", "three"])[ds.env]
    return frame, labels, dict(TARGETS[method]), ds


def essential(dag: np.ndarray, family: list[set[int]]) -> np.ndarray:
    rows = np.zeros((len(family) + 1, dag.shape[0]), dtype=bool)
    for row, variables in enumerate(family, start=1):
        rows[row, sorted(variables)] = True
    return replace_unprotected(dag.astype(np.int8), cut_pairs(rows))


def marks(out) -> dict:
    """The result as ``{(label, label): mark}``, which no column order changes."""
    m, labels = out.structure.to_numpy(), out.structure.labels
    return {(a, b): int(m[i, j]) for i, a in enumerate(labels) for j, b in enumerate(labels)}


# --- igsp -----------------------------------------------------------------------------------------


def test_the_result_is_the_true_class_with_the_edges_and_contradicted_edges_found():
    frame, env, targets, ds = sample("igsp")
    out = andrey.igsp(frame, env, targets=targets)
    assert out.structure.kind == "cpdag"
    truth = essential(ds.graph.to_numpy(), [set(ds.targets[k]) for k in (1, 2, 3)])
    assert np.array_equal(out.structure.to_numpy(), truth)
    assert out.metadata["score"] == int((truth != 0).sum() // 2)
    assert out.metadata["n_contradicting"] == 0
    assert out.metadata["indep_test"] == "fisherz"
    assert out.metadata["invariance_test"] == "gaussian"
    assert out.metadata["seed"] == 0
    assert "targets" not in out.metadata


def test_every_form_of_env_and_targets_gives_one_result():
    frame, env, targets, _ = sample("igsp")
    mapping = andrey.igsp(frame, env, targets=targets)
    cells = [targets[label] for label in env]
    column = andrey.igsp(frame.assign(t=cells, e=env), "e", targets="t")
    mask = np.zeros(frame.shape, dtype=int)
    for i, variables in enumerate(cells):
        mask[i, [int(v[1:]) for v in variables]] = 1
    alone = andrey.igsp(frame, targets=mask)
    with_env = andrey.igsp(frame, env, targets=mask)
    numbered = andrey.igsp(
        frame.to_numpy(),
        env,
        targets={k: [frame.columns.get_loc(v) for v in t] for k, t in targets.items()},
    )
    for out in (column, alone, with_env):
        assert marks(out) == marks(mapping)
    assert np.array_equal(numbered.structure.to_numpy(), mapping.structure.to_numpy())


def test_environments_with_one_target_set_are_two_settings_with_env_and_one_without():
    """Two groups of rows shift ``x1``: with ``env`` and a mapping they are two settings, and with
    the mask alone one; each result is the engine's on that list of settings."""
    frame, env, _, _ = sample("igsp", n=300)
    env = np.where(env == "two", "one b", env)
    targets = {"observed": [], "one": ["x1"], "one b": ["x1"], "three": ["x5"]}
    cells = [targets[label] for label in env]
    mask = np.zeros(frame.shape, dtype=int)
    for i, variables in enumerate(cells):
        mask[i, [int(v[1:]) for v in variables]] = 1
    settings = {"alpha": 1e-3, "invariance_alpha": 1e-3, "indep_test": "fisherz"}
    settings |= {"invariance_test": "gaussian", "categorical": None, "n_starts": 10, "seed": 0}
    for given, n_settings in (((env, targets), 3), ((None, mask), 2)):
        X, codes, _, by_code = normalize_env_targets(
            frame, *given, require_known_targets=True, method="igsp"
        )
        assert sum(t != frozenset() for t in by_code.values()) == n_settings
        graph, _ = igsp.igsp(np.asarray(X, dtype=float), codes, by_code, **settings)
        out = andrey.igsp(frame, given[0], targets=given[1])
        assert np.array_equal(out.structure.to_numpy(), graph.to_numpy())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"indep_test": "nope"}, "indep_test"),
        ({"alpha": 0.0}, "alpha must be"),
        ({"alpha": 1.0}, "alpha must be"),
        ({"invariance_alpha": 1.5}, "invariance_alpha must be"),
        ({"invariance_test": "nope"}, "invariance_test must be"),
        ({"n_starts": 0}, "n_starts must be"),
        ({"categorical": ["x0"], "indep_test": "fisherz"}, "categorical"),
        ({"indep_test": "gsq", "invariance_test": "gaussian"}, "invariance_test='gaussian'"),
    ],
)
def test_bad_arguments_raise(kwargs, message):
    frame, env, targets, _ = sample("igsp", n=50)
    with pytest.raises(ValueError, match=message):
        andrey.igsp(frame, env, targets=targets, **kwargs)


def test_a_missing_value_or_a_column_constant_on_the_observational_rows_raises():
    frame, env, targets, _ = sample("igsp", n=100)
    gap = frame.copy()
    gap.iloc[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN|missing"):
        andrey.igsp(gap, env, targets=targets)
    frame.loc[env == "observed", "x2"] = 1.0
    with pytest.raises(ValueError, match=r"columns \['x2'\] are constant on the observational"):
        andrey.igsp(frame, env, targets=targets)


@pytest.mark.parametrize("seed", range(5))
def test_reordering_the_columns_relabels_the_graph(seed):
    rng = np.random.default_rng(seed)
    for dataset in range(20):
        frame, env, targets, _ = sample("igsp", seed=dataset, n=300)
        want = marks(andrey.igsp(frame, env, targets=targets, seed=seed))
        columns = list(rng.permutation(frame.columns))
        assert marks(andrey.igsp(frame[columns], env, targets=targets, seed=seed)) == want


#: The edge 0 -> 1 with the variable 2 alone, and a chain 0 -> 1 -> 2.
EDGE = data.graphs.FixedDAG(nodes=("0", "1", "2"), arcs=(("0", "1"),))

CHAIN = data.graphs.FixedDAG(nodes=("0", "1", "2"), arcs=(("0", "1"), ("1", "2")))

INTERVENTIONS = {"hard": hard(0.0, sd=0.2**0.5), "shift": soft(shift=2.0)}


def fixed(graph, family: list[int], kind: str, seed: int):
    scm = data.SCM(
        graph=graph,
        functional=data.functional.linear(weight_range=(0.5, 1.0)),
        noise=data.noise.gaussian(),
    )
    setting = dict.fromkeys(family, INTERVENTIONS[kind])
    ds = scm.sample(n=[2000] * 2, seed=seed, scale="raw", envs=[{}, setting], shuffle=False)
    return andrey.igsp(ds.data, ds.env, targets={0: [], 1: family})


@pytest.mark.parametrize("kind", list(INTERVENTIONS))
@pytest.mark.parametrize("family", [[0], [0, 2]])
def test_an_intervention_on_a_cause_orients_its_edge_away_from_it(family, kind):
    """With the settings {0} and {0, 2}, the general definition read as in the paper orients
    0 -> 1 on every seed."""
    for seed in range(20):
        out = fixed(EDGE, family, kind, seed)
        assert out.structure.oriented_edges() == [(0, 1, "directed")], seed


@pytest.mark.parametrize("kind", list(INTERVENTIONS))
def test_a_chain_intervened_in_the_middle_is_fully_oriented(kind):
    out = fixed(CHAIN, [1], kind, 0)
    assert out.structure.oriented_edges() == [(0, 1, "directed"), (1, 2, "directed")]


# --- ut_igsp --------------------------------------------------------------------------------------


def test_the_targets_are_known_and_learned_keyed_by_the_labels_of_env():
    frame, env, targets, _ = sample("ut_igsp")
    out = andrey.ut_igsp(frame, env, targets=targets)
    assert out.structure.kind == "cpdag"
    assert out.metadata["targets"] == TRUE_TARGETS
    assert out.metadata["invariance_test"] == "gaussian"
    assert out.metadata["indep_test"] == "fisherz"
    edges = int((out.structure.to_numpy() != 0).sum() // 2)
    assert out.metadata["score"] == edges + sum(len(t) for t in TRUE_TARGETS.values())
    assert out.metadata["seed"] == 0


def test_with_no_known_targets_the_mapping_names_unknown_settings():
    frame, env, _, _ = sample("ut_igsp")
    out = andrey.ut_igsp(frame, env, targets={"observed": []})
    assert out.metadata["targets"] == TRUE_TARGETS


def test_every_form_of_env_and_targets_gives_one_result_when_the_known_targets_differ():
    frame, env, _, _ = sample("ut_igsp")
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
    frame, env, _, _ = sample("ut_igsp")
    codes = np.array([NUMBER[label] for label in env])
    out = andrey.ut_igsp(frame.to_numpy(), codes, targets={0: [], 1: [1], 2: [3]})
    assert out.metadata["targets"] == {
        NUMBER[k]: [int(v[1:]) for v in variables] for k, variables in TRUE_TARGETS.items()
    }
    assert [type(k) for k in out.metadata["targets"]] == [int] * 4


def test_the_targets_pass_back_as_given_after_save_load_and_json_alike(tmp_path):
    """Passed back, the targets are known: the refit keeps them and may learn more, so it need not
    repeat the fit. The Python keys, the keys ``load`` gives, and the keys of a JSON pass name the
    same environments, so their refits agree; ``gies`` takes the mapping too."""
    import json

    frame, env, _, _ = sample("ut_igsp")
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
    frame, env, _, _ = sample("ut_igsp")
    X = frame.to_numpy()[env == "observed"]  # setting 1: the second half of the observational rows
    codes = np.repeat([0, 1], [len(X) // 2, len(X) - len(X) // 2])
    out = andrey.ut_igsp(X, codes, targets={0: [], 1: None})
    assert out.metadata["targets"] == {0: [], 1: []}
    with pytest.raises(ValueError, match="at least one interventional environment"):
        andrey.ut_igsp(X, codes, targets=out.metadata["targets"])


def test_bootstrap_fits_kept_hold_the_labels_as_keys():
    frame, env, _, _ = sample("ut_igsp", n=200)
    X, codes = frame.to_numpy(), np.array([NUMBER[label] for label in env])
    out = andrey.bootstrap(
        andrey.ut_igsp, X, env=codes, targets={0: [], 1: [1]}, n_resamples=2, keep_fits=True
    )
    for fit in out.fits:
        assert set(fit.metadata["targets"]) == {0, 1, 2, 3}


def test_without_env_two_environments_with_one_key_raise():
    frame, env, _, _ = sample("ut_igsp", n=50)
    frame = frame.rename(columns={"x1": "a,b", "x3": "a", "x4": "b"})
    known = {"observed": [], "one": ["a,b"], "two": ["a", "b"], "three": ["x5"]}
    with pytest.raises(ValueError, match="names more than one"):
        andrey.ut_igsp(frame.assign(t=[known[label] for label in env]), targets="t")


def test_a_column_constant_on_the_observational_rows_raises():
    frame, env, targets, _ = sample("ut_igsp", n=100)
    frame.loc[env == "observed", "x2"] = 1.0
    with pytest.raises(ValueError, match=r"columns \['x2'\] are constant on the observational"):
        andrey.ut_igsp(frame, env, targets=targets)


def test_a_column_constant_within_a_setting_is_read():
    frame, env, targets, _ = sample("ut_igsp")
    frame.loc[env == "three", "x5"] = 2.0  # a hard intervention without noise
    out = andrey.ut_igsp(frame, env, targets=targets)
    assert "x5" in out.metadata["targets"]["three"]


@pytest.mark.parametrize("n_starts", [1, 5])
def test_reordering_the_columns_relabels_the_graph_and_the_targets(n_starts):
    frame, env, targets, _ = sample("ut_igsp", seed=3)
    want = andrey.ut_igsp(frame, env, targets=targets, n_starts=n_starts, seed=4)
    for order in ([5, 4, 3, 2, 1, 0], [2, 0, 5, 1, 3, 4]):
        columns = [frame.columns[k] for k in order]
        out = andrey.ut_igsp(frame[columns], env, targets=targets, n_starts=n_starts, seed=4)
        assert marks(out) == marks(want)
        assert {k: sorted(v) for k, v in out.metadata["targets"].items()} == {
            k: sorted(v) for k, v in want.metadata["targets"].items()
        }


def test_without_env_reordering_the_columns_keeps_the_keys():
    frame, env, _, _ = sample("ut_igsp", seed=3)
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


# --- what igsp and ut_igsp share ------------------------------------------------------------------

#: Environments a method cannot read, and its error.
BAD_ENVIRONMENTS = {
    "igsp": [
        (None, None, "at least one interventional environment.*use grasp or boss"),
        ("one", {"a": []}, "at least one interventional environment.*use grasp or boss"),
        ("one", {"a": ["x0"]}, "needs an observational environment"),
        ("two", {"a": [], "b": None}, "targets gives no variables for environments"),
        ("two", {"a": []}, "targets gives no variables for environments"),
        ("one", None, "each environment needs its targets"),
        ("small", {"a": [], "b": ["x0"]}, "at least 3 rows in every environment"),
    ],
    "ut_igsp": [
        (None, None, "at least one interventional environment"),
        ("one", {"a": []}, "at least one interventional environment"),
        ("one", None, "needs an observational environment"),
        ("one", {"a": ["x0"]}, "needs an observational environment"),
        ("one", {"zz": []}, "that env does not have"),
        ("small", {"a": []}, "at least 3 rows in every environment"),
    ],
}


@pytest.mark.parametrize("alpha", [None, 0.05])
@pytest.mark.parametrize(
    ("method", "env", "targets", "message"),
    [(method, *case) for method, cases in BAD_ENVIRONMENTS.items() for case in cases],
)
def test_bad_environments_raise(method, env, targets, message, alpha):
    frame, _, _, _ = sample(method, n=50)
    n = len(frame)
    rows = {
        None: None,
        "one": np.full(n, "a"),
        "two": np.repeat(["a", "b"], [n - n // 2, n // 2]),
        "small": np.array(["a"] * (n - 2) + ["b", "b"]),
    }
    with pytest.raises(ValueError, match=message):
        getattr(andrey, method)(frame, rows[env], targets=targets, alpha=alpha)


@pytest.mark.parametrize("method", METHODS)
def test_without_env_the_messages_name_each_environment_by_its_targets(method):
    frame, _, _, _ = sample(method, n=50)
    n = len(frame)
    mask = np.zeros(frame.shape, dtype=int)
    mask[: n // 2, 0] = mask[n // 2 :, 1] = 1
    with pytest.raises(ValueError, match="needs an observational environment") as raised:
        getattr(andrey, method)(frame, targets=mask)
    assert "['x0']" in str(raised.value) and "['x1']" in str(raised.value)
    assert "label of env" not in str(raised.value)
    with pytest.raises(ValueError, match="at least one interventional environment") as raised:
        getattr(andrey, method)(frame)
    assert "['']" not in str(raised.value)


def at_a_fixed_invariance_p_value(method: str, monkeypatch):
    """A fit of ``method`` under an invariance test whose p-value is always 0.01: the result, and
    whether the fit found a variable that varies, read from IGSP's one invariance answer or from
    UT-IGSP's learned targets, every variable or only the known ones."""
    monkeypatch.setattr(igsp, "make_invariance_test", lambda *args, **kwargs: lambda y, S: 0.01)
    answers: list[bool] = []
    rules = igsp.igsp_rules

    def spy(invariant, family):
        answers.append(invariant(0, (), 0))
        return rules(invariant, family)

    monkeypatch.setattr(igsp, "igsp_rules", spy)

    def fit(frame, env, targets, **levels):
        answers.clear()
        out = getattr(andrey, method)(frame, env, targets=targets, **levels)
        if method == "igsp":
            [invariant] = answers
            return out, not invariant
        found = out.metadata["targets"]
        every = {"observed": [], **dict.fromkeys(("one", "two", "three"), list(frame.columns))}
        known = {"observed": [], "one": ["x1"], "two": ["x3"], "three": []}
        assert found in (every, known), found
        return out, found == every

    return fit


@pytest.mark.parametrize("method", METHODS)
def test_invariance_alpha_reaches_the_invariance_test_and_alpha_does_not(method, monkeypatch):
    """Under an invariance test whose p-value is always 0.01, a variable varies at a level above
    it and does not below it."""
    fit = at_a_fixed_invariance_p_value(method, monkeypatch)
    frame, env, targets, _ = sample(method)
    for levels, varies in (
        ({"invariance_alpha": 0.05}, True),
        ({"invariance_alpha": 0.001}, False),
        ({"alpha": 0.05}, True),
        ({"alpha": 0.001}, False),
        ({"alpha": 0.05, "invariance_alpha": 0.001}, False),
    ):
        assert fit(frame, env, targets, **levels)[1] == varies, levels


@pytest.mark.parametrize("method", METHODS)
def test_alpha_none_is_one_over_the_observational_rows_and_a_float_is_kept(method):
    fit = getattr(andrey, method)
    frame, env, targets, _ = sample(method)
    rows = int((env == "observed").sum())
    default = fit(frame, env, targets=targets)
    explicit = fit(frame, env, targets=targets, alpha=1 / rows)
    assert default.metadata["alpha"] == explicit.metadata["alpha"] == 1 / rows
    assert marks(default) == marks(explicit)
    assert default.metadata.get("targets") == explicit.metadata.get("targets")
    assert fit(frame, env, targets=targets, alpha=0.01).metadata["alpha"] == 0.01
    # Every environment mapped to [] is observational: their rows are pooled and all counted.
    split = np.where((env == "observed") & (np.arange(len(env)) % 2 == 0), "again", env)
    pooled = fit(frame, split, targets={**targets, "again": []})
    assert pooled.metadata["alpha"] == 1 / rows


@pytest.mark.parametrize("method", METHODS)
def test_alpha_none_sets_the_invariance_level_by_the_observational_rows(method, monkeypatch):
    """Under an invariance test whose p-value is always 0.01, a variable varies at 50
    observational rows, a level of 0.02, and does not at 500, a level of 0.002."""
    fit = at_a_fixed_invariance_p_value(method, monkeypatch)
    for n, varies in ((50, True), (500, False)):
        frame, env, targets, _ = sample(method, n=n)
        out, found = fit(frame, env, targets)
        assert out.metadata["alpha"] == 1 / n
        assert found == varies, n


@pytest.mark.parametrize("indep_test", ["fisherz", "kci"])
@pytest.mark.parametrize("method", METHODS)
def test_row_order_environment_labels_and_split_observational_rows_change_nothing(
    method, indep_test
):
    """The same graph, and for UT-IGSP the same targets under each environment's label."""
    fit = getattr(andrey, method)
    frame, env, targets, _ = sample(method, n=150 if indep_test == "kci" else 500)
    kwargs = {"indep_test": indep_test, "n_starts": 2}
    want = fit(frame, env, targets=targets, **kwargs)
    learned = want.metadata.get("targets", {})
    order = np.random.default_rng(1).permutation(len(frame))
    shuffled = fit(frame.iloc[order], env[order], targets=targets, **kwargs)
    assert marks(shuffled) == marks(want)
    assert shuffled.metadata.get("targets", {}) == learned
    renamed = {"observed": "z", "one": "y", "two": "x", "three": "w"}  # reverses the sorted order
    number = {"observed": 3, "one": 0, "two": 2, "three": 1}
    for relabel in (renamed, number):
        relabeled = fit(
            frame,
            np.array([relabel[label] for label in env]),
            targets={relabel[k]: v for k, v in targets.items()},
            **kwargs,
        )
        assert marks(relabeled) == marks(want)
        assert relabeled.metadata.get("targets", {}) == {relabel[k]: v for k, v in learned.items()}
    split = np.where((env == "observed") & (np.arange(len(env)) % 2 == 1), "observed 2", env)
    two = fit(frame, split, targets={**targets, "observed 2": []}, **kwargs)
    assert marks(two) == marks(want)


@pytest.mark.parametrize("method", METHODS)
def test_a_seed_repeats_and_one_start_does_not_depend_on_it(method, monkeypatch):
    fit = getattr(andrey, method)
    frame, env, targets, _ = sample(method, seed=5)
    first = fit(frame, env, targets=targets, seed=3)
    assert marks(fit(frame, env, targets=targets, seed=3)) == marks(first)
    monkeypatch.setenv("ANDREY_SEED", "3")
    assert fit(frame, env, targets=targets).metadata["seed"] == 3
    single = [fit(frame, env, targets=targets, n_starts=1, seed=s) for s in range(5)]
    assert all(marks(out) == marks(single[0]) for out in single)
    assert "seed" not in single[0].metadata


@pytest.mark.parametrize("method", METHODS)
def test_the_indicator_form_with_a_categorical_test_reads_categories_whatever_their_codes(method):
    """Categories ``a -> b -> c`` and a setting that sets ``b`` from outside, a target IGSP is
    given and UT-IGSP learns."""
    rng = np.random.default_rng(0)
    n = 1500
    env = np.repeat(["observed", "set b"], n)
    targets = {"observed": [], "set b": ["b"] if method == "igsp" else None}
    a = rng.integers(0, 3, 2 * n)
    b = np.where(rng.random(2 * n) < 0.8, a, rng.integers(0, 3, 2 * n))
    b[n:] = rng.choice(3, size=n, p=[0.7, 0.2, 0.1])  # set from outside, with a new marginal
    c = np.where(rng.random(2 * n) < 0.8, b, rng.integers(0, 3, 2 * n))

    def table(codes: list[int]) -> pd.DataFrame:
        """The columns as categories, category ``k`` coded ``codes[k]``."""
        return pd.DataFrame(
            {k: pd.Categorical(np.asarray(codes)[v]) for k, v in zip("abc", (a, b, c), strict=True)}
        )

    fit = getattr(andrey, method)
    out = fit(table([0, 1, 2]), env, targets=targets)
    assert out.metadata["indep_test"] == "gsq"
    assert out.metadata["invariance_test"] == "indicator"
    if method == "igsp":
        edges = {("a", "b", "directed"), ("b", "c", "directed")}
        assert set(out.structure.oriented_edges()) == edges
    else:
        assert out.metadata["targets"]["set b"] == ["b"]
        assert ("a", "b", "directed") in out.structure.oriented_edges()
    for codes in ([2, 0, 1], [1, 2, 0]):
        recoded = fit(table(codes), env, targets=targets)
        assert marks(recoded) == marks(out)
        assert recoded.metadata.get("targets") == out.metadata.get("targets")
    # The Gaussian form would regress on the codes, so it refuses the categorical tests.
    for indep_test in ("gsq", "cond_gauss"):
        with pytest.raises(ValueError, match="invariance_test='gaussian'"):
            fit(
                table([0, 1, 2]),
                env,
                targets=targets,
                indep_test=indep_test,
                invariance_test="gaussian",
            )
