"""``andrey.igsp``: its inputs, outputs, errors, and what the result does not depend on."""

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


def sample(seed: int = 2, n: int = 500):
    """Linear-Gaussian data from an observational environment and three settings: shift ``x1``,
    shift ``x3`` and ``x4``, and a hard intervention on ``x5``.

    Returns the DataFrame, the label of each row's environment, the targets mapping, and the
    dataset. At the default seed the fit finds the true interventional essential graph.
    """
    p = 6
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=p, avg_degree=2.0),
        functional=data.functional.linear(weight_range=(0.5, 1.0)),
        noise=data.noise.gaussian(),
    )
    envs = [{}, {1: soft(shift=2.0)}, {3: soft(shift=2.0), 4: soft(shift=2.0)}, {5: hard(2.0)}]
    ds = scm.sample(n=n, seed=seed, scale="raw", envs=envs)
    frame = pd.DataFrame(ds.data, columns=[f"x{j}" for j in range(p)])
    labels = np.array(["observed", "one", "two", "three"])[ds.env]
    targets = {"observed": [], "one": ["x1"], "two": ["x3", "x4"], "three": ["x5"]}
    return frame, labels, targets, ds


def essential(dag: np.ndarray, family: list[set[int]]) -> np.ndarray:
    rows = np.zeros((len(family) + 1, dag.shape[0]), dtype=bool)
    for row, variables in enumerate(family, start=1):
        rows[row, sorted(variables)] = True
    return replace_unprotected(dag.astype(np.int8), cut_pairs(rows))


def marks(out) -> dict:
    """The result as ``{(label, label): mark}``, which no column order changes."""
    m, labels = out.structure.to_numpy(), out.structure.labels
    return {(a, b): int(m[i, j]) for i, a in enumerate(labels) for j, b in enumerate(labels)}


def test_the_result_is_the_true_class_with_the_edges_and_contradicted_edges_found():
    frame, env, targets, ds = sample()
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
    frame, env, targets, _ = sample()
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
    frame, env, _, _ = sample(n=300)
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
    ("env", "targets", "message"),
    [
        (None, None, "at least one interventional environment.*use grasp or boss"),
        ("one", {"a": []}, "at least one interventional environment.*use grasp or boss"),
        ("one", {"a": ["x0"]}, "needs an observational environment"),
        ("two", {"a": [], "b": None}, "targets gives no variables for environments"),
        ("two", {"a": []}, "targets gives no variables for environments"),
        ("one", None, "each environment needs its targets"),
        ("small", {"a": [], "b": ["x0"]}, "at least 3 rows in every environment"),
    ],
)
def test_bad_environments_raise(env, targets, message):
    frame, _, _, _ = sample(n=50)
    n = len(frame)
    rows = {
        None: None,
        "one": np.full(n, "a"),
        "two": np.repeat(["a", "b"], [n - n // 2, n // 2]),
        "small": np.array(["a"] * (n - 2) + ["b", "b"]),
    }
    with pytest.raises(ValueError, match=message):
        andrey.igsp(frame, rows[env], targets=targets)


@pytest.mark.parametrize("method", ["igsp", "ut_igsp"])
def test_without_env_the_messages_name_each_environment_by_its_targets(method):
    frame, _, _, _ = sample(n=50)
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
    frame, env, targets, _ = sample(n=50)
    with pytest.raises(ValueError, match=message):
        andrey.igsp(frame, env, targets=targets, **kwargs)


def test_a_missing_value_or_a_column_constant_on_the_observational_rows_raises():
    frame, env, targets, _ = sample(n=100)
    gap = frame.copy()
    gap.iloc[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN|missing"):
        andrey.igsp(gap, env, targets=targets)
    frame.loc[env == "observed", "x2"] = 1.0
    with pytest.raises(ValueError, match=r"columns \['x2'\] are constant on the observational"):
        andrey.igsp(frame, env, targets=targets)


def test_invariance_alpha_reaches_the_invariance_test_and_alpha_does_not(monkeypatch):
    """Under an invariance test whose p-value is always 0.01, a variable varies at a level above
    it and does not below it."""
    monkeypatch.setattr(igsp, "make_invariance_test", lambda *args, **kwargs: lambda y, S: 0.01)
    answers: list[bool] = []
    rules = igsp.igsp_rules

    def spy(invariant, family):
        answers.append(invariant(0, (), 0))
        return rules(invariant, family)

    monkeypatch.setattr(igsp, "igsp_rules", spy)
    frame, env, targets, _ = sample()
    for levels, invariant in (
        ({"invariance_alpha": 0.05}, False),
        ({"invariance_alpha": 0.001}, True),
        ({"alpha": 0.05}, False),
        ({"alpha": 0.001}, True),
        ({"alpha": 0.05, "invariance_alpha": 0.001}, True),
    ):
        answers.clear()
        andrey.igsp(frame, env, targets=targets, **levels)
        assert answers == [invariant]


def test_alpha_none_is_one_over_the_observational_rows_and_a_float_is_kept():
    frame, env, targets, _ = sample()
    rows = int((env == "observed").sum())
    default = andrey.igsp(frame, env, targets=targets)
    explicit = andrey.igsp(frame, env, targets=targets, alpha=1 / rows)
    assert default.metadata["alpha"] == explicit.metadata["alpha"] == 1 / rows
    assert marks(default) == marks(explicit)
    assert andrey.igsp(frame, env, targets=targets, alpha=0.01).metadata["alpha"] == 0.01
    # Every environment mapped to [] is observational: their rows are pooled and all counted.
    split = np.where((env == "observed") & (np.arange(len(env)) % 2 == 0), "again", env)
    pooled = andrey.igsp(frame, split, targets={**targets, "again": []})
    assert pooled.metadata["alpha"] == 1 / rows


def test_alpha_none_sets_the_invariance_level_by_the_observational_rows(monkeypatch):
    """Under an invariance test whose p-value is always 0.01, a variable varies at 50
    observational rows, a level of 0.02, and does not at 500, a level of 0.002."""
    monkeypatch.setattr(igsp, "make_invariance_test", lambda *args, **kwargs: lambda y, S: 0.01)
    answers: list[bool] = []
    rules = igsp.igsp_rules

    def spy(invariant, family):
        answers.append(invariant(0, (), 0))
        return rules(invariant, family)

    monkeypatch.setattr(igsp, "igsp_rules", spy)
    for n, invariant in ((50, False), (500, True)):
        frame, env, targets, _ = sample(n=n)
        answers.clear()
        assert andrey.igsp(frame, env, targets=targets).metadata["alpha"] == 1 / n
        assert answers == [invariant]


@pytest.mark.parametrize("indep_test", ["fisherz", "kci"])
def test_row_order_environment_labels_and_split_observational_rows_change_nothing(indep_test):
    frame, env, targets, _ = sample(n=150 if indep_test == "kci" else 500)
    kwargs = {"indep_test": indep_test, "n_starts": 2}
    want = marks(andrey.igsp(frame, env, targets=targets, **kwargs))
    order = np.random.default_rng(1).permutation(len(frame))
    assert marks(andrey.igsp(frame.iloc[order], env[order], targets=targets, **kwargs)) == want
    renamed = {"observed": "z", "one": "y", "two": "x", "three": "w"}  # reverses the sorted order
    relabeled = andrey.igsp(
        frame,
        np.array([renamed[label] for label in env]),
        targets={renamed[k]: v for k, v in targets.items()},
        **kwargs,
    )
    assert marks(relabeled) == want
    number = {"observed": 3, "one": 0, "two": 2, "three": 1}
    numbered = andrey.igsp(
        frame,
        np.array([number[label] for label in env]),
        targets={number[k]: v for k, v in targets.items()},
        **kwargs,
    )
    assert marks(numbered) == want
    split = np.where((env == "observed") & (np.arange(len(env)) % 2 == 1), "observed 2", env)
    two = andrey.igsp(frame, split, targets={**targets, "observed 2": []}, **kwargs)
    assert marks(two) == want


@pytest.mark.parametrize("seed", range(5))
def test_reordering_the_columns_relabels_the_graph(seed):
    rng = np.random.default_rng(seed)
    for dataset in range(20):
        frame, env, targets, _ = sample(seed=dataset, n=300)
        want = marks(andrey.igsp(frame, env, targets=targets, seed=seed))
        columns = list(rng.permutation(frame.columns))
        assert marks(andrey.igsp(frame[columns], env, targets=targets, seed=seed)) == want


def test_rescaling_and_shifting_the_columns_changes_nothing():
    frame, env, targets, _ = sample(seed=3)
    want = marks(andrey.igsp(frame, env, targets=targets))
    moved = frame * 2.0 ** np.arange(-3, 3) + np.arange(6) * 10.0
    assert marks(andrey.igsp(moved, env, targets=targets)) == want


def test_a_seed_repeats_and_one_start_does_not_depend_on_it(monkeypatch):
    frame, env, targets, _ = sample(seed=5)
    first = andrey.igsp(frame, env, targets=targets, seed=3)
    assert marks(andrey.igsp(frame, env, targets=targets, seed=3)) == marks(first)
    monkeypatch.setenv("ANDREY_SEED", "3")
    assert andrey.igsp(frame, env, targets=targets).metadata["seed"] == 3
    single = [andrey.igsp(frame, env, targets=targets, n_starts=1, seed=s) for s in range(5)]
    assert all(marks(out) == marks(single[0]) for out in single)
    assert "seed" not in single[0].metadata


def test_knowledge_keeps_required_edges_drops_forbidden_ones_and_orients_by_it():
    frame, env, targets, _ = sample()
    base = andrey.igsp(frame, env, targets=targets)
    present = next(e for e in base.structure.oriented_edges())
    absent = next(
        (a, b)
        for a in frame.columns
        for b in frame.columns
        if a < b and not any({a, b} == {e[0], e[1]} for e in base.structure.oriented_edges())
    )
    knowledge = andrey.Knowledge(required=[absent], forbidden=[present[:2]])
    out = andrey.igsp(frame, env, targets=targets, knowledge=knowledge)
    edges = {(a, b): kind for a, b, kind in out.structure.oriented_edges()}
    assert edges.get(absent) == "directed"
    assert present[:2] not in edges


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


def test_the_indicator_form_with_a_categorical_test_reads_categories_whatever_their_codes():
    rng = np.random.default_rng(0)
    n = 1500
    env = np.repeat(["observed", "set b"], n)
    targets = {"observed": [], "set b": ["b"]}
    a = rng.integers(0, 3, 2 * n)
    b = np.where(rng.random(2 * n) < 0.8, a, rng.integers(0, 3, 2 * n))
    b[n:] = rng.choice(3, size=n, p=[0.7, 0.2, 0.1])  # set from outside, with a new marginal
    c = np.where(rng.random(2 * n) < 0.8, b, rng.integers(0, 3, 2 * n))

    def table(codes: list[int]) -> pd.DataFrame:
        """The columns as categories, category ``k`` coded ``codes[k]``."""
        return pd.DataFrame(
            {k: pd.Categorical(np.asarray(codes)[v]) for k, v in zip("abc", (a, b, c), strict=True)}
        )

    out = andrey.igsp(table([0, 1, 2]), env, targets=targets)
    assert out.metadata["indep_test"] == "gsq"
    assert out.metadata["invariance_test"] == "indicator"
    assert set(out.structure.oriented_edges()) == {("a", "b", "directed"), ("b", "c", "directed")}
    for codes in ([2, 0, 1], [1, 2, 0]):
        assert marks(andrey.igsp(table(codes), env, targets=targets)) == marks(out)
