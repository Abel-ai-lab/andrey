"""GnIES: the noise-intervention score, its outer search over targets, and the facade's contract."""

from __future__ import annotations

import itertools
import logging
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

import andrey
from andrey.core import ARROW, TAIL
from andrey.core.knowledge import resolve
from andrey.core.score import BICScore
from andrey.search import gnies as engine
from andrey.search.ges import required_dag
from andrey.search.gies import replace_unprotected, search_class
from andrey.search.gnies import NoiseInterventionScore, alternating_fit, singleton_cut
from andrey.search.hc import tie_ranks

D = 5


def _fit(*args, **kwargs) -> andrey.StructureOutput:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        return andrey.gnies(*args, **kwargs)


def _sem(
    weights: np.ndarray,
    sizes: list[int],
    scales: list[dict[int, float]],
    seed: int,
    shifts: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rows of ``x = x @ weights + e`` from each environment, ``weights[i, j]`` the weight of
    ``i -> j``; environment ``k`` multiplies the noise of each key of ``scales[k]`` by its value
    and adds ``shifts[k]`` to every column."""
    rng = np.random.default_rng(seed)
    d = weights.shape[0]
    blocks, env = [], []
    for k, (n, scale) in enumerate(zip(sizes, scales, strict=True)):
        sd = np.ones(d)
        for j, s in scale.items():
            sd[j] = s
        noise = rng.standard_normal((n, d)) * sd
        blocks.append(noise @ np.linalg.inv(np.eye(d) - weights) + (shifts[k] if shifts else 0.0))
        env += [k] * n
    return np.vstack(blocks), np.array(env)


def _chain_weights(d: int = D) -> np.ndarray:
    """``0 -> 1 -> 2``, ``0 -> 3 <- 2``, ``3 -> 4`` with signed weights."""
    w = np.zeros((d, d))
    for (i, j), v in {(0, 1): 0.9, (1, 2): -0.8, (0, 3): 0.7, (2, 3): 0.6, (3, 4): -0.9}.items():
        w[i, j] = v
    return w


@pytest.fixture(scope="module")
def case() -> tuple[np.ndarray, np.ndarray]:
    """Unequal environments; environment 1 scales node 1's noise and 2 scales node 4's."""
    return _sem(_chain_weights(), [400, 250, 150], [{}, {1: 2.5}, {4: 0.4}], seed=1)


@pytest.fixture(scope="module")
def fitted(case) -> andrey.StructureOutput:
    return _fit(*case)


def _within_cov(X: np.ndarray, env: np.ndarray) -> np.ndarray:
    """``sum_e (X_e - mean_e)' (X_e - mean_e) / N``, written out."""
    total = np.zeros((X.shape[1], X.shape[1]))
    for k in np.unique(env):
        c = X[env == k] - X[env == k].mean(axis=0)
        total += c.T @ c
    return total / X.shape[0]


def _marks(out: andrey.StructureOutput) -> np.ndarray:
    return out.structure.to_numpy()


def _targets(out: andrey.StructureOutput) -> list:
    return [v for v, _ in out.metadata["index_edges"]]


# ---- the score ----------------------------------------------------------------------------------


def test_non_target_score_is_the_bic_of_the_within_environment_covariance(case):
    X, env = case
    score = NoiseInterventionScore(X, env, penalty_discount=1.3, ebic_gamma=0.5)
    bic = BICScore.from_cov(_within_cov(X, env), X.shape[0], penalty_discount=1.3, ebic_gamma=0.5)
    for node, parents in [(2, ()), (2, (1,)), (3, (0, 2)), (4, (0, 1, 3))]:
        assert score.view([]).score(node, parents) == pytest.approx(bic.score(node, parents))


def test_alternating_fit_reaches_a_stationary_point_and_never_raises_its_objective(
    case, monkeypatch
):
    X, env = case
    score = NoiseInterventionScore(X, env)
    n = score.sizes.astype(float)
    node, parents = 3, [0, 2]
    fit = alternating_fit(score.covs, score.sizes, score.cov, node, parents)
    assert fit is not None
    L, b = fit
    Spp, Spj = score.covs[:, parents][:, :, parents], score.covs[:, parents, node]
    w = score.covs[:, node, node] - 2 * Spj @ b + np.einsum("kab,a,b->k", Spp, b, b)
    gradient = ((n / w)[:, None] * (Spp @ b - Spj)).sum(axis=0)
    assert np.linalg.norm(gradient) <= 1e-8 * np.linalg.norm(((n / w)[:, None] * Spj).sum(axis=0))
    steps = []
    for cap in range(1, 8):
        monkeypatch.setattr(engine, "MAX_ITER", cap)
        steps.append(alternating_fit(score.covs, score.sizes, score.cov, node, parents)[0])
    assert all(later <= earlier for earlier, later in itertools.pairwise(steps))
    assert steps[-1] >= L


def _closed_form_total(X, env, parents, targets):
    """The deviance of Gamella et al. (2022), Eq. 7, at ``penalty_discount=1``, without the
    extended-BIC term: per node ``-2`` times its maximized log-likelihood, less the constant
    ``N (1 + log(2 pi))``, plus ``log(N)`` per parent and per extra variance of a target."""
    N, d = X.shape
    labels = np.unique(env)
    centered = [X[env == k] - X[env == k].mean(axis=0) for k in labels]
    total = 0.0
    for j in range(d):
        P = parents[j]
        if j not in targets:
            Z, y = np.vstack(centered)[:, P], np.vstack(centered)[:, j]
            b = np.linalg.lstsq(Z, y, rcond=None)[0] if P else np.zeros(0)
            total += N * np.log(np.mean((y - Z @ b) ** 2)) + np.log(N) * len(P)
            continue
        b = np.linalg.lstsq(np.vstack(centered)[:, P], np.vstack(centered)[:, j], rcond=None)[0]
        for _ in range(20_000):
            w = [np.mean((c[:, j] - c[:, P] @ b) ** 2) for c in centered]
            A = sum(c[:, P].T @ c[:, P] / wk for c, wk in zip(centered, w, strict=True))
            r = sum(c[:, P].T @ c[:, j] / wk for c, wk in zip(centered, w, strict=True))
            b = np.linalg.solve(A, r)
        w = [np.mean((c[:, j] - c[:, P] @ b) ** 2) for c in centered]
        total += sum(len(c) * np.log(wk) for c, wk in zip(centered, w, strict=True))
        total += np.log(N) * (len(P) + len(labels) - 1)
    return total


def test_total_is_the_papers_penalized_likelihood():
    weights = _chain_weights()
    X, env = _sem(weights, [300, 120, 80], [{}, {2: 2.0}, {2: 0.5, 4: 3.0}], seed=4)
    parents = {j: np.flatnonzero(weights[:, j]).tolist() for j in range(D)}
    targets = {2, 4}
    view = NoiseInterventionScore(X, env, ebic_gamma=0.0).view(targets)
    total = math.fsum(view.score(j, parents[j]) for j in range(D))
    assert total == pytest.approx(_closed_form_total(X, env, parents, targets), abs=1e-6)


def test_a_target_family_needs_two_rows_more_than_parents_in_every_environment(caplog):
    X, env = _sem(_chain_weights(), [200, 200, 4], [{}, {3: 2.0}, {3: 0.5}], seed=2)
    view = NoiseInterventionScore(X, env).view([3])
    with caplog.at_level(logging.DEBUG, logger="andrey.search.gnies"):
        assert view.score(3, (0, 1, 2)) == math.inf
        assert math.isfinite(view.score(3, (0, 2)))
    assert any("unavailable" in r.getMessage() for r in caplog.records)
    out = _fit(X, env, targets={0: [], 1: [3], 2: [3]})
    marks = _marks(out)
    for target in _targets(out):
        assert np.sum((marks[target] == ARROW) & (marks[:, target] == TAIL)) <= 4 - 2


def test_the_column_check_counts_the_rows_left_after_centering_each_environment():
    rng = np.random.default_rng(3)
    env = np.repeat(np.arange(5), 2)  # 10 rows, 5 left after centering each environment
    _fit(rng.standard_normal((10, 5)), env)  # as many rows left as variables
    with pytest.raises(ValueError, match="leave 5 once each environment's mean"):
        _fit(rng.standard_normal((10, 6)), env)
    X, env = _sem(_chain_weights(), [60, 40], [{}, {1: 2.0}], seed=5)
    data = pd.DataFrame(X, columns=list("abcde")).assign(f=lambda t: t["a"] - 2.0 * t["c"])
    with pytest.raises(ValueError, match="linearly dependent") as raised:
        _fit(data, env)
    assert "'f'" in str(raised.value)


def test_the_score_bits_do_not_depend_on_the_blas_thread_count():
    rng = np.random.default_rng(0)
    d = 20  # large enough that a multithreaded BLAS sums the covariances in another order
    weights = np.triu(rng.uniform(-0.8, 0.8, (d, d)) * (rng.random((d, d)) < 0.15), 1)
    X = rng.standard_normal((12_000, d)) @ np.linalg.inv(np.eye(d) - weights)
    env = np.repeat(np.arange(4), 3_000)
    scores = []
    for threads in (1, 8):
        with threadpool_limits(limits=threads):
            scores.append(_fit(X, env, targets=dict.fromkeys(range(4), [])).metadata["score"])
    assert scores[0] == scores[1]


def _class_members(dag: np.ndarray, targets: set[int]) -> list[np.ndarray]:
    """Every DAG with ``dag``'s skeleton and v-structures that orients each edge at a target
    alike (Gamella et al. 2022, Def. 2), as 0/1 matrices."""

    def v_structures(a):
        return {
            (i, j, k)
            for j in range(len(a))
            for i, k in itertools.combinations(np.flatnonzero(a[:, j]).tolist(), 2)
            if not a[i, k] and not a[k, i]
        }

    edges = [(i, j) for i, j in zip(*np.nonzero(dag), strict=True)]
    members = []
    for flips in itertools.product([False, True], repeat=len(edges)):
        a = np.zeros_like(dag)
        for (i, j), flip in zip(edges, flips, strict=True):
            a[(j, i) if flip else (i, j)] = 1
        if any(flip and (i in targets or j in targets) for (i, j), flip in zip(edges, flips)):
            continue
        order = np.linalg.matrix_power(a + np.eye(len(a), dtype=int), len(a))
        acyclic = not any(order[i, j] and order[j, i] for i, j in edges)
        if acyclic and v_structures(a) == v_structures(dag):
            members.append(a)
    return members


def _random_dag(rng: np.random.Generator, d: int) -> np.ndarray:
    order = rng.permutation(d)
    upper = np.triu(rng.random((d, d)) < 0.5, 1).astype(int)
    return upper[np.ix_(np.argsort(order), np.argsort(order))]


def _marks_of(dag: np.ndarray) -> np.ndarray:
    m = np.zeros(dag.shape, dtype=np.int8)
    m[dag == 1] = TAIL
    m[dag.T == 1] = ARROW
    return m


def test_every_dag_of_a_class_has_one_total():
    d = 6
    weights = np.zeros((d, d))
    for (i, j), v in {(0, 1): 0.8, (1, 2): 0.7, (2, 3): -0.6, (1, 3): 0.9, (3, 4): 0.8}.items():
        weights[i, j] = v
    weights[4, 5] = -0.7
    targets = {0, 5}
    X, env = _sem(weights, [300, 200, 100], [{}, {0: 2.0}, {5: 2.0}], seed=5)
    view = NoiseInterventionScore(X, env).view(targets)
    members = _class_members((weights != 0).astype(int), targets)
    assert len(members) > 1
    totals = [math.fsum(view.score(j, np.flatnonzero(m[:, j])) for j in range(d)) for m in members]
    assert max(totals) - min(totals) <= 1e-9 * abs(totals[0])


def test_the_singleton_cut_gives_the_essential_graph_of_definition_2():
    rng = np.random.default_rng(7)
    for _ in range(20):
        d = int(rng.integers(3, 7))
        dag = _random_dag(rng, d)
        targets = {int(j) for j in rng.choice(d, size=int(rng.integers(0, d)), replace=False)}
        members = _class_members(dag, targets)
        agree = np.logical_and.reduce([m == dag for m in members])
        essential = replace_unprotected(_marks_of(dag), singleton_cut(targets, d))
        directed = (essential == TAIL) & (essential.T == ARROW)
        assert np.array_equal(directed, (dag == 1) & agree)


# ---- the search ---------------------------------------------------------------------------------


def test_a_variable_whose_noise_changes_is_learned_as_a_target(case):
    X, env = case
    graph, total, learned = engine.gnies(X, env)
    assert learned == [1, 4]
    assert total < engine.gnies(X, env, fixed=True)[1] - 1.0


def test_known_targets_stay_and_a_full_mapping_runs_one_search(case):
    X, env = case
    assert 0 in _targets(_fit(X, env, targets={0: [0]}))
    out = _fit(X, env, targets={0: [], 1: [1], 2: [2]})
    assert _targets(out) == [1, 2]
    score = NoiseInterventionScore(X, env)
    ranks = tie_ranks(score.centered, np.zeros(D, dtype=bool)).tolist()
    adj, total = search_class(score.view({1, 2}), singleton_cut({1, 2}, D), ranks)
    assert np.array_equal(_marks(out), adj)
    assert out.metadata["score"] == total


def test_max_parents_and_knowledge_hold(case):
    X, env = case
    assert _fit(X, env, max_parents=0).structure.oriented_edges() == []
    forbidden = andrey.Knowledge(forbidden=[(0, 1), (1, 0)])
    edges = _fit(X, env, knowledge=forbidden).structure.oriented_edges()
    assert not any({i, j} == {0, 1} for i, j, _ in edges)


def test_required_edges_a_target_cannot_carry_keep_the_variable_out_of_the_targets():
    # The second environment has 3 rows, so a target has at most one parent; 3 must have two.
    X, env = _sem(_chain_weights(), [200, 3], [{}, {3: 3.0}], seed=6)
    knowledge = andrey.Knowledge(required=[(0, 3), (2, 3)])
    out = _fit(X, env, knowledge=knowledge)
    assert 3 not in _targets(out)
    assert {(0, 3, "directed"), (2, 3, "directed")} <= set(out.structure.oriented_edges())
    with pytest.raises(ValueError, match="2 parents to the known target 3"):
        _fit(X, env, targets={0: [], 1: [3]}, knowledge=knowledge)
    constraints = resolve(knowledge, D, None, method="gnies")
    score = NoiseInterventionScore(X, env)
    ranks = tie_ranks(score.centered, np.zeros(D, dtype=bool)).tolist()
    cut = singleton_cut({3}, D)
    adj, total = search_class(score.view({3}), cut, ranks, constraints)
    assert total == math.inf
    assert np.array_equal(adj, replace_unprotected(required_dag(constraints), cut))
    with pytest.raises(ValueError, match="no graph with the required edges"):
        engine.gnies(X, env, known={3}, fixed=True, constraints=constraints)


# ---- invariances --------------------------------------------------------------------------------


def test_a_column_relabeling_relabels_the_result(case, fitted):
    X, env = case
    perm = np.array([3, 0, 4, 2, 1])
    out = _fit(X[:, perm], env)
    assert np.array_equal(_marks(out), _marks(fitted)[np.ix_(perm, perm)])
    assert sorted(int(perm[j]) for j in _targets(out)) == _targets(fitted)


def test_a_row_shuffle_changes_nothing(case, fitted):
    X, env = case
    rows = np.random.default_rng(0).permutation(len(env))
    out = _fit(X[rows], env[rows])
    assert np.array_equal(_marks(out), _marks(fitted))
    assert _targets(out) == _targets(fitted)
    assert out.metadata["score"] == pytest.approx(fitted.metadata["score"], rel=1e-12)


def test_relabeled_environments_give_the_same_bits(case, fitted):
    X, env = case
    out = _fit(X, np.array(["c", "a", "b"])[env])
    assert np.array_equal(_marks(out), _marks(fitted))
    assert _targets(out) == _targets(fitted)
    assert out.metadata["score"] == fitted.metadata["score"]


def test_units_and_environment_means_do_not_matter(case, fitted):
    X, env = case
    scaled = X * np.array([3.0, 0.01, 7.0, 0.5, 40.0]) + np.array([1.0, -2.0, 5.0, 0.0, 9.0])
    shifted = X + np.where(env[:, None] == 2, 4.0, 0.0)
    for data in (scaled, shifted):
        out = _fit(data, env)
        assert np.array_equal(_marks(out), _marks(fitted))
        assert _targets(out) == _targets(fitted)


def test_standardizing_each_environment_changes_the_result(case, fitted):
    X, env = case
    each = X.copy()
    for k in np.unique(env):
        each[env == k] = (X[env == k] - X[env == k].mean(axis=0)) / X[env == k].std(axis=0)
    assert _targets(_fit(each, env)) != _targets(fitted)


# ---- the facade -------------------------------------------------------------------------------


def test_the_fit_repeats_and_records_its_metadata(case, fitted):
    X, env = case
    again = _fit(X, env)
    assert np.array_equal(_marks(again), _marks(fitted))
    assert again.metadata == fitted.metadata
    assert fitted.metadata["algorithm"] == "gnies"
    assert fitted.metadata["index_edges"] == [[1, "directed"], [4, "directed"]]
    assert fitted.metadata["environments"] == [0, 1, 2]
    assert (fitted.metadata["penalty_discount"], fitted.metadata["ebic_gamma"]) == (1.0, 1.0)


def test_labels_name_the_targets_and_environments_without_env(case):
    X, env = case
    data = pd.DataFrame(X, columns=list("abcde"))
    data["changed"] = np.array(["", "b", "e"], dtype=object)[env]
    out = _fit(data, targets="changed")
    assert out.metadata["index_edges"] == [["b", "directed"], ["e", "directed"]]
    assert out.metadata["environments"] == [[], ["e"], ["b"]]  # numpy.unique's row order


def test_errors_name_an_environment_from_targets_by_its_targets(case):
    X, env = case
    data = pd.DataFrame(X, columns=list("abcde"))
    data["changed"] = np.array(["", "b", "e"], dtype=object)[env]
    one_row = data.copy()
    one_row.loc[0, "changed"] = "c"
    with pytest.raises(ValueError, match=r"^gnies: .*\[\['c'\]\] have fewer"):
        _fit(one_row, targets="changed")
    flat = data.copy()
    flat.loc[env == 2, "a"] = 1.0
    with pytest.raises(ValueError, match=r"^gnies: column 'a' is constant in environment \['e'\]"):
        _fit(flat, targets="changed")


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ("one_environment", "two or more environments"),
        ("one_row", "at least 2 rows"),
        ("constant_in_one", "constant in environment"),
        ("unknown_label", "does not have"),
        ("categorical", "declared categorical"),
        ("nan", "NaN"),
    ],
)
def test_invalid_input_raises(case, change, match):
    X, env = case
    data: object = X.copy()
    targets = None
    if change == "one_environment":
        env = np.zeros(len(env), dtype=int)
    elif change == "one_row":
        env = env.copy()
        env[0] = 9
    elif change == "constant_in_one":
        data = X.copy()
        data[env == 2, 3] = 1.5
    elif change == "unknown_label":
        targets = {7: [1]}
    elif change == "categorical":
        data = pd.DataFrame(X, columns=list("abcde")).astype({"a": "category"})
    else:
        data = X.copy()
        data[3, 2] = np.nan
    with pytest.raises(ValueError, match=match):
        _fit(data, env, targets=targets)
