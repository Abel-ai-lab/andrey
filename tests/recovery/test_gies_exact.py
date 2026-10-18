from __future__ import annotations

import itertools

import numpy as np
import pytest

import andrey
from andrey.core.orient import from_structure, pdag2dag
from andrey.core.score import BICScore
from andrey.metrics import markov_equivalent
from andrey.search import gies as gies_search
from andrey.search.gies import cut_pairs, replace_unprotected
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output, assert_recorded_output_mec


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


_BASELINE_LAMBDA = 2.0


def test_gies_recovery():
    assert_recorded_output(
        andrey.gies(datagen.generate("gauss_5v"), lambda_value=_BASELINE_LAMBDA),
        "GIES",
        case="gauss_5v",
    )


# Edge weights: the v-structure 1 -> 3 <- 2 below 0 -> 1 and 0 -> 2, then 3 -> 4, and the chain
# 5 -> 6 -> 7. Each edge points from a lower to a higher index.
_GIES_EDGES = {
    (0, 1): 0.9,
    (0, 2): 0.8,
    (1, 3): 0.7,
    (2, 3): 0.6,
    (3, 4): 0.8,
    (5, 6): 0.9,
    (6, 7): 0.8,
}


def _interventional_sem(
    targets: list[list[int]], n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rows of the linear-Gaussian SEM, cycling through no intervention and each target set.

    An intervened variable takes a value drawn around 2 instead of its structural equation.
    """
    rng = np.random.default_rng(seed)
    d = 8
    mask = np.zeros((n, d), dtype=bool)
    for k, target in enumerate(targets, start=1):
        mask[np.ix_(np.arange(n) % (len(targets) + 1) == k, target)] = True
    X = np.zeros((n, d))
    for j in range(d):
        X[:, j] = sum(w * X[:, i] for (i, c), w in _GIES_EDGES.items() if c == j)
        X[:, j] += rng.standard_normal(n)
        X[mask[:, j], j] = rng.normal(2.0, 0.5, mask[:, j].sum())
    return X, mask


def test_gies_orients_the_edges_the_interventions_identify():
    """Intervening on 1 and 6 orients 0 -> 1, 5 -> 6, and 6 -> 7; 0 -- 2 stays open."""
    X, mask = _interventional_sem([[1], [6]], n=3000, seed=0)
    found = andrey.gies(X, mask, lambda_value=_BASELINE_LAMBDA).structure
    assert sorted(found.oriented_edges(index=True)) == [
        (0, 1, "directed"),
        (0, 2, "undirected"),
        (1, 3, "directed"),
        (2, 3, "directed"),
        (3, 4, "directed"),
        (5, 6, "directed"),
        (6, 7, "directed"),
    ]
    # It is the essential graph of its own representative under the same interventions.
    marks = from_structure(found)
    assert np.array_equal(replace_unprotected(pdag2dag(marks), cut_pairs(mask)), marks)


def test_gies_on_observational_data_returns_the_cpdag():
    X, _ = _interventional_sem([], n=3000, seed=0)
    found = andrey.gies(X, lambda_value=_BASELINE_LAMBDA).structure
    assert sorted(found.oriented_edges(index=True)) == [
        (0, 1, "undirected"),
        (0, 2, "undirected"),
        (1, 3, "directed"),
        (2, 3, "directed"),
        (3, 4, "directed"),
        (5, 6, "undirected"),
        (6, 7, "undirected"),
    ]


def _random_interventional(
    seed: int, d: int, p: float, n_targets: int, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """A random linear-Gaussian SEM with edge density ``p`` and single-variable targets."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)
    weights = np.zeros((d, d))  # weights[i, j]: the weight of i -> j
    for a, b in itertools.combinations(range(d), 2):
        if rng.random() < p:
            weights[order[a], order[b]] = rng.uniform(0.3, 1.0) * rng.choice((-1.0, 1.0))
    setting = np.arange(n) % (n_targets + 1)
    mask = np.zeros((n, d), dtype=bool)
    for k, target in enumerate(rng.choice(d, size=n_targets, replace=False), start=1):
        mask[setting == k, target] = True
    X = np.zeros((n, d))
    for j in order:
        X[:, j] = X @ weights[:, j] + rng.standard_normal(n)
        X[mask[:, j], j] = rng.normal(2.0, 0.5, mask[:, j].sum())
    return X, mask


def test_gies_finds_no_edges_between_independent_columns_with_interventions():
    X, mask = _random_interventional(0, 4, 0.0, 2, n=600)
    assert mask.any()
    assert andrey.gies(X, mask).structure.oriented_edges() == []


# The cases with targets run the search with interventions. On the observational (5, 15, 0.3)
# data, a step changes the candidates of a line neighbor of a vertex whose marks it changed. On
# the observational (5, 6, 0.6) data, the turning phase meets a line that fails the separation
# condition of Prop. 31.
@pytest.mark.parametrize(
    "seed,d,p,n_targets",
    [
        (0, 12, 0.4, 2),
        (1, 15, 0.3, 3),
        (3, 25, 0.15, 5),
        (5, 15, 0.3, 0),
        (5, 6, 0.6, 0),
    ],
)
def test_gies_keeps_the_full_scan_result(monkeypatch, seed, d, p, n_targets):
    """Candidates kept between steps give the graph and score of a scan of every target."""
    X, mask = _random_interventional(seed, d, p, n_targets, n=40 * d)
    mask = mask if n_targets else None
    kept_graph, kept_score = gies_search.gies(X, mask)
    monkeypatch.setattr(gies_search, "_FULL_RESCAN", True)
    full_graph, full_score = gies_search.gies(X, mask)
    assert np.array_equal(kept_graph.to_numpy(), full_graph.to_numpy())
    assert kept_score == full_score


def test_gies_step_that_does_not_lower_the_score_fails():
    X, mask = _interventional_sem([[1], [6]], n=600, seed=0)
    score = gies_search.InterventionalBICScore(X, mask)
    search = gies_search._Search(score, X.shape[1])
    with pytest.raises(RuntimeError):
        search.apply(search.adj.copy())


def test_gies_is_deterministic():
    X, mask = _interventional_sem([[2], [7]], n=600, seed=2)
    first, second = andrey.gies(X, mask), andrey.gies(X, mask)
    assert np.array_equal(first.structure.to_numpy(), second.structure.to_numpy())
    assert first.metadata["score"] == second.metadata["score"]


def _bic(adjacency: np.ndarray, score: BICScore) -> float:
    """BIC of a DAG given as a 0/1 adjacency, ``adjacency[i, j] = 1`` for ``i -> j``."""
    return sum(score.score(v, np.flatnonzero(adjacency[:, v])) for v in range(adjacency.shape[0]))


def _all_dags(d: int):
    """Every DAG on ``d`` nodes as a 0/1 adjacency."""
    pairs = list(itertools.combinations(range(d), 2))
    for states in itertools.product((0, 1, 2), repeat=len(pairs)):
        adjacency = np.zeros((d, d), dtype=np.int8)
        for (i, j), state in zip(pairs, states):
            if state:
                adjacency[(i, j) if state == 1 else (j, i)] = 1
        reach = adjacency.copy()
        for _ in range(d):
            reach = reach | ((reach @ adjacency) > 0)
        if not np.diag(reach).any():
            yield adjacency


@pytest.mark.parametrize("search_method", ["astar", "dp"])
def test_exact_search_scores_as_well_as_every_dag(search_method):
    """On four variables the search returns one of the 543 DAGs, and none has a lower BIC."""
    x = datagen.generate("gauss_5v")[:, 1:]
    score = BICScore(x)
    found = (andrey.exact_search(x, search_method=search_method).structure.to_numpy() == 1).astype(
        np.int8
    )
    dags = list(_all_dags(4))
    assert any(np.array_equal(found, dag) for dag in dags)
    assert _bic(found, score) <= min(_bic(dag, score) for dag in dags) + 1e-9


@pytest.mark.parametrize("search_method", ["astar", "dp"])
def test_exact_search_recovers_the_planted_class(search_method):
    X, _ = _interventional_sem([], n=3000, seed=1)
    planted = np.zeros((8, 8), dtype=np.int8)
    planted[tuple(np.transpose(list(_GIES_EDGES)))] = 1
    found = andrey.exact_search(X, search_method=search_method).structure
    assert markov_equivalent(found, planted)


def test_exact_search_recovery():
    assert_recorded_output_mec(
        andrey.exact_search(datagen.generate("gauss_5v")),
        "ExactSearch",
        case="gauss_5v",
    )


def test_gies_output_contract():
    out = andrey.gies(datagen.generate("gauss_5v"))
    assert out.structure.kind == "cpdag" and "score" in out.metadata


def test_exact_search_output_contract():
    assert andrey.exact_search(datagen.generate("gauss_5v")).structure.kind == "dag"


@pytest.mark.parametrize("search_method", ["astar", "dp"])
def test_exact_search_finds_no_edges_between_independent_columns(search_method):
    """Columns with a nonzero mean, so the score has to center them."""
    independent = np.random.default_rng(0).normal(5.0, 1.0, size=(500, 3))
    found = andrey.exact_search(independent, search_method=search_method).structure
    assert found.oriented_edges() == []


@pytest.mark.parametrize("search_method", ["astar", "dp"])
def test_exact_search_ignores_the_variables_means(search_method):
    """A shift can change which score-tied DAG is returned, not its Markov equivalence class."""
    x = datagen.generate("gauss_5v")
    centered = andrey.exact_search(x, search_method=search_method).structure
    shifted = andrey.exact_search(x + 7.0, search_method=search_method).structure
    assert markov_equivalent(shifted, centered)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_exact_search_rejects_a_column_whose_variance_underflows():
    """A column that is not constant but whose variance underflows to zero is rejected."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((1000, 4))
    data[:, 2] *= 1e-170
    with pytest.raises(ValueError, match="is constant"):
        andrey.exact_search(data)


def test_exact_search_fits_full_rank_columns_of_mixed_scale():
    """A full-rank column on a far smaller scale than the others fits, to the same class."""
    rng = np.random.default_rng(0)
    a, e1, e2, e3 = rng.standard_normal((4, 1000))
    b = 0.8 * a + e1
    unit = np.column_stack([a, b, 0.7 * b + e2, e3])
    mixed = unit * np.array([1.0, 1.0, 1e-13, 1.0])
    found = andrey.exact_search(mixed).structure
    assert found.oriented_edges() != []
    assert markov_equivalent(found, andrey.exact_search(unit).structure)


def test_gfci_recovery():
    out = andrey.gfci(datagen.generate("gauss_5v"), lambda_value=_BASELINE_LAMBDA)
    assert_recorded_output(out, "GFCI", case="gauss_5v")


def test_gfci_finds_no_edges_between_independent_columns():
    x = np.random.default_rng(0).standard_normal((1000, 4))
    assert andrey.gfci(x).structure.oriented_edges() == []


# The PAG of ``gauss_5v``, whose DAG 0 -> 1, 0 -> 2, 1 -> 3 <- 2, 3 -> 4 has no hidden variable.
# Every Markov-equivalent MAG has the collider at 3, tails at 1 and 2 on their edges to 3 (an
# arrowhead there adds a collider at 1 or 2, or a bidirected edge between ancestors), and 3 -> 4.
# 0's edges point either way, so both their ends are circles.
_GAUSS_5V_PAG = [
    (0, 1, "circle"),
    (0, 2, "circle"),
    (1, 3, "directed"),
    (2, 3, "directed"),
    (3, 4, "directed"),
]


def test_gfci_recovers_the_pag_of_gauss_5v():
    assert sorted(datagen.planted_pag("gauss_5v").oriented_edges()) == _GAUSS_5V_PAG
    out = andrey.gfci(datagen.generate("gauss_5v"))
    assert sorted(out.structure.oriented_edges()) == _GAUSS_5V_PAG


# Observed columns a, b, c, d, e of ``confounded_gauss_5v``; h is hidden.
_LATENT = "confounded_gauss_5v"
_PLANTED_PAG = [
    (0, 1, "partially_directed"),  # a o-> b
    (1, 2, "bidirected"),  # b <-> c: the hidden common cause
    (2, 4, "directed"),  # c -> e
    (3, 2, "partially_directed"),  # d o-> c
]


def test_gfci_recovers_the_planted_pag():
    """The planted PAG has a bidirected edge, and GES on the data has an adjacency it lacks, so
    GFCI's tests must remove it."""
    x = datagen.generate(_LATENT)
    assert sorted(datagen.planted_pag(_LATENT).oriented_edges()) == _PLANTED_PAG
    ges = andrey.ges(x).structure
    adjacent = {frozenset(edge[:2]) for edge in ges.oriented_edges(index=True)}
    assert adjacent - {frozenset(edge[:2]) for edge in _PLANTED_PAG}
    assert sorted(andrey.gfci(x).structure.oriented_edges()) == _PLANTED_PAG


def test_gfci_output_contract():
    assert andrey.gfci(datagen.generate("gauss_5v")).structure.kind == "pag"


def test_gfci_rejects_non_fisherz():
    with pytest.raises(NotImplementedError):
        andrey.gfci(datagen.generate("gauss_5v"), indep_test="kci")
