"""CALM checks a fixed case, ground truth, its recorded output, background knowledge, and
same-seed determinism.

The tree fixture has a CPDAG that can be checked by hand. The ground-truth case is a random
linear-Gaussian SEM whose Markov equivalence class CALM must return, and whose DAG it returns with
the tiers of a causal order. The recorded output holds only the graph: the weights come from a
torch optimizer.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

import andrey
from andrey.metrics import markov_equivalent, to_cpdag
from tests.recovery.helpers import datagen
from tests.recovery.helpers.harness import assert_recorded_output


def _random_sem(d: int, n: int, p: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``(X, B)``: an ``(n, d)`` linear-Gaussian sample and its true weight matrix ``B``."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)
    B = np.zeros((d, d))
    for a in range(d):
        for b in range(a + 1, d):
            i, j = order[a], order[b]
            if rng.random() < p:
                B[i, j] = rng.uniform(0.5, 2.0) * rng.choice([-1.0, 1.0])
    X = rng.normal(size=(n, d)) @ np.linalg.inv(np.eye(d) - B)
    return X, B


@pytest.fixture(scope="module")
def tree_fit():
    """CALM's default fit on ``gauss_tree_5v``, shared by the tests that read it."""
    pytest.importorskip("torch")
    return andrey.calm(datagen.generate("gauss_tree_5v"))


@pytest.fixture(scope="module")
def sem():
    """A random SEM, its weights, and CALM's default fit on it."""
    pytest.importorskip("torch")
    X, B = _random_sem(5, 2000, 0.4, seed=7)
    return X, B, andrey.calm(X)


# ---- fixed cases ---------------------------------------------------------------------------------


def test_calm_finds_no_edges_between_independent_columns() -> None:
    """No pair enters the moral graph, so no edge is a candidate."""
    pytest.importorskip("torch")
    out = andrey.calm(np.random.default_rng(0).standard_normal((1000, 4)))
    assert out.structure.oriented_edges() == []
    assert not out.weighted_adjacency.any()


def test_calm_recovers_the_class_of_a_tree(tree_fit) -> None:
    """The tree 0 -> 1, 0 -> 2, 1 -> 3, 1 -> 4 has no collider, so its CPDAG is undirected."""
    cpdag = datagen.to_variables("gauss_tree_5v", to_cpdag(tree_fit.structure))
    assert cpdag.oriented_edges(index=True) == [
        (0, 1, "undirected"),
        (0, 2, "undirected"),
        (1, 3, "undirected"),
        (1, 4, "undirected"),
    ]


def test_calm_matches_the_recorded_output(tree_fit) -> None:
    assert_recorded_output(tree_fit, "CALM", case="gauss_tree_5v")


# ---- ground truth --------------------------------------------------------------------------------


def test_calm_recovers_the_true_class(sem) -> None:
    _, B, out = sem
    assert out.structure.kind == "dag"
    assert markov_equivalent(out.structure, (B != 0).astype(int))  # [i, j] = 1 is i -> j
    assert out.weighted_adjacency.shape == B.shape


# ---- background knowledge ------------------------------------------------------------------------


def _directed(out) -> set[tuple[int, int]]:
    return {(i, j) for i, j, _ in out.structure.oriented_edges(index=True)}


def test_calm_holds_required_forbidden_and_tier_edges(sem) -> None:
    """A required edge between independent columns is kept, a forbidden true edge is dropped, and
    no edge points into an earlier tier."""
    X, B, plain = sem
    assert B[2, 4] and not (B[3].any() or B[:, 3].any())  # 2 -> 4, and 3 is independent
    assert (2, 4) in _directed(plain) and not {(3, 4), (4, 3)} & _directed(plain)
    knowledge = andrey.Knowledge(
        required=[(3, 4)], forbidden=[(2, 4), (4, 2)], tiers=[[0, 2], [1, 3, 4]]
    )
    edges = _directed(andrey.calm(X, knowledge=knowledge))
    assert (3, 4) in edges
    assert not {(2, 4), (4, 2)} & edges
    assert not {(i, j) for i, j in edges if i in (1, 3, 4) and j in (0, 2)}


def test_calm_with_the_true_order_as_tiers_returns_the_true_dag(sem) -> None:
    X, B, _ = sem
    truth = {(int(i), int(j)) for i, j in zip(*np.nonzero(B), strict=True)}
    graph = nx.DiGraph(truth)
    graph.add_nodes_from(range(len(B)))
    tiers = [[v] for v in nx.topological_sort(graph)]
    assert _directed(andrey.calm(X, knowledge=andrey.Knowledge(tiers=tiers))) == truth


# ---- settings ------------------------------------------------------------------------------------


def _weak_tree() -> np.ndarray:
    """The tree ``x0 -> x1 -> x2``, ``x1 -> x3`` on 300 rows, ``x1 -> x2`` of weight 0.6 against
    noise of standard deviation 3: a Fisher-Z p-value between ``1e-5`` and ``1e-4`` given ``x1``'s
    other neighbors, which the moral-graph screen keeps at its default level and drops at
    ``1e-8``."""
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(300)
    x1 = x0 + rng.standard_normal(300)
    x2 = 0.6 * x1 + 3.0 * rng.standard_normal(300)
    x3 = -0.8 * x1 + rng.standard_normal(300)
    return np.column_stack([x0, x1, x2, x3])


TREE_PAIRS = {frozenset(pair) for pair in ((0, 1), (1, 2), (1, 3))}


def _pairs(out) -> set[frozenset[int]]:
    return {frozenset(edge) for edge in _directed(out)}


@pytest.fixture(scope="module")
def weak():
    pytest.importorskip("torch")
    X = _weak_tree()
    return X, andrey.calm(X)


def test_calm_screen_alpha_sets_the_candidates_only_with_the_screen(weak) -> None:
    """At ``screen_alpha=1e-8`` the weak edge is no candidate; without the screen every pair is
    one, whatever the level."""
    X, default = weak
    assert _pairs(default) == TREE_PAIRS
    assert _pairs(andrey.calm(X, screen_alpha=1e-8)) == TREE_PAIRS - {frozenset((1, 2))}
    assert _pairs(andrey.calm(X, use_moral_graph=False, screen_alpha=1e-8)) == TREE_PAIRS


def test_calm_thresholds_keep_a_subset_of_the_edges(weak) -> None:
    """A larger ``w_threshold`` drops an edge. A learned gate is a sigmoid, below one, while a
    required gate is held at one, so ``gate_threshold=1.0`` keeps only the required edge."""
    X, default = weak
    assert _directed(andrey.calm(X, w_threshold=0.7)) < _directed(default)
    knowledge = andrey.Knowledge(required=[(1, 2)])
    assert _directed(andrey.calm(X, gate_threshold=1.0, knowledge=knowledge)) == {(1, 2)}


def test_calm_a_larger_lambda1_gives_fewer_weights(weak) -> None:
    X, default = weak
    sparser = andrey.calm(X, lambda1=0.2)
    assert np.count_nonzero(sparser.weighted_adjacency) < np.count_nonzero(
        default.weighted_adjacency
    )


@pytest.mark.filterwarnings("ignore::andrey.SearchLimitWarning")
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("lr", 0.05),
        ("tau", 0.2),
        ("rho_init", 1.0),
        ("rho_mult", 100.0),
        ("rho_max", 1e-3),  # reached at the start, so the rounds end after the first
        ("h_tol", 1e3),  # cleared by the first round's residual, so the rounds end there
    ],
)
def test_calm_solver_settings_reach_the_optimizer(name, value) -> None:
    """On two short rounds, another value of each setting gives other weights."""
    pytest.importorskip("torch")
    X = _weak_tree()
    short = {"subproblem_iter": 100, "max_iter": 2}
    base = andrey.calm(X, **short).weighted_adjacency
    assert not np.array_equal(andrey.calm(X, **short, **{name: value}).weighted_adjacency, base)


def test_calm_h_final_is_the_residual_the_rounds_stop_on(sem) -> None:
    """``h_final`` is the acyclicity residual of the gates before the thresholds, the value the
    rounds compare with ``h_tol``, so a looser ``h_tol`` ends them at a larger one."""
    X, _, tight = sem
    loose = andrey.calm(X, h_tol=1e-2)
    assert 0 < tight.metadata["h_final"] <= 1e-8 < loose.metadata["h_final"] <= 1e-2


# ---- what the method promises --------------------------------------------------------------------


def test_calm_is_deterministic() -> None:
    pytest.importorskip("torch")
    X, _ = _random_sem(6, 800, 0.35, seed=3)
    first = andrey.calm(X, seed=0, subproblem_iter=100, max_iter=2)
    second = andrey.calm(X, seed=0, subproblem_iter=100, max_iter=2)
    assert first.structure.oriented_edges() != []
    assert first == second


def test_calm_rejects_a_non_finite_loss() -> None:
    """Data whose covariance overflows makes the loss non-finite, so the fit stops with an error."""
    pytest.importorskip("torch")
    X, _ = _random_sem(4, 300, 0.4, seed=1)
    with pytest.raises(ValueError, match="not finite"):
        andrey.calm(X * 1e170, use_moral_graph=False, subproblem_iter=100, max_iter=2)


def test_calm_leaves_the_global_torch_generator_alone() -> None:
    """CALM draws from its own seeded generator, so a caller's global torch stream is unchanged."""
    torch = pytest.importorskip("torch")
    X, _ = _random_sem(4, 300, 0.4, seed=1)
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    andrey.calm(X, seed=0, subproblem_iter=60, max_iter=1)
    assert torch.equal(torch.rand(3), expected)
