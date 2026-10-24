"""Check grow-shrink branch priority, BOSS's tolerance for score ties, and column-order invariance.

``_GSTNode.trace`` takes the first branch allowed by the prefix. Branch order therefore determines
the search path. BOSS's grow and shrink steps break score ties by variable index; its order move
takes the leftmost tied position. Each step requires an improvement larger than ``_SCORE_EPS``.
BOSS and GRaSP search the columns sorted by tie rank, so a variable's index, the seeded starting
order, and the visiting order follow the columns' data, and reordering the columns relabels the
graph.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

import andrey
from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore
from andrey.data import sample_discrete, sample_mixed
from andrey.search import boss as boss_mod
from andrey.search import grasp as grasp_mod
from andrey.search.boss import boss, ranked_score
from andrey.search.grasp import grasp

D_NODES, N_SAMPLES = 12, 400


def _sample() -> np.ndarray:
    """Return a sample with several improving additions."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N_SAMPLES, D_NODES))
    for j in range(1, D_NODES):
        x[:, j] += 0.8 * x[:, j - 1]
    return x


def _descending(scores: list[float], tolerance: float = 0.0) -> bool:
    return all(a >= b - tolerance for a, b in zip(scores, scores[1:]))


def _exact_batch(gains: dict[int, float]) -> SimpleNamespace:
    """A batch scorer whose scores are the negated ``gains``, each a full score (bound 0)."""

    def score_many_with_base(vertex, parents, moves, *, bounds=None):
        if bounds is not None:
            bounds[:] = [0.0] * len(moves)
        return [-gains[u] for _, u in moves]

    return SimpleNamespace(score_many_with_base=score_many_with_base)


def _leaf(score: float) -> SimpleNamespace:
    """A traced leaf whose negated local score is ``score``, a full score."""
    return SimpleNamespace(score=score, bound=0.0)


def test_boss_grows_best_scoring_branch_first() -> None:
    """BOSS's tree orders a node's improving additions best-first, up to ``_SCORE_EPS`` ties."""
    x = _sample()
    score = BICScore(x, lambda_value=1.0)
    tree = boss_mod._GrowShrinkTree(0, score, DeltaBICScore(score), D_NODES)

    node = tree.root
    node._grow([v for v in range(D_NODES) if v != 0], [])

    # Require two branches so `_descending` cannot pass vacuously.
    assert len(node.branches) >= 2, "fixture yields too few improving additions to order"
    assert _descending([b.grow_score for b in node.branches], boss_mod._SCORE_EPS)


@pytest.mark.parametrize(
    "gains, expected",
    [
        ({0: 10.0, 1: 10.0 + 0.5e-6, 2: 0.5e-6, 3: 2e-6}, [0, 1, 3]),
        ({0: 10.0, 1: 10.0 + 0.75e-6, 2: 10.0 + 1.5e-6}, [1, 2, 0]),
    ],
)
def test_boss_grow_ties_use_variable_index_and_require_an_improvement(gains, expected) -> None:
    tree = SimpleNamespace(vertex=4, empty_score=0.0, delta=_exact_batch(gains))
    node = boss_mod._GSTNode(tree)

    node._grow(sorted(gains, reverse=True), [])

    assert [branch.add for branch in node.branches] == expected


def test_boss_shrink_ties_use_variable_index_and_require_an_improvement() -> None:
    eps = boss_mod._SCORE_EPS
    gains = {0: 10.0, 1: 10.0 + eps / 2}
    tree = SimpleNamespace(vertex=2, empty_score=0.0, delta=_exact_batch(gains), leaves={})
    node = boss_mod._GSTNode(tree)
    parents = [1, 0]

    node._shrink(parents)

    assert node.remove == [0]
    assert parents == [1]
    assert node.shrink_score == gains[0]


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (10.0, 10.0, [1, 0, 2]),
        (10.0, 10.0 + 0.5e-6, [1, 0, 2]),
        (10.0, 10.0 + 2e-6, [0, 2, 1]),
        (0.75e-6, 1.5e-6, [0, 2, 1]),
        (0.75e-6, 0.5e-6, [0, 1, 2]),
        (1e-6, 1e-6, [0, 1, 2]),
    ],
)
def test_boss_mutation_prefers_the_leftmost_improving_tied_position(left, right, expected) -> None:
    # Moving variable 1 to either end changes the total from zero to `left` or `right`.
    gsts = [
        SimpleNamespace(trace=lambda prefix: _leaf(left if 1 in prefix else 0)),
        SimpleNamespace(trace=lambda prefix: _leaf(right if 2 in prefix else 0)),
        SimpleNamespace(trace=lambda prefix: _leaf(0)),
    ]
    order = [0, 1, 2]
    positions = {v: v for v in order}

    assert boss_mod._better_mutation(1, order, gsts, positions) == (expected != [0, 1, 2])
    assert order == expected
    assert positions == {v: i for i, v in enumerate(expected)}


def test_grasp_grows_best_scoring_branch_first() -> None:
    """GRaSP's tree orders a node's improving additions best-first."""
    x = _sample()
    score = BICScore(x, lambda_value=1.0)
    tree = grasp_mod._GST(0, score, DeltaBICScore(score), D_NODES)

    node = tree.root
    node.grow([v for v in range(D_NODES) if v != 0], [])

    # Require two branches so `_descending` cannot pass vacuously.
    assert len(node.branches) >= 2, "fixture yields too few improving additions to order"
    assert _descending([b.grow_score for b in node.branches])


def _linear(seed: int, n: int = 400, d: int = 10) -> np.ndarray:
    """Linear-Gaussian data on a random DAG whose topological order is the column order."""
    rng = np.random.default_rng(seed)
    b = np.tril(rng.uniform(0.4, 1.2, (d, d)) * (rng.random((d, d)) < 0.3), -1)
    x = np.zeros((n, d))
    for j in range(d):
        x[:, j] = rng.standard_normal(n) + x @ b[j]
    return x


def _mixed() -> tuple[np.ndarray, dict]:
    """A conditional linear Gaussian network of 6 columns, with its degenerate Gaussian score."""
    ds = sample_mixed(d=6, n=1000, seed=2)
    mask = np.zeros(6, dtype=bool)
    mask[ds.provenance["config"]["categorical"]] = True
    return ds.data, {"score_func": "local_score_degen_gauss", "categorical": mask}


def _categorical(score_func: str, seed: int) -> tuple[np.ndarray, dict]:
    """A discrete network of 10 columns, with a categorical score."""
    return sample_discrete(d=10, n=1000, seed=seed).data, {"score_func": score_func}


_ORDER_CASES = {
    **{f"linear-{seed}": lambda seed=seed: (_linear(seed), {}) for seed in (1, 2, 5)},
    "mixed": _mixed,
    "bdeu": partial(_categorical, "local_score_BDeu", 2),
    "categorical-bic": partial(_categorical, "local_score_BIC_categorical", 4),
}


def _graph(search, x: np.ndarray, order: np.ndarray, **kwargs) -> np.ndarray:
    """The CPDAG ``search`` finds on the columns of ``x`` in ``order``, mapped back."""
    back = np.argsort(order)
    if "categorical" in kwargs:
        kwargs = {**kwargs, "categorical": kwargs["categorical"][order]}
    return search(x[:, order], **kwargs)[0].to_numpy()[np.ix_(back, back)]


_SEARCHES = {
    "boss": boss,
    **{f"grasp-depth{depth}": partial(grasp, depth=depth) for depth in (1, 2, 3)},
}


@pytest.mark.parametrize("search", list(_SEARCHES))
@pytest.mark.parametrize("case", list(_ORDER_CASES))
def test_reordering_the_columns_relabels_the_graph(search, case) -> None:
    """Every column order gives the graph of the stored order, mapped back.

    Ranking the columns by position instead changes the graph under at least one of 40 orders, so
    the orders reach choices that go by variable index.
    """
    search = _SEARCHES[search]
    x, kwargs = _ORDER_CASES[case]()
    d = x.shape[1]
    rng = np.random.default_rng(0)
    orders = [np.arange(d)[::-1], *(rng.permutation(d) for _ in range(39))]
    want = _graph(search, x, np.arange(d), **kwargs)
    for order in orders[:4]:
        np.testing.assert_array_equal(_graph(search, x, order, **kwargs), want)

    with andrey.config(tie_break="position"):
        by_position = _graph(search, x, np.arange(d), **kwargs)
        assert any((_graph(search, x, o, **kwargs) != by_position).any() for o in orders)


@pytest.mark.parametrize("case", list(_ORDER_CASES))
def test_ranks_move_with_the_columns_and_the_score_is_the_same(case) -> None:
    """A reordering moves each column's rank with it and leaves the sorted score bit for bit."""
    x, kwargs = _ORDER_CASES[case]()
    score_func = kwargs.get("score_func", "local_score_BIC_from_cov")
    mask = kwargs.get("categorical")
    d = x.shape[1]
    order = np.random.default_rng(1).permutation(d)
    score, ranks = ranked_score(x, score_func, 1.0, mask)
    moved, moved_ranks = ranked_score(
        np.asfortranarray(x[:, order]), score_func, 1.0, None if mask is None else mask[order]
    )
    np.testing.assert_array_equal(moved_ranks, ranks[order])
    sets = [(v, [u for u in range(d) if u != v][: k + 1]) for v in range(d) for k in range(3)]
    assert [moved.score(v, pa) for v, pa in sets] == [score.score(v, pa) for v, pa in sets]


@pytest.mark.parametrize("search", [boss, grasp])
def test_the_seed_still_matters(search) -> None:
    """Different seeds shuffle the sorted columns differently and can reach different graphs."""
    x = _linear(1)
    graphs = {search(x, random_state=seed)[0].to_numpy().tobytes() for seed in range(4)}
    assert len(graphs) > 1
