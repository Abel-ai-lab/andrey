"""The error bounds of ``score_many_with_base`` and the decisions BOSS and GRaSP make with them.

Each batched score comes with a bound on its distance from ``BICScore.score``. BOSS and GRaSP
decide a comparison with the batched scores wherever the bounds leave its side unchanged, and by
full scores elsewhere, so every decision is the one full scores make.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore
from andrey.search import boss as boss_mod
from andrey.search import grasp as grasp_mod

EPS = boss_mod._SCORE_EPS


def _full(score: BICScore, v: int, base: tuple[int, ...], op: str, u: int) -> float:
    parents = sorted([*base, u]) if op == "add" else [p for p in base if p != u]
    return score.score(v, parents)


def _nearly_collinear(noise: float) -> tuple[BICScore, int, tuple[int, ...]]:
    """Column 6 is close to a linear function of the base ``(0, 1, 2, 4, 5)``, whose columns 4
    and 5 are themselves close to combinations of columns 0-2; ``noise`` sets how close."""
    rng = np.random.default_rng(0)
    n = 500
    z = rng.standard_normal((n, 4))
    x4 = z[:, 0] + z[:, 1] + noise * rng.standard_normal(n)
    x5 = z[:, 2] - z[:, 0] + noise * rng.standard_normal(n)
    v = z[:, 0] + 2 * x4 - x5 + noise * rng.standard_normal(n)
    x = np.column_stack([z, x4, x5, v])
    return BICScore.from_cov(np.cov(x, rowvar=False), n), 6, (0, 1, 2, 4, 5)


def _scored(score: BICScore, v: int, base: tuple[int, ...], moves: list) -> tuple[list, list, list]:
    bounds: list[float] = []
    got = DeltaBICScore(score).score_many_with_base(v, base, moves, bounds=bounds)
    return got, bounds, [_full(score, v, base, op, u) for op, u in moves]


def test_bounds_cover_the_distance_from_full_scores() -> None:
    """On a nearly collinear base the updates are off by more than the tolerance, and each
    bound covers its update's distance from the full score."""
    score, v, base = _nearly_collinear(1e-2)
    moves = [("add", 3)] + [("remove", u) for u in base]
    got, bounds, full = _scored(score, v, base, moves)
    distance = np.abs(np.subtract(got, full))
    assert distance.max() > EPS
    assert np.all(distance <= bounds)


@pytest.mark.parametrize("seed", [0, 1])
def test_well_conditioned_bounds_stay_far_below_the_tolerance(seed: int) -> None:
    x = np.random.default_rng(seed).standard_normal((600, 9))
    score = BICScore(x)
    for v, base in [(0, ()), (0, (2,)), (0, (1, 3, 5, 7)), (4, (0, 1, 2, 3, 5, 6))]:
        moves = [("add", u) for u in range(9) if u != v and u not in base]
        moves += [("remove", u) for u in base]
        got, bounds, full = _scored(score, v, base, moves)
        assert np.all(np.abs(np.subtract(got, full)) <= bounds)
        assert max(bounds) < 1e-9


def test_full_scores_report_a_zero_bound() -> None:
    """A base too ill-conditioned for the update, and a move that is no single add, are scored
    in full."""
    score, v, base = _nearly_collinear(1e-6)
    moves = [("add", 3), ("add", base[0])] + [("remove", u) for u in base]
    got, bounds, full = _scored(score, v, base, moves)
    assert got == full
    assert bounds == [0.0] * len(moves)


def _threshold_score(gain: float) -> BICScore:
    """A score under which adding parent 2 to ``[1]`` of variable 0 improves by ``gain``."""
    x = np.random.default_rng(0).standard_normal((100, 4))
    x[:, 1] += x[:, 0]
    x[:, 2] += x[:, 1]
    x[:, 3] += x[:, 0] + x[:, 2]
    raw = BICScore(x, lambda_value=0.0)
    lam = (raw.score(0, [1]) - raw.score(0, [1, 2]) - gain) / raw.log_n
    return BICScore.from_cov(raw.cov, raw.n, lambda_value=lam)


@pytest.mark.parametrize(
    "margin, candidate_error, base_error",
    [(2e-7, 4e-7, 0.0), (-2e-7, -4e-7, 0.0), (2e-7, 0.0, 4e-7), (-2e-7, 0.0, -4e-7)],
    ids=["missed-add", "false-add", "base-too-high", "base-too-low"],
)
def test_boss_grow_decides_an_open_threshold_by_full_scores(
    monkeypatch: pytest.MonkeyPatch, margin: float, candidate_error: float, base_error: float
) -> None:
    """Adding parent 2 improves by ``EPS + margin``; an error within the reported bound moves the
    batched comparison to the other side of the tolerance, and the full scores decide it."""
    parents = [1]
    score = _threshold_score(EPS + margin)
    delta = DeltaBICScore(score)
    tree = boss_mod._GrowShrinkTree(0, score, delta, 4)
    exact = -score.score(0, parents)
    node = boss_mod._GSTNode(tree, score=exact + base_error, bound=abs(base_error))

    def rounded_batch(vertex, base_parents, deltas, *, bounds):
        bounds[:] = [abs(candidate_error)] * len(deltas)
        return [
            score.score(vertex, [*base_parents, u]) + (candidate_error if u == 2 else 0.0)
            for _, u in deltas
        ]

    monkeypatch.setattr(delta, "score_many_with_base", rounded_batch)
    batched_gain = -rounded_batch(0, parents, [("add", 2)], bounds=[])[0] - node.grow_score
    assert (batched_gain > EPS) != (margin > 0)

    node._grow([2, 3], parents)

    assert (2 in [branch.add for branch in node.branches]) == (margin > 0)


def _stub_tree(
    exact: dict[int, float],
    error: dict[int, float],
    bound: float | dict[int, float],
    rescored: list,
    *,
    grow: bool,
) -> SimpleNamespace:
    """A tree whose moves gain ``exact``, keyed by the variable a grow adds to no parents or a
    shrink removes from all of ``exact``. The batch returns the gains with ``error``, within
    ``bound`` (one for all moves or one per variable); the full score returns them exactly and
    records the variable."""

    def score_many_with_base(vertex, parents, moves, *, bounds):
        bounds[:] = [bound[u] if isinstance(bound, dict) else bound for _, u in moves]
        return [-(exact[u] + error[u]) for _, u in moves]

    def full(vertex, parents):
        (u,) = set(parents) if grow else set(exact) - set(parents)
        rescored.append(u)
        return -exact[u]

    return SimpleNamespace(
        vertex=9,
        empty_score=0.0,
        required=[],
        leaves={},
        delta=SimpleNamespace(score_many_with_base=score_many_with_base),
        score=SimpleNamespace(score=full),
    )


def test_boss_grow_groups_open_ties_by_full_scores() -> None:
    """The full gains of 0 and 1 are within the tolerance, a tie the variable index breaks; the
    batched gains are farther apart, but within their bounds of a tie."""
    rescored: list[int] = []
    tree = _stub_tree(
        {0: 10.0, 1: 10.0 + 0.9e-6}, {0: -0.1e-6, 1: 0.1e-6}, 0.2e-6, rescored, grow=True
    )
    node = boss_mod._GSTNode(tree)

    node._grow([1, 0], [])

    assert [branch.add for branch in node.branches] == [0, 1]
    assert sorted(rescored) == [0, 1]


def test_boss_grow_groups_a_full_score_with_a_batched_one() -> None:
    """A full score (bound 0) ties a batched one within the tolerance; the batched gain is farther,
    within its bound, and only the batched candidate is rescored."""
    rescored: list[int] = []
    tree = _stub_tree(
        {0: 10.0, 1: 10.0 + 0.9e-6}, {0: 0.0, 1: 0.2e-6}, {0: 0.0, 1: 0.25e-6}, rescored, grow=True
    )
    node = boss_mod._GSTNode(tree)

    node._grow([1, 0], [])

    assert [branch.add for branch in node.branches] == [0, 1]
    assert rescored == [1]


def test_boss_shrink_decides_an_open_threshold_by_full_scores() -> None:
    """Removing 0 improves the node's score by 0.9 times the tolerance, not enough; the batched
    gain is above the tolerance, within its bound."""
    rescored: list[int] = []
    tree = _stub_tree(
        {0: 0.9 * EPS, 1: -5.0}, {0: 0.2 * EPS, 1: 0.0}, 0.3 * EPS, rescored, grow=False
    )
    node = boss_mod._GSTNode(tree)
    parents = [0, 1]

    node._shrink(parents)

    assert node.remove == []
    assert parents == [0, 1]
    assert rescored == [0]


class _Leaf:
    """A traced leaf whose batched score is ``exact + error``, within ``bound``."""

    def __init__(self, exact: float, error: float = 0.0, bound: float = 0.0) -> None:
        self.exact = exact
        self.score = exact + error
        self.bound = bound

    def settle(self) -> None:
        self.score = self.exact
        self.bound = 0.0


@pytest.mark.parametrize("gain, error", [(0.9 * EPS, 0.2 * EPS), (1.1 * EPS, -0.2 * EPS)])
def test_boss_mutation_decides_an_open_move_by_full_scores(gain: float, error: float) -> None:
    """Moving variable 1 past 2 changes the total by ``gain``; the batched total is on the other
    side of the tolerance, within its bound."""
    zero = _Leaf(0.0)
    moved = _Leaf(gain, error, abs(error) + 0.05 * EPS)
    gsts = [
        SimpleNamespace(trace=lambda prefix: zero),
        SimpleNamespace(trace=lambda prefix: moved if 2 in prefix else zero),
        SimpleNamespace(trace=lambda prefix: zero),
    ]
    order = [0, 1, 2]
    positions = {v: v for v in order}

    assert boss_mod._better_mutation(1, order, gsts, positions) == (gain > EPS)
    assert order == ([0, 2, 1] if gain > EPS else [0, 1, 2])
    assert moved.bound == 0.0


@pytest.mark.parametrize("gain, error", [(0.5 * EPS, -2 * EPS), (-0.5 * EPS, 2 * EPS)])
def test_grasp_grow_rechecks_a_gap_within_the_bounds(
    monkeypatch: pytest.MonkeyPatch, gain: float, error: float
) -> None:
    """Adding parent 2 changes the score by ``gain``; the batched change is beyond the tolerance
    on the other side, within its bound, and the full scores decide it."""
    parents = [1]
    score = _threshold_score(gain)
    delta = DeltaBICScore(score)
    tree = grasp_mod._GST(0, score, delta, 4)
    node = grasp_mod._GSTNode(tree, score=-score.score(0, parents))

    def rounded_batch(vertex, base_parents, deltas, *, bounds):
        bounds[:] = [abs(error) + 0.1 * EPS if u == 2 else 0.0 for _, u in deltas]
        return [
            score.score(vertex, [*base_parents, u]) - (error if u == 2 else 0.0) for _, u in deltas
        ]

    monkeypatch.setattr(delta, "score_many_with_base", rounded_batch)

    node.grow([2, 3], parents)

    assert (2 in [branch.add for branch in node.branches]) == (gain > 0)


def test_grasp_grow_orders_siblings_within_their_bounds_by_full_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sibling 2 scores above sibling 1 by less than the tolerance; the batched scores put 1
    above 2 by more than it, within their bounds."""
    cov = np.array([[1.0, 0.6, 0.6 + 2.7e-9], [0.6, 1.0, 0.2], [0.6 + 2.7e-9, 0.2, 1.0]])
    score = BICScore.from_cov(cov, 100)
    delta = DeltaBICScore(score)
    tree = grasp_mod._GST(0, score, delta, 3)
    exact = {u: -score.score(0, [u]) for u in (1, 2)}
    assert 0 < exact[2] - exact[1] < EPS
    shift = (exact[2] - exact[1] + 1.5 * EPS) / 2  # puts 1 above 2 by 1.5 * EPS
    error = {1: shift, 2: -shift}

    def rounded_batch(vertex, parents, deltas, *, bounds):
        bounds[:] = [shift + 0.1 * EPS] * len(deltas)
        return [score.score(vertex, [u]) - error[u] for _, u in deltas]

    monkeypatch.setattr(delta, "score_many_with_base", rounded_batch)

    tree.root.grow([1, 2], [])

    assert [branch.add for branch in tree.root.branches] == [2, 1]
