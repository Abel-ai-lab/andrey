"""The error guard of ``score_many_with_base``, the absolute scores BOSS and GRaSP compare.

Each score is the base's full score plus a Schur update. Where the update's error bound exceeds
``_SCORE_ERROR_LIMIT``, the full score serves the move instead, so every returned score stays
within the limit of ``BICScore.score``; on well-conditioned data the bound never reaches it and the
update serves every move, bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest

from andrey.core import score_delta
from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore


def _full(score: BICScore, v: int, base: tuple[int, ...], op: str, u: int) -> float:
    parents = sorted([*base, u]) if op == "add" else [p for p in base if p != u]
    return score.score(v, parents)


def _nearly_collinear() -> tuple[BICScore, int, tuple[int, ...]]:
    """Column 6 is a near-exact linear function of the base ``(0, 1, 2, 4, 5)``, whose columns 4
    and 5 are themselves near-exact combinations of columns 0-2."""
    rng = np.random.default_rng(0)
    n = 500
    z = rng.standard_normal((n, 4))
    x4 = z[:, 0] + z[:, 1] + 1e-3 * rng.standard_normal(n)
    x5 = z[:, 2] - z[:, 0] + 1e-3 * rng.standard_normal(n)
    v = z[:, 0] + 2 * x4 - x5 + 1e-3 * rng.standard_normal(n)
    x = np.column_stack([z, x4, x5, v])
    return BICScore.from_cov(np.cov(x, rowvar=False), n), 6, (0, 1, 2, 4, 5)


def test_ill_conditioned_updates_fall_back_to_full_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a nearly collinear base the update is off by far more than the limit; every returned
    score is within the limit of the full score."""
    score, v, base = _nearly_collinear()
    moves = [("add", 3)] + [("remove", u) for u in base]
    full = [_full(score, v, base, op, u) for op, u in moves]
    got = DeltaBICScore(score).score_many_with_base(v, base, moves)
    assert np.all(np.abs(np.subtract(got, full)) <= score_delta._SCORE_ERROR_LIMIT)

    monkeypatch.setattr(score_delta, "_SCORE_ERROR_LIMIT", np.inf)
    unguarded = DeltaBICScore(score).score_many_with_base(v, base, moves)
    assert np.abs(np.subtract(unguarded, full)).max() > 1e-3


@pytest.mark.parametrize("seed", [0, 1])
def test_well_conditioned_updates_keep_every_move(
    monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    """On independent columns no bound reaches the limit, so the scores equal the unguarded
    update's bit for bit."""
    x = np.random.default_rng(seed).standard_normal((600, 9))
    queries = [
        (v, base, [("add", u) for u in range(9) if u != v and u not in base])
        for v, base in [(0, (2,)), (0, (1, 3, 5, 7)), (4, (0, 1, 2, 3, 5, 6))]
    ]
    queries += [(v, base, [("remove", u) for u in base]) for v, base, _ in queries]
    guarded = [DeltaBICScore(BICScore(x)).score_many_with_base(*q) for q in queries]
    monkeypatch.setattr(score_delta, "_SCORE_ERROR_LIMIT", np.inf)
    assert guarded == [DeltaBICScore(BICScore(x)).score_many_with_base(*q) for q in queries]
