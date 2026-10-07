"""The PURE Schur delta-BIC batch primitive ``DeltaBICScore.delta_many_with_base``.

``delta_many_with_base(v, base, moves)`` returns RAW deltas ``score(v, base +/- u) - score(v,
base)`` for a batch of single add/remove moves. Unlike the chained ``score_many_with_base`` (HC /
BOSS), it is *pure*: it shares only the condition-gated ``inv(cov[S, S])`` cache (a pure function of
``S``), recomputes each residual fresh from that inverse, and never writes a chained residual or
a derived score. This file pins three contracts:

* **units** -- each raw delta is within ``1e-9`` relative of two full ``BICScore.score`` calls, for
  adds and removes, empty and non-empty bases;
* **fallback** -- an ill-conditioned base (``cond >= _COND_THRESHOLD``) sends every candidate to the
  two-full-score fallback (monkeypatch-counted), still exact;
* **purity** -- the same query returns byte-identical deltas even after the chained
  ``score_many_with_base`` has polluted the shared residual cache in between.
"""

from __future__ import annotations

import numpy as np
import pytest

from andrey.core import score_delta
from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore


def _well_conditioned(seed: int, n: int, d: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d))


def _full_delta(score: BICScore, v: int, base: tuple[int, ...], u: int, op: str) -> float:
    if op == "add":
        s_new = sorted([*base, u])
    else:
        s_new = [x for x in base if x != u]
    return float(score.score(v, s_new) - score.score(v, list(base)))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_add_deltas_match_two_full_scores(seed: int) -> None:
    """Batched add deltas equal two full scores within 1e-9 relative (well-conditioned)."""
    x = _well_conditioned(seed, 600, 9)
    score = BICScore(x, lambda_value=2.0)
    delta = DeltaBICScore(score)
    v = 0
    for base in [(), (2,), (1, 3), (1, 3, 5, 7)]:
        us = [u for u in range(9) if u != v and u not in base]
        got = delta.delta_many_with_base(v, base, [("add", u) for u in us])
        want = [_full_delta(score, v, base, u, "add") for u in us]
        for g, w in zip(got, want, strict=True):
            assert abs(g - w) <= 1e-9 * max(1.0, abs(w)), (base, g, w)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_remove_deltas_match_two_full_scores(seed: int) -> None:
    """Batched/scalar remove deltas equal two full scores within 1e-9 relative."""
    x = _well_conditioned(seed, 600, 9)
    score = BICScore(x, lambda_value=2.0)
    delta = DeltaBICScore(score)
    v = 0
    for base in [(2,), (1, 3), (1, 3, 5, 7)]:
        for u in base:
            got = delta.delta_many_with_base(v, base, [("remove", u)])[0]
            want = _full_delta(score, v, base, u, "remove")
            assert abs(got - want) <= 1e-9 * max(1.0, abs(want)), (base, u, got, want)


def test_empty_and_ordering_invariance() -> None:
    """Empty batch returns empty; per-candidate delta is independent of batch membership."""
    x = _well_conditioned(3, 500, 6)
    delta = DeltaBICScore(BICScore(x, lambda_value=2.0))
    assert delta.delta_many_with_base(0, (1, 2), []) == []
    solo = delta.delta_many_with_base(0, (1, 2), [("add", 4)])[0]
    batched = delta.delta_many_with_base(0, (1, 2), [("add", 3), ("add", 4), ("add", 5)])[1]
    assert solo == batched  # candidate 4's delta does not depend on who else is in the batch


def test_fallback_fires_on_ill_conditioned_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """A near-singular base (cond >= _COND_THRESHOLD) routes every candidate to two full scores."""
    rng = np.random.default_rng(0)
    base_cols = rng.standard_normal((400, 4))
    # Two columns collinear to ~1 part in 1e9 -> cov[base, base] condition well above 1e10.
    collinear = base_cols[:, 0:1] + 1e-11 * rng.standard_normal((400, 1))
    x = np.hstack([base_cols, collinear])  # column 4 ~= column 0
    score = BICScore(x, lambda_value=2.0)
    delta = DeltaBICScore(score)

    calls: list[tuple] = []
    real = score_delta._delta_two_scores

    def _counting(score_obj, i, base, u, op):
        calls.append((i, base, u, op))
        return real(score_obj, i, base, u, op)

    monkeypatch.setattr(score_delta, "_delta_two_scores", _counting)

    base = (0, 4)  # ill-conditioned (cols 0 and 4 collinear)
    got = delta.delta_many_with_base(1, base, [("remove", 0), ("remove", 4)])
    assert calls, "ill-conditioned base did not trigger the two-full-score fallback"
    # The fallback is still exact: it IS two full scores.
    want = [_full_delta(score, 1, base, 0, "remove"), _full_delta(score, 1, base, 4, "remove")]
    assert got == want


def test_purity_survives_chained_cache_pollution() -> None:
    """``delta_many_with_base`` ignores the chained residual cache ``score_many_with_base`` writes.

    HC / BOSS call the chained ``score_many_with_base``, which writes rank-one-updated residuals
    (history-dependent at the ULP) back to the shared cache. The pure primitive must not read them:
    the same query returns byte-identical deltas before and after that pollution.
    """
    x = _well_conditioned(1, 700, 8)
    score = BICScore(x, lambda_value=2.0)
    delta = DeltaBICScore(score)

    query = (0, (1, 2, 3), [("add", 4), ("add", 5), ("remove", 2)])
    before = delta.delta_many_with_base(*query)

    # Pollute the shared residual cache via the chained surface, over overlapping targets/bases.
    for _ in range(3):
        delta.score_many_with_base(0, (1, 2, 3), [("add", 4), ("add", 6)])
        delta.score_many_with_base(0, (1, 2, 3, 4), [("remove", 3), ("add", 5)])

    after = delta.delta_many_with_base(*query)
    assert before == after  # byte-identical, not merely close


def test_purity_repeated_interleaved_query_is_bit_identical() -> None:
    """Interleaving two different pure queries never perturbs either one's bits."""
    x = _well_conditioned(2, 600, 7)
    delta = DeltaBICScore(BICScore(x, lambda_value=2.0))
    q1 = (0, (1, 2), [("add", 3), ("add", 4)])
    q2 = (5, (1, 6), [("add", 0), ("remove", 6)])
    a1 = delta.delta_many_with_base(*q1)
    a2 = delta.delta_many_with_base(*q2)
    for _ in range(5):
        assert delta.delta_many_with_base(*q2) == a2
        assert delta.delta_many_with_base(*q1) == a1
