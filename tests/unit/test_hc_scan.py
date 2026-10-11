"""Hill climbing's and tabu search's move scan keeps deltas between passes without changing a pick.

:class:`andrey.search.hc.MoveScanner` computes a delta only when a move first needs it under the
target's current parents. The Schur delta reads a residual cache that earlier deltas write, so a
pick equal to a full rescan's also needs the deltas computed in the same order. The reference below
rescans every move each pass through a memo, the scan :class:`MoveScanner` replaces; each side has
its own scorer, so equal picks and equal keys mean equal delta values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore
from andrey.search._parallel import _EPS_IMPROVE, _KEY_DP
from andrey.search.hc import MoveScanner, _apply_move, _key_after


def _reference_scan(targets, adj_int, parents_of, delta, d, memo, floor, tabu):
    """Every legal move with delta below ``floor``, rescanned u-major, v-minor through ``memo``."""

    def cached(v, pa_v, u, op):
        key = (v, pa_v, op, u)
        if key not in memo:
            memo[key] = delta.delta(v, pa_v, u, op)
        return memo[key]

    def reaches(a, b):
        seen, todo = 0, adj_int[a]
        while todo:
            lsb = todo & -todo
            todo ^= lsb
            if not seen & lsb:
                seen |= lsb
                todo |= adj_int[lsb.bit_length() - 1]
        return bool(seen >> b & 1)

    pa = [tuple(parents_of[i]) for i in range(d)]
    inv_n = 1.0 / delta.score_obj.n
    best = None

    def offer(value, u, v, rank, op):
        nonlocal best
        if value < floor and _key_after(adj_int, (op, u, v)) not in tabu:
            cand = (round(value * inv_n, _KEY_DP), u, v, rank, (op, u, v))
            if best is None or cand[:4] < best[:4]:
                best = cand

    for u in range(d):
        for v in targets:
            if u == v:
                continue
            if adj_int[u] >> v & 1:
                d_rem = cached(v, pa[v], u, "remove")
                offer(d_rem, u, v, 0, "remove")
                others = adj_int[u] & ~(1 << v)
                if not any(others >> c & 1 and reaches(c, v) for c in range(d)):
                    offer(cached(u, pa[u], v, "add") + d_rem, u, v, 1, "reverse")
            elif not adj_int[v] >> u & 1 and not reaches(v, u):
                offer(cached(v, pa[v], u, "add"), u, v, 0, "add")
    return best


def _data(seed: int, n: int, d: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b = np.tril(rng.uniform(0.4, 1.2, (d, d)) * (rng.random((d, d)) < 0.3), -1)
    x = np.zeros((n, d))
    for j in range(d):
        x[:, j] = rng.standard_normal(n) + x @ b[j]
    return x


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("targets", [None, list(range(3, 9))])
def test_scanner_picks_as_full_rescan(seed: int, targets: list[int] | None) -> None:
    """Along a walk that revisits parent sets, picks, keys, and residual caches match a full rescan.

    A residual cache holds its keys in first-write order, so equal caches show the deltas were
    computed in the same order. The walk takes the best move outside a short tabu list, worsening
    or not, and every third step drops a random edge, returning its head to an earlier parent set.
    """
    d = 12
    x = _data(seed, 400, d)
    ours = DeltaBICScore(BICScore(x))
    scanner = MoveScanner(ours, d)
    reference = DeltaBICScore(BICScore(x))
    memo: dict = {}
    heads = range(d) if targets is None else targets
    rng = np.random.default_rng(seed)
    adj_int = [0] * d
    parents_of: dict[int, list[int]] = {i: [] for i in range(d)}
    recent: list[tuple[int, ...]] = []
    for step in range(90):
        recent = [*recent[-5:], tuple(adj_int)]
        tabu = set(recent)
        for floor in (-_EPS_IMPROVE, math.inf):
            want = _reference_scan(heads, adj_int, parents_of, reference, d, memo, floor, tabu)
            assert (
                scanner.best(adj_int, parents_of, targets=targets, floor=floor, tabu=tabu) == want
            )
        if targets is None:
            move = want
        else:  # step both sides over the whole range, in the same order
            move = scanner.best(adj_int, parents_of, floor=math.inf, tabu=tabu)
            assert move == _reference_scan(
                range(d), adj_int, parents_of, reference, d, memo, math.inf, tabu
            )
        assert ours.residual_cache == reference.residual_cache
        edges = [(u, v) for v in range(d) for u in parents_of[v]]
        if step % 3 == 2 and edges:
            u, v = edges[rng.integers(len(edges))]
            _apply_move(adj_int, parents_of, ("remove", u, v))
        elif move is not None:
            _apply_move(adj_int, parents_of, move[4])
    assert sum(len(p) for p in parents_of.values()) > 0
