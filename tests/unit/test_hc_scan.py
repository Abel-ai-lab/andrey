"""Hill climbing's and tabu search's move scan picks what a full rescan picks.

:class:`andrey.search.hc.MoveScanner` keeps its deltas between passes, recomputes only the rows of
targets whose parents changed, and checks only its lowest candidates for cycles. The reference below
rescans every move each pass, with each target's row deltas computed afresh. A delta depends only on
its target and parent set, so equal picks and equal keys show the scanner's tables, cycle checks,
and tie-break are right.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore, add_deltas, remove_deltas
from andrey.search._parallel import _EPS_IMPROVE, _KEY_DP
from andrey.search.hc import MoveScanner, _apply_move, _key_after, hc
from andrey.search.tabu import tabu


def _reference_scan(score, adj_int, parents_of, d, floor, tabu):
    """Every legal move with delta below ``floor``, rescanned u-major, v-minor."""
    add = [add_deltas(score, v, parents_of[v]) for v in range(d)]
    remove = [remove_deltas(score, v, parents_of[v]) for v in range(d)]

    def reaches(a, b):
        seen, todo = 0, adj_int[a]
        while todo:
            lsb = todo & -todo
            todo ^= lsb
            if not seen & lsb:
                seen |= lsb
                todo |= adj_int[lsb.bit_length() - 1]
        return bool(seen >> b & 1)

    inv_n = 1.0 / score.n
    best = None

    def offer(value, u, v, rank, op):
        nonlocal best
        if value < floor and _key_after(adj_int, (op, u, v)) not in tabu:
            cand = (round(value * inv_n, _KEY_DP), u, v, rank, (op, u, v))
            if best is None or cand[:4] < best[:4]:
                best = cand

    for u in range(d):
        for v in range(d):
            if u == v:
                continue
            if adj_int[u] >> v & 1:
                offer(remove[v][u], u, v, 0, "remove")
                others = adj_int[u] & ~(1 << v)
                if not any(others >> c & 1 and reaches(c, v) for c in range(d)):
                    covered = parents_of[v] == sorted([*parents_of[u], u])
                    value = 0.0 if covered else add[u][v] + remove[v][u]
                    offer(value, u, v, 1, "reverse")
            elif not adj_int[v] >> u & 1 and not reaches(v, u):
                offer(add[v][u], u, v, 0, "add")
    return best


def _data(seed: int, n: int, d: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b = np.tril(rng.uniform(0.4, 1.2, (d, d)) * (rng.random((d, d)) < 0.3), -1)
    x = np.zeros((n, d))
    for j in range(d):
        x[:, j] = rng.standard_normal(n) + x @ b[j]
    return x


@pytest.mark.parametrize("seed", range(3))
def test_scanner_picks_as_full_rescan(seed: int) -> None:
    """Along a walk that adds, reverses, and drops edges, picks and keys match a full rescan.

    The walk takes the best move outside a short tabu list, worsening or not, and every third step
    drops a random edge, so targets return to earlier parent sets and adds that closed a cycle no
    longer do.
    """
    d = 12
    x = _data(seed, 400, d)
    scanner = MoveScanner(DeltaBICScore(BICScore(x)), d)
    reference = BICScore(x)
    rng = np.random.default_rng(seed)
    adj_int = [0] * d
    parents_of: dict[int, list[int]] = {i: [] for i in range(d)}
    recent: list[tuple[int, ...]] = []
    for step in range(90):
        recent = [*recent[-5:], tuple(adj_int)]
        tabu = set(recent)
        for floor in (-_EPS_IMPROVE, math.inf):
            want = _reference_scan(reference, adj_int, parents_of, d, floor, tabu)
            assert scanner.best(adj_int, parents_of, floor=floor, tabu=tabu) == want
        move = want
        edges = [(u, v) for v in range(d) for u in parents_of[v]]
        if step % 3 == 2 and edges:
            u, v = edges[rng.integers(len(edges))]
            _apply_move(adj_int, parents_of, ("remove", u, v))
        elif move is not None:
            _apply_move(adj_int, parents_of, move[4])
    assert sum(len(p) for p in parents_of.values()) > 0


@pytest.mark.parametrize("seed", range(8))
def test_reversing_a_covered_edge_ties_at_zero(seed: int) -> None:
    """Reversing a covered edge leads to an equivalent DAG: its delta is zero and never improves.

    On the complete DAG ``0 -> 1 -> 2, 0 -> 2`` with nearly collinear ``0`` and ``1``, every removal
    loses a strong effect and both legal reverses are covered. Their Schur deltas carry rounding
    noise far above ``_EPS_IMPROVE``, so only an exact zero leaves no improving move.
    """
    rng = np.random.default_rng(seed)
    n = 100_000
    w = rng.standard_normal(n)
    u = w + 1e-3 * rng.standard_normal(n)
    x = np.column_stack([w, u, 1000 * (u - w) + rng.standard_normal(n)])
    scanner = MoveScanner(DeltaBICScore(BICScore(x)), 3)
    adj_int, parents_of = [0b110, 0b100, 0], {0: [], 1: [0], 2: [0, 1]}
    assert scanner.best(adj_int, parents_of) is None
    assert scanner.best(adj_int, parents_of, floor=math.inf) == (0.0, 0, 1, 1, ("reverse", 0, 1))


@pytest.mark.parametrize("search", [hc, tabu])
def test_no_columns_give_an_empty_graph(search) -> None:
    graph, _ = search(np.zeros((10, 0)))
    assert graph.to_numpy().shape == (0, 0)
