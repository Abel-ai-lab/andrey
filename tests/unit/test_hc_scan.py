"""Hill climbing's and tabu search's move scan picks what a full rescan picks.

:class:`andrey.search.hc.MoveScanner` keeps its deltas between passes, recomputes only the rows of
targets whose parents changed, and checks only its lowest candidates for cycles. The reference below
rescans every move each pass, with each target's row deltas computed afresh. A delta depends only on
its target and parent set, so equal picks and equal keys show the scanner's tables, cycle checks,
and tie-break are right. Exact ties break by each column's hash, so the graph found does not depend
on the column order.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

from andrey.core.score import BICScore, make_score
from andrey.core.score_delta import DeltaBICScore, FullScoreDelta, add_deltas, remove_deltas
from andrey.search._parallel import _EPS_IMPROVE, _KEY_DP
from andrey.search.hc import MoveScanner, _apply_move, _key_after, hc, tie_ranks
from andrey.search.tabu import tabu


def _reference_scan(score, adj_int, parents_of, d, floor, tabu, ranks):
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
            key = (cand[0], ranks[u], ranks[v], rank)
            if best is None or key < (best[0], ranks[best[1]], ranks[best[2]], best[3]):
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


def _categorical(seed: int, n: int, d: int) -> np.ndarray:
    """Codes in ``0..2``: each column copies the sum of earlier ones mod 3, or else is noise."""
    rng = np.random.default_rng(seed)
    parents = np.tril(rng.random((d, d)) < 0.3, -1)
    x = np.zeros((n, d), dtype=np.int64)
    for j in range(d):
        copy = x[:, parents[j]].sum(axis=1) % 3
        x[:, j] = np.where(rng.random(n) < 0.6, copy, rng.integers(0, 3, n))
    return x.astype(np.float64)


@pytest.mark.parametrize("seed", range(3))
def test_scanner_picks_as_full_rescan(seed: int) -> None:
    """Along a walk that adds, reverses, and drops edges, picks and keys match a full rescan.

    The walk takes the best move outside a short tabu list, worsening or not, and every third step
    drops a random edge, so targets return to earlier parent sets and adds that closed a cycle no
    longer do.
    """
    d = 12
    x = _data(seed, 400, d)
    ranks = tie_ranks(x)
    scanner = MoveScanner(DeltaBICScore(BICScore(x)), d, ranks=ranks)
    reference = BICScore(x)
    rng = np.random.default_rng(seed)
    adj_int = [0] * d
    parents_of: dict[int, list[int]] = {i: [] for i in range(d)}
    recent: list[tuple[int, ...]] = []
    for step in range(90):
        recent = [*recent[-5:], tuple(adj_int)]
        tabu = set(recent)
        for floor in (-_EPS_IMPROVE, math.inf):
            want = _reference_scan(reference, adj_int, parents_of, d, floor, tabu, ranks)
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
    scanner = MoveScanner(DeltaBICScore(BICScore(x)), 3, ranks=np.arange(3))
    adj_int, parents_of = [0b110, 0b100, 0], {0: [], 1: [0], 2: [0, 1]}
    assert scanner.best(adj_int, parents_of) is None
    assert scanner.best(adj_int, parents_of, floor=math.inf) == (0.0, 0, 1, 1, ("reverse", 0, 1))


@pytest.mark.parametrize("search", [hc, tabu])
def test_no_columns_give_an_empty_graph(search) -> None:
    graph, _ = search(np.zeros((10, 0)))
    assert graph.to_numpy().shape == (0, 0)


_TIED_DATA = {
    "local_score_BIC_from_cov": lambda seed: _data(seed, 500, 10),
    "local_score_BDeu": lambda seed: _categorical(seed, 1000, 10),
    "local_score_BIC_categorical": lambda seed: _categorical(seed, 1000, 10),
}


def _graph(search, x, score_func, order):
    """The CPDAG ``search`` finds on the columns of ``x`` in ``order``, mapped back."""
    back = np.argsort(order)
    return search(x[:, order], score_func=score_func)[0].to_numpy()[np.ix_(back, back)]


@pytest.mark.parametrize("search", [hc, tabu])
@pytest.mark.parametrize("score_func", list(_TIED_DATA))
def test_exact_ties_break_independently_of_column_order(search, score_func, monkeypatch) -> None:
    """Every column order gives the graph of the original order, mapped back.

    Under these score-equivalent scores, the first move adds an edge whose opposite direction has
    the same key, so ties decide the walk. Ranking the columns by position instead changes the graph
    under at least one of the orders, so the orders exercise the tie-break.
    """
    x = _TIED_DATA[score_func](1)
    d = x.shape[1]
    score = make_score(score_func, x)
    delta = DeltaBICScore(score) if isinstance(score, BICScore) else FullScoreDelta(score)
    adj_int, parents_of = [0] * d, {i: [] for i in range(d)}
    first = MoveScanner(delta, d, ranks=tie_ranks(x)).best(adj_int, parents_of)
    assert first is not None and first[4][0] == "add"
    _, u, v = first[4]
    assert round((score.score(u, [v]) - score.score(u, [])) / score.n, _KEY_DP) == first[0]

    rng = np.random.default_rng(0)
    orders = [np.arange(d)[::-1], *(rng.permutation(d) for _ in range(3))]
    want = _graph(search, x, score_func, np.arange(d))
    for order in orders:
        np.testing.assert_array_equal(_graph(search, x, score_func, order), want)

    module = importlib.import_module(search.__module__)
    monkeypatch.setattr(module, "tie_ranks", lambda x, categorical=None: np.arange(x.shape[1]))
    by_position = _graph(search, x, score_func, np.arange(d))
    assert any((_graph(search, x, score_func, o) != by_position).any() for o in orders)


def test_tie_ranks_move_with_the_columns() -> None:
    """Reordering distinct columns reorders their ranks; identical columns rank by position."""
    x = _data(0, 50, 6)
    order = np.array([3, 0, 5, 1, 4, 2])
    np.testing.assert_array_equal(tie_ranks(x[:, order]), tie_ranks(x)[order])
    np.testing.assert_array_equal(tie_ranks(x[:, [2, 2, 2]]), [0, 1, 2])


def test_recoding_categories_keeps_the_tie_ranks() -> None:
    """A categorical column ranks by which rows share a category, not by the codes."""
    x = _categorical(0, 200, 6)
    recoded = x * np.array([3.0, -1.0, 2.0, -5.0, 1.0, 7.0]) + np.arange(6.0)
    every = np.ones(6, dtype=bool)
    assert not np.array_equal(tie_ranks(recoded), tie_ranks(x))
    np.testing.assert_array_equal(tie_ranks(recoded, every), tie_ranks(x, every))


@pytest.mark.parametrize("search", [hc, tabu])
def test_rounding_in_the_last_bits_keeps_the_graph(search) -> None:
    """Data off by a few rounding units in every column, as one BLAS kernel's output is from
    another's, rank their columns the same and give the same graph."""
    x = _TIED_DATA["local_score_BIC_from_cov"](1)
    rng = np.random.default_rng(0)
    unit = np.finfo(np.float64).eps * np.abs(x).max(axis=0)
    nudged = x + rng.integers(-4, 5, size=x.shape) * unit
    assert (nudged != x).any(axis=0).all()
    np.testing.assert_array_equal(tie_ranks(nudged), tie_ranks(x))
    np.testing.assert_array_equal(search(nudged)[0].to_numpy(), search(x)[0].to_numpy())
