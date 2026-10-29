"""Hill climbing over DAGs, the engine behind :func:`andrey.hc`.

Under BIC, candidate moves use Schur-complement deltas from :mod:`andrey.core.score_delta`; under
the categorical and mixed-data scores, the difference of two full local scores. :class:`MoveScanner`
caches them between passes, recomputing only changed targets. Output is the DAG's CPDAG and total
score; lower is better.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Collection
from itertools import compress
from operator import ne

import numpy as np

from andrey.core import ARROW, NULL, TAIL, GraphStructure, backend
from andrey.core.knowledge import Constraints, mpdag
from andrey.core.mixed import column_mask
from andrey.core.orient import dag2cpdag, to_structure
from andrey.core.score import (
    DISCRETE_SCORES,
    MIXED_SCORES,
    SCORES,
    BICScore,
    LocalScore,
    make_score,
)
from andrey.core.score_delta import DeltaBICScore, FullScoreDelta, add_deltas, remove_deltas
from andrey.core.seeding import start_rng
from andrey.core.warning_policy import warn_search_limit
from andrey.search._parallel import _EPS_IMPROVE, _KEY_DP

# Moves are compared by ``delta / n_samples`` rounded to ``_KEY_DP`` decimal places; ``delta`` is
# the change in score, with lower values preferred. Under a score-equivalent score, adding
# ``x -> y`` or ``y -> x`` from the empty graph ties mathematically, but computing their score
# changes can introduce different rounding errors. If the rounded values tie, the lower
# ``(ranks[u], ranks[v], op_rank)`` wins, with ``ranks`` from :func:`tie_ranks`.
# Schur updates compute BIC score changes from covariance submatrices. Their rounding error scales
# as ~n_samples * cond(cov) * eps, where cond(cov) is the covariance condition number and eps is
# machine precision. Dividing by n_samples leaves ~cond(cov) * eps; ``_KEY_DP`` = 9 usually absorbs
# this error for condition numbers up to ~1e5, independent of sample count. Rounding mitigates
# numerical differences but can still split a tie within one grid step: a value near a rounding
# boundary can round either way.
# Rounding preserves the order of values that remain distinct; the reported score is recomputed
# from the final structure.
__all__ = ["hc"]

#: The local scores :func:`hc` accepts.
HC_SCORES: tuple[str, ...] = SCORES

#: A later start replaces the best DAG so far only when its total score is lower by more than this.
START_EPS = 1e-6


# (delta, u, v, op_rank, (op, u, v)). op_rank: remove 0 < reverse 1; add is 0 and never shares
# ``(u, v)`` with a remove.
ScannedMove = tuple[float, int, int, int, tuple[str, int, int]]

# Out-neighbor bitset rows, ``tuple(adj_int)``, used as tabu keys.
StructureKey = tuple[int, ...]


def tie_ranks(x: np.ndarray, categorical: np.ndarray | None = None) -> np.ndarray:
    """Each column's rank for choosing between moves with equal rounded score changes.

    A continuous column's pattern records whether each value rises, falls, or stays equal to the
    preceding row's value, using comparisons rather than subtraction. A categorical column's
    pattern assigns codes in order of first appearance, so changing category labels preserves the
    pattern. Columns are ordered by a hash of their pattern, then by a hash of their values if
    pattern hashes match. For example, a column and a strictly increasing function of it share a
    pattern. If both hashes match, as for identical columns, column position decides. Otherwise,
    each rank moves with its column when the columns are reordered. The rank is arbitrary with
    respect to causal structure.

    Distinct pattern hashes give the same ranks across machines when the patterns are unchanged.
    Last-bit differences in data, such as those from different CPUs' BLAS calculations, preserve
    continuous patterns unless they change a comparison between consecutive rows. If pattern
    hashes match, the fallback to raw values can make ranks sensitive to these differences or to
    category recoding.

    Under ``tie_break="position"`` (:data:`andrey.config`, ``ANDREY_TIE_BREAK``), each column's
    rank is its position, so the methods that order ties by these ranks order them by column
    position.

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_features)
        Data matrix; rows are observations, columns are variables.
    categorical : ndarray of bool, shape (n_features,), optional
        Marks columns that use first-appearance category codes. ``None`` treats all columns as
        continuous.

    Returns
    -------
    ndarray of int, shape (n_features,)
        Each column's position in the tie-break order, from zero to ``n_features - 1``. Lower ranks
        take precedence when moves have equal rounded score changes.
    """
    if backend.config.tie_break == "position":
        return np.arange(x.shape[1])
    mask = np.zeros(x.shape[1], dtype=bool) if categorical is None else categorical
    patterns = []
    for col, is_categorical in zip(x.T, mask.tolist(), strict=True):
        if is_categorical:
            _, first, codes = np.unique(col, return_index=True, return_inverse=True)
            seen = np.empty(first.size, dtype=np.int64)
            seen[np.argsort(first)] = np.arange(first.size)
            pattern = seen[codes]
        else:
            col = np.ascontiguousarray(col)
            pattern = (col[1:] > col[:-1]).astype(np.int8) - (col[1:] < col[:-1])
        patterns.append(hashlib.blake2b(pattern.tobytes(), digest_size=8).digest())
    shared = {key for key, count in Counter(patterns).items() if count > 1}
    digests = [
        (key, hashlib.blake2b(col.tobytes(), digest_size=8).digest() if key in shared else b"")
        for key, col in zip(patterns, x.T, strict=True)
    ]
    order = sorted(range(len(digests)), key=digests.__getitem__)
    ranks = np.empty(len(order), dtype=np.intp)
    ranks[order] = np.arange(len(order))
    return ranks


def _key_after(adj_int: list[int], move: tuple[str, int, int]) -> StructureKey:
    """The structure key of the DAG that ``move`` leads to; ``adj_int`` itself is unchanged."""
    op, u, v = move
    rows = list(adj_int)
    if op == "add":
        rows[u] |= 1 << v
    else:
        rows[u] &= ~(1 << v)
        if op == "reverse":
            rows[v] |= 1 << u
    return tuple(rows)


class MoveScanner:
    """The best single-edge move of a pass, from delta tables kept between passes.

    ``add[v, u]`` and ``rem[v, u]`` depend only on target ``v`` and its parent set. Changed parent
    sets trigger one vectorized Schur update per affected row
    (:func:`~andrey.core.score_delta.add_deltas`, :func:`~andrey.core.score_delta.remove_deltas`).
    Every scanner computes the same deltas for the same target and parents.

    ``single[v, u]`` is the add or remove delta of ``u -> v``, or ``inf`` when ``v -> u`` exists or
    ``v`` reaches ``u``; ``low[v]`` is the row minimum. Reverses use the edge list and sum an add of
    ``v`` at target ``u`` with a remove of ``u`` at target ``v``. A covered edge ``u -> v``
    (``pa(v) = pa(u) + u``) reverses to a Markov equivalent DAG with equal score under BIC, the
    categorical scores, and the mixed-data scores, all score-equivalent. Its finite delta is set to
    zero to prevent rounding noise exceeding ``_EPS_IMPROVE`` from creating a false improvement. A
    BIC delta is ``inf`` when the move enters an unavailable family
    (:func:`~andrey.core.score.local_score_bic`), so no move does.

    ``forbidden[v, u]`` marks edges ``u -> v`` no add or reverse creates; removing one stays
    legal. ``None`` forbids none. ``required[v, u]`` marks edges ``u -> v`` no move removes or
    reverses; the caller starts from a DAG that has them. ``None`` requires none. ``ranks[u]`` is
    column ``u``'s place in the tie-break order, :func:`tie_ranks` of the data. With
    ``max_parents``, a node holding that many parents gains none: no add into it, and no reverse
    that turns one of its out-edges into a parent; removes stay legal.

    ``reach[x, y]`` records whether ``x`` reaches ``y``. Added edges extend it; removed edges
    trigger recomputation for nodes that reached them and rebuild rows whose targets reach fewer
    nodes. Added edges can only invalidate adds. A row stays valid until its lowest add closes a
    cycle, then drops every node its target reaches. A pass checks only its lowest candidates for
    cycles.
    """

    def __init__(
        self,
        delta: DeltaBICScore | FullScoreDelta,
        d: int,
        *,
        ranks: np.ndarray,
        forbidden: np.ndarray | None = None,
        required: np.ndarray | None = None,
        max_parents: int | None = None,
    ) -> None:
        self._delta = delta
        self._score = delta.score_obj
        self._d = d
        self._inv_n = 1.0 / self._score.n
        self._ranks: list[int] = np.asarray(ranks).tolist()
        self._add = np.full((d, d), np.inf)
        self._rem = np.full((d, d), np.inf)
        self._parent = np.zeros((d, d), dtype=bool)  # [v, u]: the edge u -> v exists
        self._single = np.full((d, d), np.inf)
        self._low = np.full(d, np.inf)
        self._tails = np.zeros(0, dtype=np.intp)  # the edge list, tails[i] -> heads[i]
        self._heads = np.zeros(0, dtype=np.intp)
        self._cyclic = np.zeros(0, dtype=bool)  # reversing edge i closes a cycle
        self._reach = np.zeros((d, d), dtype=bool)
        self._adj: list[int] | None = None  # the DAG the tables describe
        # Ids of parent sets: u -> v is covered iff set_of[v] == set_with[u], the id of pa(u) + u.
        self._ids: dict[tuple[int, ...], int] = {}
        self._set_of = np.zeros(d, dtype=np.intp)
        self._set_with = np.zeros(d, dtype=np.intp)
        # [v, u]: the edge u -> v is never added, nor made by a reverse.
        self._forbidden = np.zeros((d, d), dtype=bool) if forbidden is None else forbidden
        # [v, u]: the edge u -> v is never removed, nor reversed.
        self._required = required
        self._max_parents = max_parents

    def best(
        self,
        adj_int: list[int],
        parents_of: dict[int, list[int]],
        *,
        floor: float = -_EPS_IMPROVE,
        tabu: Collection[StructureKey] | None = None,
    ) -> ScannedMove | None:
        """The best legal move with delta below ``floor``.

        Moves are keyed ``(delta / n_samples rounded to _KEY_DP places, ranks[u], ranks[v],
        op_rank)``, a strict total order that moves with the columns. The default ``floor`` admits
        improving moves only; ``floor=inf`` admits every legal move. A move that leads to a
        structure in ``tabu`` is skipped.
        """
        if self._d < 2:
            return None
        self._sync(adj_int, parents_of)
        low = self._low.copy()
        tails, heads = self._tails, self._heads
        with np.errstate(invalid="ignore"):  # a degenerate row's deltas may be infinite
            rev = self._add[tails, heads] + self._rem[heads, tails]
        rev[(self._set_of[heads] == self._set_with[tails]) & np.isfinite(rev)] = 0.0
        rev[np.isnan(rev) | self._cyclic | self._forbidden[tails, heads]] = np.inf
        if self._required is not None:
            rev[self._required[heads, tails]] = np.inf
        if self._max_parents is not None:  # a reverse gives the tail a parent
            rev[self._parent[tails].sum(axis=1) >= self._max_parents] = np.inf
        held: list[tuple[int, int, float]] = []  # tabu moves set aside for this call
        try:
            while True:
                best = self._pick(low, rev, floor)
                if best is None or tabu is None or _key_after(adj_int, best[4]) not in tabu:
                    return best
                _, u, v, rank, _ = best
                if rank:
                    rev[(tails == u) & (heads == v)] = np.inf
                else:
                    held.append((v, u, float(self._single[v, u])))
                    self._single[v, u] = np.inf
                    self._low[v] = low[v] = self._single[v].min()
        finally:
            for v, u, value in reversed(held):
                self._single[v, u] = value
                self._low[v] = self._single[v].min()

    def _sync(self, adj_int: list[int], parents_of: dict[int, list[int]]) -> None:
        """Bring the tables, the edge list, and ``reach`` up to ``adj_int``."""
        old = self._adj
        if old == adj_int:
            return
        d = self._d
        first = old is None
        old = [0] * d if old is None else old
        gained: list[tuple[int, int]] = []
        lost: list[tuple[int, int]] = []
        for u in range(d) if first else compress(range(d), map(ne, adj_int, old)):
            gained += [(u, v) for v in _bits(adj_int[u] & ~old[u])]
            lost += [(u, v) for v in _bits(old[u] & ~adj_int[u])]
        for v in range(d) if first else {v for _, v in gained + lost}:
            pa = parents_of[v]
            self._add[v], self._rem[v] = self._deltas(v, pa)
            ids = self._ids
            self._set_of[v] = ids.setdefault(tuple(pa), len(ids))
            self._set_with[v] = ids.setdefault(tuple(sorted([*pa, v])), len(ids))
        reach = self._reach
        rows = set(range(d)) if first else {x for pair in gained + lost for x in pair}
        if lost:
            for u, v in lost:
                self._parent[v, u] = False
            rows.update(self._unreach([u for u, _ in lost]))
            keep = np.ones(self._tails.size, dtype=bool)
            for a, b in lost:
                keep &= (self._tails != a) | (self._heads != b)
            self._tails, self._heads = self._tails[keep], self._heads[keep]
            self._cyclic = np.zeros(self._tails.size, dtype=bool)
        for a, b in gained:
            self._parent[b, a] = True
            if not reach[a, b]:  # a now reaches b, and so does every node that reaches a
                above = reach[:, a].copy()
                above[a] = True
                below = reach[b].copy()
                below[b] = True
                reach[above] |= below
        pairs = np.array(gained, dtype=np.intp).reshape(-1, 2)
        self._tails = np.concatenate((self._tails, pairs[:, 0]))
        self._heads = np.concatenate((self._heads, pairs[:, 1]))
        self._cyclic = np.concatenate((self._cyclic, np.zeros(len(gained), dtype=bool)))
        for v in rows:
            row = np.where(self._parent[v], self._rem[v], self._add[v])
            row[self._parent[:, v] | reach[v] | np.isnan(row)] = np.inf
            row[self._forbidden[v] & ~self._parent[v]] = np.inf  # forbidden adds; removes stay
            if self._required is not None:
                row[self._required[v]] = np.inf  # required edges are never removed
            if self._max_parents is not None and len(parents_of[v]) >= self._max_parents:
                row[~self._parent[v]] = np.inf  # a full parent set takes no add; removes stay
            self._single[v] = row
            self._low[v] = row.min()
        self._adj = list(adj_int)

    def _deltas(self, v: int, pa: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Every add delta and every remove delta at target ``v`` with parents ``pa``.

        BIC takes the vectorized Schur updates. A score without one, such as BDeu, differences two
        full local scores per candidate, all scored in one :meth:`score_many` call. Entries that are
        not a legal add or remove, or are a forbidden add, are ``inf``.
        """
        delta = self._delta
        if isinstance(delta, DeltaBICScore):
            cond = delta.cond_threshold
            return (
                add_deltas(delta.score_obj, v, pa, cond_threshold=cond),
                remove_deltas(delta.score_obj, v, pa, cond_threshold=cond),
            )
        adds = [u for u in range(self._d) if u != v and u not in pa and not self._forbidden[v, u]]
        sets = [pa, *(sorted([*pa, u]) for u in adds), *([p for p in pa if p != u] for u in pa)]
        scores = np.array(self._score.score_many(v, sets))
        add = np.full(self._d, np.inf)
        rem = np.full(self._d, np.inf)
        add[adds] = scores[1 : 1 + len(adds)] - scores[0]
        rem[pa] = scores[1 + len(adds) :] - scores[0]
        return add, rem

    def _unreach(self, tails: list[int]) -> list[int]:
        """Recompute ``reach`` after edges out of ``tails`` went; return the nodes it changed for.

        Only a node that reached a tail, or is one, can reach fewer nodes. Such a node reaches
        another only if it did before, and then it reached more nodes than the other did, so
        visiting them by their old count, fewest first, visits every child before its parents.
        """
        reach, parent = self._reach, self._parent
        hit = np.logical_or.reduce(reach[:, tails], axis=1)
        hit[tails] = True
        nodes = np.flatnonzero(hit)
        changed = []
        for x in nodes[np.argsort(reach[nodes].sum(axis=1), kind="stable")].tolist():
            kids = parent[:, x]
            row = reach[kids].any(axis=0) | kids
            if not np.array_equal(row, reach[x]):
                reach[x] = row
                changed.append(x)
        return changed

    def _drop_reached(self, v: int, low: np.ndarray) -> None:
        """Drop cycle-closing adds from row ``v`` and update its minimum."""
        self._single[v, self._reach[v]] = np.inf
        self._low[v] = low[v] = self._single[v].min()

    def _reverse_ok(self, u: int, v: int) -> bool:
        """Whether reversing ``u -> v`` keeps the graph acyclic: ``u`` reaches ``v`` only by it."""
        return not self._reach[self._parent[:, u], v].any()  # no child of u reaches v

    def _pick(self, low: np.ndarray, rev: np.ndarray, floor: float) -> ScannedMove | None:
        """The legal move with the lowest key among those below ``floor``.

        ``rev`` scores reversing each listed edge. A row whose candidate closes a cycle drops every
        such add, and a reverse that closes one is marked, until an edge goes.
        """
        single, parent, reach = self._single, self._parent, self._reach
        tails, heads = self._tails, self._heads
        lowest = np.inf
        while True:
            v = int(np.argmin(low))
            if not low[v] < floor:
                break
            u = int(np.argmin(single[v]))
            if not reach[v, u]:  # a remove, or an add that closes no cycle
                lowest = float(low[v])
                break
            self._drop_reached(v, low)
        while rev.size:
            i = int(np.argmin(rev))
            if not rev[i] < min(lowest, floor):
                break
            if self._reverse_ok(int(tails[i]), int(heads[i])):
                lowest = float(rev[i])
                break
            rev[i] = np.inf
            self._cyclic[i] = True
        if not lowest < floor:
            return None
        # Monotone rounding puts every lowest-key tie within one grid step of the raw minimum.
        inv_n = self._inv_n
        near = lowest * inv_n + 2 * 10.0**-_KEY_DP
        found: list[tuple[float, int, int, int, str]] = []
        for v in np.flatnonzero(low * inv_n <= near).tolist():
            us = np.flatnonzero(single[v] * inv_n <= near).tolist()
            if reach[v, us].any():
                self._drop_reached(v, low)
                us = np.flatnonzero(single[v] * inv_n <= near).tolist()
            found += [
                (float(single[v, u]), u, v, 0, "remove" if parent[v, u] else "add") for u in us
            ]
        for i in np.flatnonzero(rev * inv_n <= near).tolist():
            u, v = int(tails[i]), int(heads[i])
            if self._reverse_ok(u, v):
                found.append((float(rev[i]), u, v, 1, "reverse"))
        ranks = self._ranks
        best: ScannedMove | None = None
        best_key: tuple[float, int, int, int] | None = None
        for value, u, v, rank, op in found:
            if value < floor:
                key = (round(value * inv_n, _KEY_DP), ranks[u], ranks[v], rank)
                if best_key is None or key < best_key:
                    best, best_key = (key[0], u, v, rank, (op, u, v)), key
        return best


def _bits(row: int) -> list[int]:
    """The positions of the set bits of ``row``, lowest first."""
    out = []
    while row:
        lsb = row & -row
        out.append(lsb.bit_length() - 1)
        row ^= lsb
    return out


def start_dag(
    d: int, constraints: Constraints | None, max_parents: int | None = None
) -> tuple[list[int], dict[int, list[int]]]:
    """The bitset adjacency and sorted parent sets of the start DAG: the required edges, if any.

    Raises :class:`ValueError` when the required edges give a node more than ``max_parents``
    parents.
    """
    adj_int: list[int] = [0] * d
    parents_of: dict[int, list[int]] = {i: [] for i in range(d)}
    if constraints is not None:
        for u, v in zip(*np.nonzero(constraints.required)):
            adj_int[u] |= 1 << int(v)
            parents_of[int(v)].append(int(u))  # row-major: each parent list stays sorted
    if max_parents is not None and any(len(pa) > max_parents for pa in parents_of.values()):
        raise ValueError(
            f"the required edges give a node more than max_parents={max_parents} parents"
        )
    return adj_int, parents_of


def _apply_move(
    adj_int: list[int], parents_of: dict[int, list[int]], move: tuple[str, int, int]
) -> None:
    """Apply ``move`` in place to the bitset adjacency and the sorted parent-set mirror.

    Reverse is atomic: ``v`` loses parent ``u`` and ``u`` gains parent ``v`` in one step.
    """
    op, u, v = move
    if op == "add":
        adj_int[u] |= 1 << v
        parents_of[v] = sorted(parents_of[v] + [u])
    elif op == "remove":
        adj_int[u] &= ~(1 << v)
        parents_of[v] = [p for p in parents_of[v] if p != u]
    else:  # reverse: u -> v becomes v -> u
        adj_int[u] &= ~(1 << v)
        adj_int[v] |= 1 << u
        parents_of[v] = [p for p in parents_of[v] if p != u]
        parents_of[u] = sorted(parents_of[u] + [v])


def _reaches(adj_int: list[int], a: int, b: int) -> bool:
    """Whether a directed path leads from ``a`` to ``b`` in the bitset adjacency."""
    seen = 1 << a
    frontier = [a]
    while frontier:
        row = adj_int[frontier.pop()] & ~seen
        if row >> b & 1:
            return True
        seen |= row
        frontier += _bits(row)
    return False


#: Uniform tries at an edge an operation can apply to, before :func:`perturb` lists all of them.
_TRIES = 32


def perturb(
    adj_int: list[int],
    parents_of: dict[int, list[int]],
    changes: int,
    rng: random.Random,
    *,
    ranks: np.ndarray,
    forbidden: np.ndarray | None = None,
    required: np.ndarray | None = None,
    max_parents: int | None = None,
) -> None:
    """Make ``changes`` random edge changes to the DAG in place, in at most ``3 * changes`` draws.

    Each draw takes an operation uniformly from add, remove, and reverse, then one of the edges it
    can apply to uniformly, as bnlearn's restarts do. An add joins two non-adjacent nodes. No add or
    reverse makes an edge ``u -> v`` that ``forbidden[v, u]`` bars or gives a node more than
    ``max_parents`` parents, and no remove or reverse touches an edge that ``required[v, u]`` holds;
    a draw whose add or reverse would close a cycle changes nothing. Nodes are indexed by their
    place in the order of ``ranks`` (:func:`tie_ranks`), so the draws move with the columns. A draw
    tries up to ``_TRIES`` uniform candidates and lists the legal ones only when all of them fail,
    which keeps the pick uniform and its cost independent of the number of nodes while most
    candidates are legal.
    """
    d = len(adj_int)
    by_rank = np.argsort(ranks, kind="stable")  # place -> column
    place = np.empty(d, dtype=np.intp)
    place[by_rank] = np.arange(d)  # column -> place
    in_rank_order = np.ix_(by_rank, by_rank)
    # [i, j] by place: no i -> j; i -> j stays.
    barred = np.zeros((d, d), dtype=bool) if forbidden is None else forbidden.T[in_rank_order]
    held = np.zeros((d, d), dtype=bool) if required is None else required.T[in_rank_order]
    cap = d if max_parents is None else max_parents  # a node with fewer parents can gain one
    edge = np.zeros((d, d), dtype=bool)  # [i, j]: i -> j
    n_parents = np.zeros(d, dtype=np.intp)
    edges = sorted((int(place[u]), int(place[v])) for v, pa in parents_of.items() for u in pa)
    for i, j in edges:
        edge[i, j] = True
        n_parents[j] += 1
    slot = {e: k for k, e in enumerate(edges)}

    def can_add(i: int, j: int) -> bool:
        return not (i == j or edge[i, j] or edge[j, i] or barred[i, j]) and n_parents[j] < cap

    def can_remove(i: int, j: int) -> bool:
        return not held[i, j]

    def can_reverse(i: int, j: int) -> bool:
        return not (held[i, j] or barred[j, i]) and n_parents[i] < cap

    def draw(op: str) -> tuple[int, int] | None:
        """A uniformly drawn pair ``(i, j)`` that ``op`` applies to, or ``None`` if none."""
        if op == "add":
            for _ in range(_TRIES):
                i, j = rng.randrange(d), rng.randrange(d)
                if can_add(i, j):
                    return i, j
            legal = ~(edge | edge.T | barred)
            np.fill_diagonal(legal, False)
            legal[:, n_parents >= cap] = False
            cells = np.flatnonzero(legal)
            return divmod(int(cells[rng.randrange(cells.size)]), d) if cells.size else None
        allowed = can_remove if op == "remove" else can_reverse
        if not edges:
            return None
        for _ in range(_TRIES):
            pair = edges[rng.randrange(len(edges))]
            if allowed(*pair):
                return pair
        legal_edges = [pair for pair in edges if allowed(*pair)]
        return legal_edges[rng.randrange(len(legal_edges))] if legal_edges else None

    made = 0
    for _ in range(3 * changes):
        if made == changes:
            break
        op = rng.choice(("add", "remove", "reverse"))
        pair = draw(op)
        if pair is None:
            continue
        i, j = pair
        u, v = int(by_rank[i]), int(by_rank[j])
        if op == "add" and _reaches(adj_int, v, u):
            continue
        if op == "reverse":
            rest = list(adj_int)
            rest[u] &= ~(1 << v)
            if _reaches(rest, u, v):
                continue
        _apply_move(adj_int, parents_of, (op, u, v))
        made += 1
        if op == "add":
            edge[i, j] = True
            n_parents[j] += 1
            slot[i, j] = len(edges)
            edges.append((i, j))
            continue
        edge[i, j] = False
        n_parents[j] -= 1
        k = slot.pop((i, j))
        if op == "reverse":
            edge[j, i] = True
            n_parents[i] += 1
            edges[k] = (j, i)
            slot[j, i] = k
        else:  # the last edge takes the removed one's slot
            last = edges.pop()
            if k < len(edges):
                edges[k] = last
                slot[last] = k


def _climb(
    scanner: MoveScanner,
    adj_int: list[int],
    parents_of: dict[int, list[int]],
    max_iter: int | None,
) -> int:
    """Take the best improving move until none is left or ``max_iter`` are taken; count them."""
    moves = 0
    while max_iter is None or moves < max_iter:
        best = scanner.best(adj_int, parents_of)
        if best is None:
            break
        _apply_move(adj_int, parents_of, best[4])
        moves += 1
    return moves


def hc(
    data: np.ndarray,
    *,
    score_func: str = "bic",
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    max_iter: int | None = None,
    categorical: np.ndarray | None = None,
    constraints: Constraints | None = None,
    max_parents: int | None = None,
    n_starts: int = 1,
    n_perturbations: int = 1,
    random_state: int = 0,
) -> tuple[GraphStructure, float]:
    """Discover a CPDAG by hill climbing over a BIC, categorical, or mixed-data score.

    Runs greedy add/remove/reverse edge moves from the empty graph (or the required edges of
    ``constraints``) until no move improves the total score or, when set, ``max_iter`` moves are
    taken, then canonicalizes the discovered DAG to its CPDAG. Moves with equal rounded score
    changes are chosen by :func:`tie_ranks`, whose ranks follow the columns when reordered, subject
    to its hash and rounding limits. These choices can affect later moves and the final CPDAG; the
    ranks do not indicate causal direction. Under a mixed-data score the DAG stays a conditional
    linear Gaussian network: no move gives a categorical node a continuous parent. Returns the
    CPDAG and the DAG's total objective (lower is better), the sum of the final per-node local
    scores.

    Each start after the first climbs again from the best DAG so far after :func:`perturb` makes
    ``n_perturbations`` random changes to it, which keep the moves' rules, and replaces it only
    when its DAG scores lower by more than :data:`START_EPS`.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Data matrix; rows are observations, columns are variables.
    score_func : str
        Local score: ``"bic"`` (linear-Gaussian BIC),
        ``"bdeu"`` (BDeu on category codes,
        :class:`~andrey.core.score.BDeuScore` with its default priors),
        ``"discrete_bic"`` (the multinomial BIC on category codes,
        :class:`~andrey.core.score.CategoricalBICScore`), or a mixed-data BIC,
        ``"degen_gauss"`` (:class:`~andrey.core.score.DGScore`) or
        ``"cond_gauss"`` (:class:`~andrey.core.score.CGScore`).
    penalty_discount : float
        Weight on the BIC complexity term of the deviance the search minimizes; BDeu ignores it.
    ebic_gamma : float
        Weight on the extended-BIC term of the linear-Gaussian BIC,
        ``2 * ebic_gamma * log C(n_features - 1, n_parents)`` per node
        (:func:`~andrey.core.score.ebic_penalty`); ``0.0`` is the standard BIC. The other scores
        ignore it.
    max_iter : int or None
        Maximum moves of each start; ``None`` sets no limit. A start that takes ``max_iter`` moves
        warns with ``SearchLimitWarning``, once per call.
    categorical : ndarray of bool or None
        Marks the categorical columns for a mixed-data score; the other scores ignore it.
    constraints : Constraints or None
        Background knowledge: the search starts from the required edges, never removes or reverses
        one, and never adds a forbidden edge. The CPDAG is then narrowed by the knowledge
        (:func:`~andrey.core.knowledge.mpdag`). Under a mixed-data score, a required edge from a
        continuous into a categorical column raises ``ValueError``.
    max_parents : int or None
        Most parents of any node; no move gives a node more. ``None`` sets no cap.
    n_starts : int
        Number of climbs, at least ``1``.
    n_perturbations : int
        Random changes to the best DAG so far before each start after the first, at least ``1``.
    random_state : int
        Seed of the changes; start ``k`` draws from :func:`~andrey.core.seeding.start_rng`.

    Returns
    -------
    tuple[GraphStructure, float]
        The CPDAG (``kind="cpdag"``) and the total score of the discovered DAG.
    """
    X = np.asarray(data, dtype=np.float64)
    d = X.shape[1]

    if score_func not in HC_SCORES:
        raise NotImplementedError(f"unsupported score_func {score_func!r}; use one of {HC_SCORES}")

    def new_score() -> LocalScore:
        return make_score(
            score_func,
            X,
            penalty_discount=penalty_discount,
            ebic_gamma=ebic_gamma,
            categorical=categorical,
        )

    score = new_score()
    forbidden = None
    mask = np.full(d, score_func in DISCRETE_SCORES)
    if score_func in MIXED_SCORES:  # [v, u]: categorical v, continuous u
        mask = column_mask(categorical, d)
        forbidden = mask[:, None] & ~mask[None, :]
    if constraints is not None:
        if forbidden is not None and (constraints.required & forbidden.T).any():
            raise ValueError(
                "knowledge requires an edge from a continuous into a categorical column, which a "
                "mixed-data score does not allow"
            )
        knowledge_forbidden = constraints.forbidden.T  # [v, u]: no u -> v
        forbidden = knowledge_forbidden if forbidden is None else forbidden | knowledge_forbidden
    required = None if constraints is None else constraints.required.T

    start_adj, start_parents = start_dag(d, constraints, max_parents)
    ranks = tie_ranks(X, mask)
    best: tuple[list[int], dict[int, list[int]], float] | None = None
    capped = False
    for start in range(n_starts):
        if start == 0:
            adj_int, parents_of = start_adj, start_parents
        else:
            assert best is not None  # set by start 0
            # Each start builds its own score, deltas, and tables; the last start's are dropped.
            score = new_score()
            adj_int, parents_of = list(best[0]), {v: list(pa) for v, pa in best[1].items()}
            perturb(
                adj_int,
                parents_of,
                n_perturbations,
                start_rng(random_state, start),
                ranks=ranks,
                forbidden=forbidden,
                required=required,
                max_parents=max_parents,
            )
        delta = DeltaBICScore(score) if isinstance(score, BICScore) else FullScoreDelta(score)
        # One BLAS thread keeps the Schur updates reproducible across thread counts.
        with backend.one_blas_thread():
            scanner = MoveScanner(
                delta,
                d,
                ranks=ranks,
                forbidden=forbidden,
                required=required,
                max_parents=max_parents,
            )
            moves = _climb(scanner, adj_int, parents_of, max_iter)
        capped |= max_iter is not None and moves == max_iter
        total = float(sum(score.score(i, parents_of[i]) for i in range(d)))
        if best is None or total < best[2] - START_EPS:
            best = (adj_int, parents_of, total)
        del scanner, delta
    if capped and max_iter is not None:
        warn_search_limit("hc", "max_iter", max_iter, "moves")
    assert best is not None  # n_starts >= 1
    _, parents_of, total = best

    # Emit the discovered DAG as an unsigned endpoint-mark matrix (u -> v: TAIL at u, ARROW at v),
    # then reduce it to the CPDAG of its equivalence class.
    dag = np.full((d, d), NULL, dtype=np.int8)
    for v in range(d):
        for p in parents_of[v]:
            dag[p, v] = TAIL
            dag[v, p] = ARROW
    cpdag = dag2cpdag(dag)
    if constraints is not None:
        cpdag = mpdag(cpdag, constraints)
    return to_structure(cpdag, kind="cpdag"), total
