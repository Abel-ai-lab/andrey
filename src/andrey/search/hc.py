"""Hill climbing over DAGs, the engine behind :func:`andrey.hc`.

Under BIC, candidate moves use Schur-complement deltas from :mod:`andrey.core.score_delta`; under
BDeu and the mixed-data scores, the difference of two full local scores. :class:`MoveScanner`
caches them between passes, recomputing only changed targets. Output is the DAG's CPDAG and total
score; lower is better.
"""

from __future__ import annotations

from collections.abc import Collection
from itertools import compress
from operator import ne

import numpy as np

from andrey.core import ARROW, NULL, TAIL, GraphStructure, backend
from andrey.core.mixed import column_mask
from andrey.core.orient import dag2cpdag, to_structure
from andrey.core.score import MIXED_SCORES, BICScore, make_score
from andrey.core.score_delta import DeltaBICScore, FullScoreDelta, add_deltas, remove_deltas
from andrey.core.warning_policy import SearchLimitWarning, warn_once
from andrey.search._parallel import _EPS_IMPROVE, _KEY_DP

# Moves are keyed on ``delta / n_samples`` rounded to ``_KEY_DP`` places, not on the raw delta.
# Two equal moves (add x->y vs y->x, from different cov submatrices) differ in their last bits;
# the key makes them tie, and the lower ``(u, v, op_rank)`` wins. Schur noise scales as
# ~n_samples * cond(cov) * eps, like a real improvement, so the normalized residual is
# ~cond * eps: ``_KEY_DP`` = 9 clears it for cond up to ~1e5, independent of n. A mitigation, not
# a proof: a normalized delta within ULP of a grid line can round either way. Rounding is
# monotone, so non-tied order holds; the reported score is recomputed from the final structure.
__all__ = ["hc"]

#: The local scores :func:`hc` accepts.
HC_SCORES: tuple[str, ...] = (
    "local_score_BIC_from_cov",
    "local_score_BDeu",
    "local_score_degen_gauss",
    "local_score_cond_gauss",
)


# (delta, u, v, op_rank, (op, u, v)). op_rank: remove 0 < reverse 1; add is 0 and never shares
# ``(u, v)`` with a remove.
ScannedMove = tuple[float, int, int, int, tuple[str, int, int]]

# Out-neighbor bitset rows, ``tuple(adj_int)``, used as tabu keys.
StructureKey = tuple[int, ...]


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
    (``pa(v) = pa(u) + u``) reverses to a Markov equivalent DAG with equal score under BIC, BDeu,
    and the mixed-data scores, all score-equivalent. Its delta is set to zero to prevent rounding
    noise exceeding ``_EPS_IMPROVE`` from creating a false improvement.

    ``forbidden[v, u]`` marks edges ``u -> v`` no add or reverse creates; ``None`` forbids none.

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
        forbidden: np.ndarray | None = None,
    ) -> None:
        self._delta = delta
        self._score = delta.score_obj
        self._d = d
        self._inv_n = 1.0 / self._score.n
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

    def best(
        self,
        adj_int: list[int],
        parents_of: dict[int, list[int]],
        *,
        floor: float = -_EPS_IMPROVE,
        tabu: Collection[StructureKey] | None = None,
    ) -> ScannedMove | None:
        """The best legal move with delta below ``floor``.

        Moves are keyed ``(delta / n_samples rounded to _KEY_DP places, u, v, op_rank)``, a strict
        total order. The default ``floor`` admits improving moves only; ``floor=inf`` admits every
        legal move. A move that leads to a structure in ``tabu`` is skipped.
        """
        if self._d < 2:
            return None
        self._sync(adj_int, parents_of)
        low = self._low.copy()
        tails, heads = self._tails, self._heads
        with np.errstate(invalid="ignore"):  # a degenerate row's deltas may be infinite
            rev = self._add[tails, heads] + self._rem[heads, tails]
        rev[self._set_of[heads] == self._set_with[tails]] = 0.0
        rev[np.isnan(rev) | self._cyclic | self._forbidden[tails, heads]] = np.inf
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
            row[self._parent[:, v] | reach[v] | self._forbidden[v] | np.isnan(row)] = np.inf
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
        best: ScannedMove | None = None
        for value, u, v, rank, op in found:
            if value < floor:
                cand = (round(value * inv_n, _KEY_DP), u, v, rank, (op, u, v))
                if best is None or cand[:4] < best[:4]:
                    best = cand
        return best


def _bits(row: int) -> list[int]:
    """The positions of the set bits of ``row``, lowest first."""
    out = []
    while row:
        lsb = row & -row
        out.append(lsb.bit_length() - 1)
        row ^= lsb
    return out


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


def hc(
    data: np.ndarray,
    *,
    score_func: str = "local_score_BIC_from_cov",
    lambda_value: float = 1.0,
    max_iter: int | None = None,
    categorical: np.ndarray | None = None,
) -> tuple[GraphStructure, float]:
    """Discover a CPDAG by Hill-Climbing over a BIC, BDeu, or mixed-data score.

    Runs greedy add/remove/reverse edge moves from the empty graph until no move improves the total
    score or, when set, ``max_iter`` moves are taken, then canonicalizes the discovered DAG to its
    CPDAG. Under a mixed-data score the DAG stays a conditional linear Gaussian network: no move
    gives a categorical node a continuous parent. Returns the CPDAG and the DAG's total objective
    (lower is better), the sum of the final per-node local scores.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Data matrix; rows are observations, columns are variables.
    score_func : str
        Local score: ``"local_score_BIC_from_cov"`` (linear-Gaussian BIC), ``"local_score_BDeu"``
        (BDeu on category codes, :class:`~andrey.core.score.BDeuScore` with its default priors), or
        a mixed-data BIC, ``"local_score_degen_gauss"`` (:class:`~andrey.core.score.DGScore`) or
        ``"local_score_cond_gauss"`` (:class:`~andrey.core.score.CGScore`).
    lambda_value : float
        Weight on the BIC complexity term of the deviance the search minimizes; BDeu ignores it.
    max_iter : int or None
        Maximum moves; ``None`` sets no limit. Taking ``max_iter`` moves warns with
        ``SearchLimitWarning``.
    categorical : ndarray of bool or None
        Marks the categorical columns for a mixed-data score; the other scores ignore it.

    Returns
    -------
    tuple[GraphStructure, float]
        The CPDAG (``kind="cpdag"``) and the total score of the discovered DAG.
    """
    X = np.asarray(data, dtype=np.float64)
    d = X.shape[1]

    if score_func not in HC_SCORES:
        raise NotImplementedError(f"unsupported score_func {score_func!r}; use one of {HC_SCORES}")
    score = make_score(score_func, X, lambda_value=lambda_value, categorical=categorical)
    delta = DeltaBICScore(score) if isinstance(score, BICScore) else FullScoreDelta(score)
    forbidden = None
    if score_func in MIXED_SCORES:  # [v, u]: categorical v, continuous u
        mask = column_mask(categorical, d)
        forbidden = mask[:, None] & ~mask[None, :]

    adj_int: list[int] = [0] * d
    parents_of: dict[int, list[int]] = {i: [] for i in range(d)}

    # One BLAS thread keeps the Schur updates reproducible across thread counts.
    with backend.one_blas_thread():
        scanner = MoveScanner(delta, d, forbidden=forbidden)
        moves = 0
        while max_iter is None or moves < max_iter:
            best = scanner.best(adj_int, parents_of)
            if best is None:
                break
            _apply_move(adj_int, parents_of, best[4])
            moves += 1
    if max_iter is not None and moves == max_iter:
        warn_once(
            f"hc stopped at its limit of max_iter={max_iter} moves; a longer search may find a "
            "better graph, so raise max_iter",
            SearchLimitWarning,
        )

    # Emit the discovered DAG as an unsigned endpoint-mark matrix (u -> v: TAIL at u, ARROW at v),
    # then reduce it to the CPDAG of its equivalence class.
    dag = np.full((d, d), NULL, dtype=np.int8)
    for v in range(d):
        for p in parents_of[v]:
            dag[p, v] = TAIL
            dag[v, p] = ARROW
    cpdag = dag2cpdag(dag)

    total = float(sum(score.score(i, parents_of[i]) for i in range(d)))
    return to_structure(cpdag, kind="cpdag"), total
