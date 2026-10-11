"""Hill climbing over DAGs, the engine behind :func:`andrey.hc`.

The algorithm, its settings, and its references are described on :func:`andrey.hc`.

Candidate moves are scored by the Schur-complement deltas of :mod:`andrey.core.score_delta`;
:class:`MoveScanner` keeps them between passes and recomputes only the targets a move changed.
The move budget also regularizes: greedy ascent adds the strongest edges first, so stopping early
keeps noise-fitting edges out of large, dense graphs. The DAG is converted to its CPDAG for output;
the reported objective is the DAG's total BIC, lower is better.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from itertools import compress
from operator import ne

import numpy as np

from andrey.core import ARROW, NULL, TAIL, GraphStructure, backend
from andrey.core.orient import dag2cpdag, to_structure
from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore, add_deltas, remove_deltas
from andrey.core.warning_policy import SearchLimitWarning, warn_once
from andrey.search._parallel import _EPS_IMPROVE, _KEY_DP

# ``_EPS_IMPROVE`` (the strict-improvement floor) and ``_KEY_DP`` (the tie-break quantization
# decimals) are shared with GES via ``search._parallel`` so the two learners break near-ties by the
# same rule. A move is a candidate only when its raw delta clears ``-_EPS_IMPROVE`` (above float64
# Schur-delta noise, below the smallest real improvement); the comparison key quantizes the
# *normalized* delta ``delta / n_samples`` to ``_KEY_DP`` places. Two mathematically equal moves
# (add x->y vs y->x, from different cov submatrices) differ in their last bits, so the key, not the
# raw delta, decides between them, and the tie goes to the lower ``(u, v, op_rank)``. The noise
# scales as ~n_samples * cond(cov) * eps, as does the signal (a real BIC improvement); keying on
# delta/n makes the grid track the signal, so after normalization the residual is ~cond*eps, which
# _KEY_DP=9 clears for cond up to ~1e5 (all non-degenerate data), independent of n. It is a
# mitigation, not a proof -- a normalized delta within ULP of a grid line could still round either
# way. The objective is recomputed from the final structure (not accumulated), so this never touches
# the reported score, and delta/n is monotone in delta, so non-tied move order is unchanged.
__all__ = ["hc"]


def _descendants(adj_int: list[int], d: int) -> list[int]:
    """Descendant bitsets of each node, by one pass over the DAG in reverse topological order.

    ``adj_int[u]`` is the out-neighbor bitset of ``u`` (bit ``v`` set iff the edge ``u -> v``
    exists).
    """
    indegree = [0] * d
    for row in adj_int:
        while row:
            lsb = row & -row
            indegree[lsb.bit_length() - 1] += 1
            row ^= lsb
    order: list[int] = []
    stack = [u for u in range(d) if not indegree[u]]
    while stack:
        u = stack.pop()
        order.append(u)
        row = adj_int[u]
        while row:
            lsb = row & -row
            c = lsb.bit_length() - 1
            indegree[c] -= 1
            if not indegree[c]:
                stack.append(c)
            row ^= lsb
    reach = [0] * d
    for u in reversed(order):
        below = row = adj_int[u]
        while row:
            lsb = row & -row
            below |= reach[lsb.bit_length() - 1]
            row ^= lsb
        reach[u] = below
    return reach


# A scanned candidate: (delta, u, v, op_rank, (op, u, v)). op_rank orders the ops that share a
# ``(u, v)`` (remove 0 < reverse 1; add is 0 and never coexists with remove), so ``key = r[:4]``
# is a strict total order over the moves of a pass.
ScannedMove = tuple[float, int, int, int, tuple[str, int, int]]

# A DAG's identity for a tabu list: its out-neighbor bitset rows, ``tuple(adj_int)``.
StructureKey = tuple[int, ...]


def _key_after(adj_int: list[int], move: tuple[str, int, int]) -> StructureKey:
    """The structure key of the DAG that ``move`` leads to; ``adj_int`` itself is unchanged."""
    op, u, v = move
    rows = list(adj_int)
    if op == "add":
        rows[u] |= 1 << v
    else:  # remove, or reverse: u -> v goes, and reverse adds v -> u
        rows[u] &= ~(1 << v)
        if op == "reverse":
            rows[v] |= 1 << u
    return tuple(rows)


class MoveScanner:
    """The best single-edge move of a pass, from delta tables kept between passes.

    The delta of adding or removing ``u`` at target ``v`` depends only on ``v``'s parent set, so the
    tables ``add[v, u]`` and ``rem[v, u]`` hold row ``v`` for ``v``'s current parents. A pass
    recomputes the rows of the targets whose parents changed, each in one vectorized Schur step
    (:func:`~andrey.core.score_delta.add_deltas`, :func:`~andrey.core.score_delta.remove_deltas`).
    A delta depends on its target and parent set alone, so every scanner computes the same value.

    ``single[v, u]`` is the add or remove delta of ``u -> v``, ``inf`` when ``v -> u`` exists
    (moving it is a reverse) or when ``v`` reaches ``u`` (adding it closes a cycle), and ``low[v]``
    is the minimum of row ``v``. Reverses are scored over the edge list. Reversing a covered edge
    ``u -> v`` (``pa(v) = pa(u) + u``) leads to a Markov equivalent DAG of equal score, so its delta
    is exactly zero rather than the rounding noise of two Schur deltas, which can exceed
    ``_EPS_IMPROVE`` and turn a no-op into an improvement.

    ``reach[x, y]`` holds whether ``x`` reaches ``y``; it is extended on each added edge and
    recomputed when an edge goes, and then every row whose target reaches fewer nodes is rebuilt. An
    added edge can only make more adds close a cycle, so a row stays valid until its lowest add is
    found to close one; then the row drops every node its target now reaches. A pass therefore
    checks for cycles only its lowest candidates.

    A move ``u -> v`` belongs to the scan whose ``targets`` hold its head ``v``, so partitioning the
    targets partitions the moves (:mod:`andrey.search._parallel_hc`). Reversing ``u -> v`` sums an
    add of ``v`` at target ``u`` and a remove of ``u`` at target ``v``.
    """

    def __init__(self, delta: DeltaBICScore, d: int) -> None:
        self._score = delta.score_obj
        self._cond = delta.cond_threshold
        self._d = d
        self._inv_n = 1.0 / self._score.n  # key on delta/n: noise and signal scale with n
        self._add = np.full((d, d), np.inf)
        self._rem = np.full((d, d), np.inf)
        self._parent = np.zeros((d, d), dtype=bool)  # [v, u]: the edge u -> v exists
        self._single = np.full((d, d), np.inf)
        self._low = np.full(d, np.inf)
        self._tails = np.zeros(0, dtype=np.intp)  # the edge list, tails[i] -> heads[i]
        self._heads = np.zeros(0, dtype=np.intp)
        self._cyclic = np.zeros(0, dtype=bool)  # reversing edge i was found to close a cycle
        self._reach = np.zeros((d, d), dtype=bool)
        self._adj: list[int] | None = None  # the DAG the tables describe
        # Ids of parent sets: u -> v is covered iff set_of[v] == set_with[u], the id of pa(u) + u.
        self._ids: dict[tuple[int, ...], int] = {}
        self._set_of = np.zeros(d, dtype=np.intp)
        self._set_with = np.zeros(d, dtype=np.intp)

    def best(
        self,
        adj_int: list[int],
        parents_of: dict[int, list[int]],
        *,
        targets: Sequence[int] | None = None,
        floor: float = -_EPS_IMPROVE,
        tabu: Collection[StructureKey] | None = None,
    ) -> ScannedMove | None:
        """The best legal move with delta below ``floor`` whose head is in ``targets``.

        ``targets`` defaults to every node. Moves are keyed ``(delta / n_samples rounded to
        _KEY_DP places, u, v, op_rank)``, a strict total order, so the best of a partition's
        bests is the best overall. The default ``floor`` admits improving moves only; ``floor=inf``
        admits every legal move. A move that leads to a structure in ``tabu`` is skipped.
        """
        if self._d < 2:
            return None
        self._sync(adj_int, parents_of)
        low = self._low.copy()
        edges = np.arange(self._tails.size)
        if targets is not None:
            head = np.zeros(self._d, dtype=bool)
            head[list(targets)] = True
            low[~head] = np.inf
            edges = edges[head[self._heads]]
        tails, heads = self._tails[edges], self._heads[edges]
        with np.errstate(invalid="ignore"):  # a degenerate row's deltas may be infinite
            rev = self._add[tails, heads] + self._rem[heads, tails]
        rev[self._set_of[heads] == self._set_with[tails]] = 0.0
        rev[np.isnan(rev) | self._cyclic[edges]] = np.inf
        held: list[tuple[int, int, float]] = []  # tabu moves set aside for this call
        try:
            while True:  # the best move, unless it leads into the tabu list; then the next best
                best = self._pick(low, rev, edges, floor)
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
        for u, v in gained:
            self._parent[v, u] = True
        for u, v in lost:
            self._parent[v, u] = False
        for v in range(d) if first else {v for _, v in gained + lost}:
            pa = parents_of[v]
            self._add[v] = add_deltas(self._score, v, pa, cond_threshold=self._cond)
            self._rem[v] = remove_deltas(self._score, v, pa, cond_threshold=self._cond)
            ids = self._ids
            self._set_of[v] = ids.setdefault(tuple(pa), len(ids))
            self._set_with[v] = ids.setdefault(tuple(sorted([*pa, v])), len(ids))
        reach = self._reach
        rows = set(range(d)) if first else {x for pair in gained + lost for x in pair}
        if first or lost:
            new = _bit_matrix(_descendants(adj_int, d), d)
            rows.update(np.flatnonzero((new != reach).any(axis=1)).tolist())
            self._reach = reach = new
            keep = np.ones(self._tails.size, dtype=bool)
            for a, b in lost:
                keep &= (self._tails != a) | (self._heads != b)
            self._tails, self._heads = self._tails[keep], self._heads[keep]
            self._cyclic = np.zeros(self._tails.size, dtype=bool)
        else:
            for a, b in gained:
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
            self._single[v] = row
            self._low[v] = row.min()
        self._adj = list(adj_int)

    def _drop_reached(self, v: int, low: np.ndarray) -> None:
        """Drop from row ``v`` every add that closes a cycle now, and update its minimum."""
        self._single[v, self._reach[v]] = np.inf
        self._low[v] = low[v] = self._single[v].min()

    def _reverse_ok(self, u: int, v: int) -> bool:
        """Whether reversing ``u -> v`` keeps the graph acyclic: ``u`` reaches ``v`` only by it."""
        return not self._reach[self._parent[:, u], v].any()  # no child of u reaches v

    def _pick(
        self, low: np.ndarray, rev: np.ndarray, edges: np.ndarray, floor: float
    ) -> ScannedMove | None:
        """The legal move with the lowest key among those below ``floor``.

        ``rev`` scores reversing the edges ``edges``. A row whose candidate closes a cycle drops
        every such add, and a reverse that closes one is marked, until an edge goes.
        """
        single, parent, reach = self._single, self._parent, self._reach
        tails, heads = self._tails[edges], self._heads[edges]
        lowest = np.inf
        while True:  # the lowest legal single move below floor
            v = int(np.argmin(low))
            if not low[v] < floor:
                break
            u = int(np.argmin(single[v]))
            if not reach[v, u]:  # a remove, or an add that closes no cycle
                lowest = float(low[v])
                break
            self._drop_reached(v, low)
        while rev.size:  # a legal reverse below both
            i = int(np.argmin(rev))
            if not rev[i] < min(lowest, floor):
                break
            if self._reverse_ok(int(tails[i]), int(heads[i])):
                lowest = float(rev[i])
                break
            rev[i] = np.inf
            self._cyclic[edges[i]] = True
        if not lowest < floor:
            return None
        # Rounding is monotone, so the moves that round to the lowest key lie within one grid step
        # of the lowest unrounded value; the exact key is computed for those few.
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


def _bit_matrix(rows: list[int], d: int) -> np.ndarray:
    """Boolean matrix of bitset rows: ``out[i, j]`` is bit ``j`` of ``rows[i]``."""
    width = (d + 7) // 8
    raw = np.frombuffer(b"".join(r.to_bytes(width, "little") for r in rows), dtype=np.uint8)
    return np.unpackbits(raw.reshape(len(rows), width), axis=1, count=d, bitorder="little").view(
        bool
    )


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
    max_iter: int = 200,
) -> tuple[GraphStructure, float]:
    """Discover a CPDAG by Hill-Climbing over the linear-Gaussian BIC score.

    Runs greedy add/remove/reverse edge moves from the empty graph until no move improves the total
    BIC or ``max_iter`` moves are taken, then canonicalizes the discovered DAG to its CPDAG. Returns
    the CPDAG and the DAG's total BIC objective (lower is better), the sum of the final per-node
    local scores.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Data matrix; rows are observations, columns are variables.
    score_func : str
        Local score; ``"local_score_BIC_from_cov"`` (linear-Gaussian BIC) is supported.
    lambda_value : float
        Weight on the BIC complexity term of the deviance the search minimizes.
    max_iter : int
        Maximum number of accepted edge moves (default 200). The cap bounds
        the ascent so a large, dense problem does not overfit the recovered graph with spurious
        edges; small or sparse problems reach the local optimum first and never hit it. Taking
        ``max_iter`` moves warns with ``SearchLimitWarning``.

    Returns
    -------
    tuple[GraphStructure, float]
        The CPDAG (``kind="cpdag"``) and the total BIC score of the discovered DAG.
    """
    if score_func != "local_score_BIC_from_cov":
        raise NotImplementedError(
            f"unsupported score_func {score_func!r}; use 'local_score_BIC_from_cov'"
        )
    X = np.asarray(data, dtype=np.float64)
    d = X.shape[1]

    score = BICScore(X, lambda_value=lambda_value)
    delta = DeltaBICScore(score)

    adj_int: list[int] = [0] * d
    parents_of: dict[int, list[int]] = {i: [] for i in range(d)}

    # Pin BLAS to one thread for the delta loops so the Schur updates are thread-count-reproducible
    # and equal to the (1-thread) parallel workers; above the work threshold the pass fans out.
    with backend.one_blas_thread():
        workers = backend.worker_count()
        if workers > 1 and d > 1:
            from andrey.search import _parallel_hc  # local import breaks the import cycle

            moves = _parallel_hc.run_hc(adj_int, parents_of, delta, d, max_iter, workers=workers)
        else:
            scanner = MoveScanner(delta, d)
            moves = 0
            for _ in range(max_iter):
                best = scanner.best(adj_int, parents_of)
                if best is None:
                    break
                _apply_move(adj_int, parents_of, best[4])
                moves += 1
    if moves == max_iter:
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
