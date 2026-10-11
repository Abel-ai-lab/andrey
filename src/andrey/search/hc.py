"""Hill climbing over DAGs, the engine behind :func:`andrey.hc`.

The algorithm, its settings, and its references are described on :func:`andrey.hc`.

Each candidate move is scored by the Schur-complement delta of :mod:`andrey.core.score_delta`;
:class:`MoveScanner` keeps the deltas between passes and computes only those a move made new.
The move budget also regularizes: greedy ascent adds the strongest edges first, so stopping early
keeps noise-fitting edges out of large, dense graphs. The DAG is converted to its CPDAG for output;
the reported objective is the DAG's total BIC, lower is better.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np

from andrey.core import ARROW, NULL, TAIL, GraphStructure, backend
from andrey.core.orient import dag2cpdag, to_structure
from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore
from andrey.core.warning_policy import SearchLimitWarning, warn_once
from andrey.search._parallel import _EPS_IMPROVE, _KEY_DP

# ``_EPS_IMPROVE`` (the strict-improvement floor) and ``_KEY_DP`` (the tie-break quantization
# decimals) are shared with GES via ``search._parallel`` so the two learners break near-ties by the
# same rule. A move is a candidate only when its raw delta clears ``-_EPS_IMPROVE`` (above float64
# Schur-delta noise, below the smallest real improvement); the comparison key quantizes the
# *normalized* delta ``delta / n_samples`` to ``_KEY_DP`` places. The incremental Schur delta is NOT
# bit-pure -- two mathematically-equal moves (add x->y vs y->x, from different cov submatrices)
# differ, and one scorer's chained residual cache rounds a query differently from another's -- so
# serial (one scorer) and the parallel workers (one each) would break such near-ties inconsistently
# on a raw key. The cross-scorer noise is NOT flat: delta = n_samples * (log R_new - log R_old)
# (score_delta.py), so it scales as ~n_samples * cond(cov) * eps, as does the signal (a real BIC
# improvement); keying on delta/n makes the grid track the signal, so after normalization the
# residual is ~cond*eps, which _KEY_DP=9 clears for cond up to ~1e5 (all non-degenerate data),
# independent of n. It is a mitigation, not a proof -- a normalized delta within ULP of a grid line,
# or conditioning near the fast path's own ~6-digit limit, could still round two ways. The objective
# is recomputed from the final structure (not accumulated), so this never touches the reported
# score, and delta/n is monotone in delta, so non-tied move order is unchanged.
__all__ = ["hc"]


def _closure(adj_int: list[int], d: int) -> tuple[list[int], list[int]]:
    """Descendant bitsets of each node: all, and those a path of two or more edges reaches.

    ``adj_int[u]`` is the out-neighbor bitset of ``u`` (bit ``v`` set iff the edge ``u -> v``
    exists). One pass over the DAG in reverse topological order.
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
    deep = [0] * d
    for u in reversed(order):
        below = 0
        row = adj_int[u]
        while row:
            lsb = row & -row
            below |= reach[lsb.bit_length() - 1]
            row ^= lsb
        deep[u] = below
        reach[u] = adj_int[u] | below
    return reach, deep


def _bit_matrix(rows: list[int], d: int) -> np.ndarray:
    """Boolean matrix of bitset rows: ``out[i, j]`` is bit ``j`` of ``rows[i]``."""
    width = (d + 7) // 8
    raw = np.frombuffer(b"".join(r.to_bytes(width, "little") for r in rows), dtype=np.uint8)
    bits = np.unpackbits(raw.reshape(len(rows), width), axis=1, bitorder="little")
    return bits[:, :d].astype(bool)


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


_ADD, _REMOVE = 0, 1  # first index of MoveScanner's delta tables


class MoveScanner:
    """The best single-edge move of a pass, rescoring only the targets whose parents changed.

    The delta of adding or removing ``u`` at target ``v`` depends only on ``v``'s parent set, so it
    is kept in a table, ``(op, u, v)``, whose column ``v`` belongs to ``v``'s current parents.
    When those change, the column is put aside under ``(v, parents)`` and the one for the new
    parents is brought back or started empty; a column is never discarded, so a delta is computed
    at most once per run. Each pass computes only the deltas its legal moves need that the table
    does not hold, in the order the moves are visited (``u``, then ``v``; a remove before the add
    that completes its reverse). That order matters: the Schur delta reads a residual cache that
    earlier deltas write, so the order fixes every value to the last bit. Legality, admission, and
    the tie-break run on whole arrays.

    A move ``u -> v`` belongs to the scan whose ``targets`` hold its head ``v``, so partitioning the
    targets partitions the moves (:mod:`andrey.search._parallel_hc`). Reversing ``u -> v`` sums an
    add of ``v`` at target ``u`` and a remove of ``u`` at target ``v``.
    """

    def __init__(self, delta: DeltaBICScore, d: int) -> None:
        self._delta = delta
        self._d = d
        self._inv_n = 1.0 / delta.score_obj.n  # key on delta/n: noise and signal scale with n
        self._value = np.full((2, d, d), np.nan)
        self._known = np.zeros((2, d, d), dtype=bool)
        self._parents: list[tuple[int, ...] | None] = [None] * d  # what column v belongs to
        self._shelf: dict[tuple[int, tuple[int, ...]], tuple[np.ndarray, np.ndarray]] = {}
        self._last: tuple | None = None  # the last DAG scanned, and its _moves

    def _sync(self, pa: list[tuple[int, ...]]) -> None:
        """Point each column at its target's current parents."""
        for v in range(self._d):
            old = self._parents[v]
            if old == pa[v]:
                continue
            if old is not None:
                self._shelf[(v, old)] = (self._value[:, :, v].copy(), self._known[:, :, v].copy())
            got = self._shelf.pop((v, pa[v]), None)
            if got is None:
                self._value[:, :, v] = np.nan
                self._known[:, :, v] = False
            else:
                self._value[:, :, v], self._known[:, :, v] = got
            self._parents[v] = pa[v]

    def _fill(self, k: int, u: int, v: int, pa_v: tuple[int, ...]) -> None:
        self._value[k, u, v] = self._delta.delta(v, pa_v, u, "add" if k == _ADD else "remove")
        self._known[k, u, v] = True

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
        key = (tuple(adj_int), None if targets is None else tuple(targets))
        if self._last is None or self._last[0] != key:  # tabu search scans a DAG twice
            self._last = (key, *self._moves(adj_int, parents_of, targets))
        _, edge, single, rev = self._last
        single = np.where(single < floor, single, np.inf)
        rev = np.where(rev < floor, rev, np.inf)
        while True:  # the best move, unless it leads into the tabu list; then the next best
            best = self._pick(single, rev, edge)
            if best is None or tabu is None or _key_after(adj_int, best[4]) not in tabu:
                return best
            (rev if best[3] else single)[best[1], best[2]] = np.inf

    def _moves(
        self, adj_int: list[int], parents_of: dict[int, list[int]], targets: Sequence[int] | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The edge matrix, then the add-or-remove and the reverse deltas, ``inf`` where illegal."""
        d = self._d
        pa = [tuple(parents_of[i]) for i in range(d)]
        self._sync(pa)
        if targets is None:
            head = np.ones(d, dtype=bool)
        else:
            head = np.zeros(d, dtype=bool)
            head[list(targets)] = True
        reach, deep = _closure(adj_int, d)
        bits = _bit_matrix(adj_int + reach + deep, d)
        edge, reaches, deep_reach = bits[:d], bits[d : 2 * d], bits[2 * d :]
        free = ~(edge | edge.T)
        np.fill_diagonal(free, False)
        add_ok = free & ~reaches.T & head  # u -> v closes a cycle iff v reaches u
        rem_ok = edge & head
        rev_ok = rem_ok & ~deep_reach  # ... iff u reaches v other than by the edge itself

        known_add, known_rem = self._known
        first = np.flatnonzero((rem_ok & ~known_rem) | (add_ok & ~known_add))
        second = np.flatnonzero(rev_ok & ~known_add.T)
        for k in np.sort(np.concatenate((first * 2, second * 2 + 1))).tolist():
            u, v = divmod(k >> 1, d)
            if k & 1:  # the add that completes reversing u -> v: v joins u's parents
                self._fill(_ADD, v, u, pa[u])
            else:
                self._fill(_REMOVE if edge[u, v] else _ADD, u, v, pa[v])

        add, rem = self._value
        with np.errstate(invalid="ignore"):  # a degenerate column's deltas may be infinite
            rev = add.T + rem
        single = np.where(rem_ok | add_ok, np.where(edge, rem, add), np.inf)
        return edge, single, np.where(rev_ok, rev, np.inf)

    def _pick(self, single: np.ndarray, rev: np.ndarray, edge: np.ndarray) -> ScannedMove | None:
        """The move with the lowest key among the finite entries of ``single`` and ``rev``."""
        low = min(single.min(), rev.min()) * self._inv_n
        if low == np.inf:
            return None
        # Rounding is monotone, so the moves that round to the lowest key lie within one grid step
        # of the lowest unrounded value; the exact key is computed for those few.
        near = low + 2 * 10.0**-_KEY_DP
        d = self._d
        best: ScannedMove | None = None
        for rank, table in ((0, single), (1, rev)):
            for i in np.flatnonzero(table * self._inv_n <= near).tolist():
                u, v = divmod(i, d)
                op = "reverse" if rank else ("remove" if edge[u, v] else "add")
                cand = (round(float(table[u, v]) * self._inv_n, _KEY_DP), u, v, rank, (op, u, v))
                if best is None or cand[:4] < best[:4]:
                    best = cand
        return best


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
