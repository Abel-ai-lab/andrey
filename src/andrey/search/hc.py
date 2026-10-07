"""Hill climbing over DAGs, the engine behind :func:`andrey.hc`.

The algorithm, its settings, and its references are described on :func:`andrey.hc`.

Each candidate move is scored by the Schur-complement delta of :mod:`andrey.core.score_delta`.
The move budget also regularizes: greedy ascent adds the strongest edges first, so stopping early
keeps noise-fitting edges out of large, dense graphs. The DAG is converted to its CPDAG for output;
the reported objective is the DAG's total BIC, lower is better.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from andrey.core import ARROW, NULL, TAIL, GraphStructure, backend
from andrey.core.orient import dag2cpdag, to_structure
from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore
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


def _reachability(adj_int: list[int], d: int) -> list[int]:
    """Transitive-descendant bitsets: ``out[u]`` has bit ``v`` set iff a directed path ``u ~> v``.

    ``adj_int[u]`` is the out-neighbor bitset of ``u`` (bit ``v`` set iff the edge ``u -> v``
    exists). One breadth-first bit-expansion per source over the current DAG.
    """
    out = [0] * d
    for src in range(d):
        reach = 0
        frontier = adj_int[src]
        while frontier:
            reach |= frontier
            nxt = 0
            bits = frontier
            while bits:
                lsb = bits & -bits
                nxt |= adj_int[lsb.bit_length() - 1]
                bits ^= lsb
            frontier = nxt & ~reach
        out[src] = reach
    return out


def _reverse_stays_acyclic(adj_int: list[int], reach: list[int], u: int, v: int) -> bool:
    """True when reversing ``u -> v`` to ``v -> u`` keeps the graph acyclic.

    The reversal closes a cycle iff ``u`` still reaches ``v`` through some child other than ``v``
    itself; excluding the edge being removed, no remaining child of ``u`` may reach ``v``.
    """
    target = 1 << v
    children = adj_int[u] & ~target
    while children:
        lsb = children & -children
        if reach[lsb.bit_length() - 1] & target:
            return False
        children ^= lsb
    return True


# A scanned candidate: (delta, u, v, op_rank, (op, u, v)). op_rank orders the ops that share a
# ``(u, v)`` (remove 0 < reverse 1; add is 0 and never coexists with remove), so ``key = r[:4]``
# reproduces serial's "first encountered in (u, v, op) order" tie-break as a strict total order.
ScannedMove = tuple[float, int, int, int, tuple[str, int, int]]


def _scan_moves(
    targets: Sequence[int],
    adj_int: list[int],
    parents_of: dict[int, list[int]],
    delta: DeltaBICScore,
    d: int,
    reach: list[int],
    cache: dict[tuple[int, tuple[int, ...], str, int], float],
) -> ScannedMove | None:
    """The best-improving move whose HEAD ``v`` is in ``targets``, keyed ``(delta, u, v, op_rank)``.

    A move ``u -> v`` is owned by the chunk that owns its head ``v``, so partitioning ``targets`` is
    an exact partition of the move space. Selection is a strict total order over improving
    (``delta < -EPS``) moves, so it is decomposable and visitation-order-independent -- the parallel
    reduce over per-chunk bests reproduces the whole-range scan exactly. A running ``best - EPS``
    comparison would be order-dependent when two deltas differ by less than ``EPS``; the total-order
    key resolves exact symmetries.

    Each delta comes from the Schur-complement update, memoized in ``cache`` by
    ``(target, parents-of-target, op, moved)``; reversing ``u -> v`` decomposes into an add of ``v``
    to ``u``'s parents plus a remove of ``u`` from ``v``'s (additive deltas, independent targets).
    """
    pa = [tuple(parents_of[i]) for i in range(d)]  # each node's parent tuple, built once per scan
    inv_n = 1.0 / delta.score_obj.n  # key on delta/n: noise and signal scale with n (see _KEY_DP)
    best: ScannedMove | None = None
    for u in range(d):
        row = adj_int[u]
        u_bit = 1 << u
        pa_u = pa[u]
        for v in targets:
            if u == v:
                continue
            v_bit = 1 << v
            pa_v = pa[v]
            if row & v_bit:  # edge u -> v exists
                d_rem = _cached_delta(cache, delta, v, pa_v, u, "remove")
                if d_rem < -_EPS_IMPROVE:
                    cand = (round(d_rem * inv_n, _KEY_DP), u, v, 0, ("remove", u, v))
                    if best is None or cand[:4] < best[:4]:
                        best = cand
                if _reverse_stays_acyclic(adj_int, reach, u, v):
                    d_rev = _cached_delta(cache, delta, u, pa_u, v, "add") + d_rem
                    if d_rev < -_EPS_IMPROVE:
                        cand = (round(d_rev * inv_n, _KEY_DP), u, v, 1, ("reverse", u, v))
                        if best is None or cand[:4] < best[:4]:
                            best = cand
            elif not (adj_int[v] & u_bit):  # neither u -> v nor v -> u; add is legal iff acyclic
                if not (reach[v] & u_bit):
                    d_add = _cached_delta(cache, delta, v, pa_v, u, "add")
                    if d_add < -_EPS_IMPROVE:
                        cand = (round(d_add * inv_n, _KEY_DP), u, v, 0, ("add", u, v))
                        if best is None or cand[:4] < best[:4]:
                            best = cand
    return best


def _best_move(
    adj_int: list[int],
    parents_of: dict[int, list[int]],
    delta: DeltaBICScore,
    d: int,
    cache: dict[tuple[int, tuple[int, ...], str, int], float],
) -> tuple[tuple[str, int, int] | None, float]:
    """Scan every legal single-edge move; return the most-improving ``(move, delta)``.

    The whole-range instance of :func:`_scan_moves`; returns ``(None, 0.0)`` when nothing improves.
    """
    reach = _reachability(adj_int, d)
    best = _scan_moves(range(d), adj_int, parents_of, delta, d, reach, cache)
    if best is None:
        return None, 0.0
    return best[4], best[0]


def _cached_delta(
    cache: dict[tuple[int, tuple[int, ...], str, int], float],
    delta: DeltaBICScore,
    v: int,
    pa_v: tuple[int, ...],
    u: int,
    op: str,
) -> float:
    """Return the Schur delta for ``op``-ing ``u`` at target ``v`` with parents ``pa_v``, memoized.

    The value is a pure function of the key, so a hit returns the same float the scalar delta would
    recompute; the reverse move reuses its remove-side component through the same cache.
    """
    key = (v, pa_v, op, u)
    val = cache.get(key)
    if val is None:
        val = delta.delta(v, pa_v, u, op)
        cache[key] = val
    return val


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
        edges; small or sparse problems reach the local optimum first and never hit it.

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
    if X.ndim != 2:
        raise ValueError(f"data must be a 2-D (n_samples, n_features) array, got ndim={X.ndim}")
    d = X.shape[1]

    score = BICScore(X, lambda_value=lambda_value)
    delta = DeltaBICScore(score)

    adj_int: list[int] = [0] * d
    parents_of: dict[int, list[int]] = {i: [] for i in range(d)}
    delta_cache: dict[tuple[int, tuple[int, ...], str, int], float] = {}

    # Pin BLAS to one thread for the delta loops so the Schur updates are thread-count-reproducible
    # and equal to the (1-thread) parallel workers; above the work threshold the pass fans out.
    with threadpool_limits(limits=1):
        workers = backend.worker_count()
        if workers > 1 and d > 1:
            from andrey.search import _parallel_hc  # local import breaks the import cycle

            _parallel_hc.run_hc(adj_int, parents_of, delta, d, max_iter, workers=workers)
        else:
            for _ in range(max_iter):
                move, _ = _best_move(adj_int, parents_of, delta, d, delta_cache)
                if move is None:
                    break
                _apply_move(adj_int, parents_of, move)

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
