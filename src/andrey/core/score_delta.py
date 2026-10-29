"""Schur-complement delta-BIC for one edge move in linear-Gaussian search.

Adding or removing one parent ``u`` of ``v`` costs O(|S|^2) by a rank-one update, instead of
O(|S|^3) for two :func:`~andrey.core.score.local_score_bic` calls. With
``R(v | S) = cov[v,v] - cov[v,S] inv(cov[S,S]) cov[S,v]`` and
``e_vu = cov[v,u] - cov[v,S] inv(cov[S,S]) cov[S,u]``:

* add ``u``: ``R(v | S + u) = R(v | S) - e_vu^2 / R(u | S)``
* drop ``u``: ``R(v | S - u) = R(v | S) + e_vu^2 / R(u | S_new)``

The delta is ``n * (log R_new - log R_old) +/- log_n * lambda`` (``+`` for an add), plus the
change in the extended-BIC term (:func:`~andrey.core.score.ebic_step`) when the score's
``ebic_gamma`` is not 0. A near-singular
``cov[S,S]`` falls back to two full scores (:func:`_get_inv_with_cond`), and
:func:`score_many_with_base` reports, per move, a bound on how far its score can be from the full
score (:func:`_add_error_bounds`). :func:`local_score_bic_delta` scores one move,
:func:`score_many_with_base` many, :func:`add_deltas` and :func:`remove_deltas` every add or every
removal against one base at once, and :class:`DeltaBICScore` keeps the caches a search reuses.
:class:`FullScoreDelta` differences two full local scores of any :class:`~andrey.core.score.Score`,
for a score with no rank-one update, such as BDeu, and :func:`full_add_deltas` scores every add
against one base that way.
"""

from __future__ import annotations

import contextlib
import math
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from typing import Any, cast
from weakref import WeakKeyDictionary

import numpy as np

from .score import (
    ZERO_RESIDUAL,
    BICScore,
    LocalScore,
    Score,
    ebic_penalty,
    ebic_step,
    local_score_bic,
)

# Conditioning threshold for the Schur fast path. The Schur identity loses ~log10(cond) digits;
# at cond=1e10 the residual keeps ~6 digits in float64, well above HC's 1e-9 strict-improvement
# floor. Above this (or on an exactly-singular inverse) the delta is routed to two full scores.
_COND_THRESHOLD = 1e10
_ILL = "__ill__"

# Bound on the batched instance-level inv-cov cache (least-recently-used eviction).
_INV_CACHE_CAP = 200_000

# Byte ceiling on the cached dense ``inv(cov[S, S])`` matrices. At large ``d`` an inverse is
# ``|S|^2 * 8`` bytes, so the entry-count cap alone can pin many gigabytes; this bounds memory in
# the dense large-d regime (the delta-GES path). ``inv(cov[S, S])`` is a pure function of ``S``, so
# byte-eviction only forces a recompute, never a different value.
_INV_CACHE_BYTE_CAP = 2 * 1024**3  # 2 GiB

_log = np.log
_EPS = float(np.finfo(np.float64).eps)


class _InvCache(OrderedDict):
    """LRU cache of ``inv(cov[S, S])`` that also tracks the total bytes of its stored matrices.

    Used wherever batched score calculations reuse covariance inverses
    (:func:`score_many_with_base`, :func:`delta_many_with_base`). An entry is
    ``(inverse, conditioning proxy)``, or the ``_ILL`` sentinel for a singular set; only the
    inverses count toward :attr:`nbytes`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.nbytes = 0
        #: ``(min, max)`` of the covariance diagonal, set by the first batch that needs it.
        self.diag_range: tuple[float, float] | None = None


def _evict_inv(cache: OrderedDict) -> None:
    """Evict oldest inv-cache entries past the entry-count cap, then past the byte budget."""
    while len(cache) > _INV_CACHE_CAP:
        _evict_one_inv(cache)
    if isinstance(cache, _InvCache):
        while cache.nbytes > _INV_CACHE_BYTE_CAP and len(cache) > 1:
            _evict_one_inv(cache)


def _evict_one_inv(cache: OrderedDict) -> None:
    """Pop the least-recently-used inv-cache entry, keeping :attr:`_InvCache.nbytes` in sync."""
    _key, evicted = cache.popitem(last=False)
    if isinstance(cache, _InvCache) and isinstance(evicted, tuple):
        cache.nbytes -= evicted[0].nbytes


def _get_inv_with_cond(
    inv_cov_cache: dict[Any, Any],
    S_sorted: tuple[int, ...],
    cov: np.ndarray,
    S_list: list[int],
    cond_threshold: float,
) -> tuple[np.ndarray | None, bool]:
    """Resolve or build ``inv(cov[S, S])`` and apply the conditioning gate.

    Returns ``(M, ill)``. When ``ill`` is ``True`` the caller must fall back to the full-score path;
    ``M`` is then ``None``. The conditioning proxy ``||cov[S,S]||_inf * ||inv(cov[S,S])||_inf`` uses
    true infinity-norm row sums (an upper bound on the condition number). The inverse is cached at
    ``S_sorted`` with its proxy, so a later call with a different ``cond_threshold`` applies its own
    gate; an exactly singular set caches a ``"__ill__"`` sentinel. Repeated candidate moves on the
    same set skip re-inversion.
    """
    entry = inv_cov_cache.get(S_sorted)
    if entry is None:
        XX = cov[S_list][:, S_list]
        try:
            M_candidate = np.linalg.inv(XX)
        except np.linalg.LinAlgError:
            inv_cov_cache[S_sorted] = _ILL
            return None, True
        XX_norm = float(np.abs(XX).sum(axis=1).max())
        M_norm = float(np.abs(M_candidate).sum(axis=1).max())
        entry = (M_candidate, XX_norm * M_norm)
        inv_cov_cache[S_sorted] = entry
        if isinstance(inv_cov_cache, _InvCache):
            inv_cov_cache.nbytes += M_candidate.nbytes
    elif isinstance(entry, str):
        return None, True
    M, proxy = entry
    if proxy > cond_threshold:
        return None, True
    return M, False


def _insert_sorted(S_sorted: tuple[int, ...], u: int) -> tuple[int, ...]:
    """Insert ``u`` into a sorted tuple, preserving order. Caller guarantees ``u`` not in ``S``."""
    out = list(S_sorted)
    lo, hi = 0, len(out)
    while lo < hi:
        mid = (lo + hi) // 2
        if out[mid] < u:
            lo = mid + 1
        else:
            hi = mid
    out.insert(lo, u)
    return tuple(out)


def _remove_sorted(S_sorted: tuple[int, ...], u: int) -> tuple[int, ...]:
    """Remove ``u`` from a sorted tuple. Caller guarantees ``u`` in ``S``."""
    return tuple(x for x in S_sorted if x != u)


def _delta_via_full_score(
    cov: np.ndarray,
    n: int,
    v: int,
    S_old: Sequence[int],
    u: int,
    op: str,
    *,
    penalty_discount: float,
    log_n: float,
    ebic_gamma: float = 0.0,
) -> float:
    """Fallback delta from two full :func:`local_score_bic` evaluations (exact, O(|S|^3)).

    It is ``inf`` when either family is unavailable.
    """
    if op == "add":
        S_new: list[int] = sorted([*S_old, u])
    elif op == "remove":
        S_new = [x for x in S_old if x != u]
    else:
        raise ValueError(f"unknown op: {op!r} (expected 'add' or 'remove')")
    s_old = local_score_bic(
        cov,
        n,
        v,
        list(S_old),
        penalty_discount=penalty_discount,
        ebic_gamma=ebic_gamma,
        log_n=log_n,
    )
    s_new = local_score_bic(
        cov, n, v, S_new, penalty_discount=penalty_discount, ebic_gamma=ebic_gamma, log_n=log_n
    )
    return _difference(s_new, s_old)


def _difference(s_new: float, s_old: float) -> float:
    """``s_new - s_old``, or ``inf`` when either family is unavailable (an ``inf`` score)."""
    if s_new == np.inf or s_old == np.inf:
        return np.inf
    return float(s_new - s_old)


def local_score_bic_delta(
    cov: np.ndarray,
    n: int,
    v: int,
    S_old_sorted: tuple[int, ...],
    u: int,
    op: str,
    *,
    penalty_discount: float,
    log_n: float,
    ebic_gamma: float = 0.0,
    inv_cov_cache: dict[Any, Any],
    residual_cache: dict[Any, float],
    cond_threshold: float = _COND_THRESHOLD,
) -> float:
    """Return ``score(v, S_new) - score(v, S_old)`` for a single ``add``/``remove`` of ``u``.

    Computes the delta in O(|S|^2) via the rank-one residual update, given a cached
    ``inv(cov[S_old, S_old])``. Both caches are mutated; new residuals are written back keyed to
    ``S_new``. Near-singular ``cov[S, S]`` (conditioning proxy above ``cond_threshold``, or a
    ``LinAlgError``), an add past ``n - 2`` parents, and a residual ``R(u) <= 0`` or ``R(v)`` at
    most :data:`~andrey.core.score.ZERO_RESIDUAL` times ``cov[v, v]`` route to the two-full-score
    fallback, which is ``inf`` when either family is unavailable.

    Parameters
    ----------
    cov : ndarray, shape (d, d)
        Sample covariance matrix.
    n : int
        Sample count.
    v : int
        Target node whose local score is differenced.
    S_old_sorted : tuple of int
        Current parent set BEFORE the move (sorted, hashable).
    u : int
        Variable being added or removed.
    op : {"add", "remove"}
        Move direction.
    penalty_discount : float
        Penalty discount on the BIC complexity term.
    log_n : float
        Precomputed ``log(n)``.
    ebic_gamma : float, default=0.0
        Weight on the extended-BIC term over the ``d`` columns of ``cov``
        (:func:`~andrey.core.score.ebic_step`); ``0.0`` is the standard BIC.
    inv_cov_cache : dict
        Mutable cache of ``inv(cov[S, S])`` keyed by sorted-tuple parent set.
    residual_cache : dict
        Mutable cache of residual variances keyed by ``(target, sorted_S_tuple)``.
    cond_threshold : float
        Conditioning-gate threshold on ``||cov[S,S]||_inf * ||inv||_inf``.
    """
    v = int(v)
    u = int(u)

    key_v_old = (v, S_old_sorted)
    R_v_old = residual_cache.get(key_v_old)
    if R_v_old is None:
        if not S_old_sorted:
            R_v_old = float(cov[v, v])
        else:
            S_list = list(S_old_sorted)
            M_old, ill = _get_inv_with_cond(
                inv_cov_cache, S_old_sorted, cov, S_list, cond_threshold
            )
            if ill:
                return _delta_via_full_score(
                    cov,
                    n,
                    v,
                    list(S_old_sorted),
                    u,
                    op,
                    penalty_discount=penalty_discount,
                    log_n=log_n,
                    ebic_gamma=ebic_gamma,
                )
            cs_v = cov[S_list, v]
            R_v_old = float(cov[v, v] - cs_v @ (M_old @ cs_v))
        residual_cache[key_v_old] = R_v_old

    floor = ZERO_RESIDUAL * float(cov[v, v])
    if not R_v_old > floor or (op == "add" and len(S_old_sorted) + 1 > n - 2):
        return _delta_via_full_score(
            cov,
            n,
            v,
            list(S_old_sorted),
            u,
            op,
            penalty_discount=penalty_discount,
            log_n=log_n,
            ebic_gamma=ebic_gamma,
        )

    if op == "add":
        if not S_old_sorted:
            R_u_old = float(cov[u, u])
            e_vu = float(cov[v, u])
        else:
            S_list = list(S_old_sorted)
            M_old, ill = _get_inv_with_cond(
                inv_cov_cache, S_old_sorted, cov, S_list, cond_threshold
            )
            if ill:
                return _delta_via_full_score(
                    cov,
                    n,
                    v,
                    list(S_old_sorted),
                    u,
                    op,
                    penalty_discount=penalty_discount,
                    log_n=log_n,
                    ebic_gamma=ebic_gamma,
                )
            cs_u = cov[S_list, u]
            beta_u = M_old @ cs_u
            R_u_old = residual_cache.get((u, S_old_sorted))
            if R_u_old is None:
                R_u_old = float(cov[u, u] - cs_u @ beta_u)
                residual_cache[(u, S_old_sorted)] = R_u_old
            e_vu = float(cov[v, u] - cov[v, S_list] @ beta_u)

        if R_u_old <= 0.0:
            return _delta_via_full_score(
                cov,
                n,
                v,
                list(S_old_sorted),
                u,
                op,
                penalty_discount=penalty_discount,
                log_n=log_n,
                ebic_gamma=ebic_gamma,
            )
        R_v_new = R_v_old - e_vu * (e_vu / R_u_old)
        if R_v_new <= floor:
            return _delta_via_full_score(
                cov,
                n,
                v,
                list(S_old_sorted),
                u,
                op,
                penalty_discount=penalty_discount,
                log_n=log_n,
                ebic_gamma=ebic_gamma,
            )
        delta = n * (_log(R_v_new) - _log(R_v_old)) + log_n * penalty_discount
        if ebic_gamma:
            delta += ebic_step(len(S_old_sorted), cov.shape[0], ebic_gamma)
        S_new_sorted = _insert_sorted(S_old_sorted, u)
        residual_cache[(v, S_new_sorted)] = float(R_v_new)
        return float(delta)

    if op == "remove":
        S_new_sorted = _remove_sorted(S_old_sorted, u)
        if not S_new_sorted:
            R_u_new = float(cov[u, u])
            e_vu = float(cov[v, u])
        else:
            S_new_list = list(S_new_sorted)
            M_new, ill = _get_inv_with_cond(
                inv_cov_cache, S_new_sorted, cov, S_new_list, cond_threshold
            )
            if ill:
                return _delta_via_full_score(
                    cov,
                    n,
                    v,
                    list(S_old_sorted),
                    u,
                    op,
                    penalty_discount=penalty_discount,
                    log_n=log_n,
                    ebic_gamma=ebic_gamma,
                )
            cs_u = cov[S_new_list, u]
            beta_u = M_new @ cs_u
            R_u_new = residual_cache.get((u, S_new_sorted))
            if R_u_new is None:
                R_u_new = float(cov[u, u] - cs_u @ beta_u)
                residual_cache[(u, S_new_sorted)] = R_u_new
            e_vu = float(cov[v, u] - cov[v, S_new_list] @ beta_u)

        if R_u_new <= 0.0:
            return _delta_via_full_score(
                cov,
                n,
                v,
                list(S_old_sorted),
                u,
                op,
                penalty_discount=penalty_discount,
                log_n=log_n,
                ebic_gamma=ebic_gamma,
            )
        R_v_new = R_v_old + e_vu * (e_vu / R_u_new)
        if R_v_new <= floor:
            return _delta_via_full_score(
                cov,
                n,
                v,
                list(S_old_sorted),
                u,
                op,
                penalty_discount=penalty_discount,
                log_n=log_n,
                ebic_gamma=ebic_gamma,
            )
        delta = n * (_log(R_v_new) - _log(R_v_old)) - log_n * penalty_discount
        if ebic_gamma:
            delta -= ebic_step(len(S_new_sorted), cov.shape[0], ebic_gamma)
        residual_cache[(v, S_new_sorted)] = float(R_v_new)
        return float(delta)

    raise ValueError(f"unknown op: {op!r} (expected 'add' or 'remove')")


# Instance-level inv-cov caches keyed by the score object, so the batched surfaces amortize
# inversions across calls without mutating :class:`BICScore`. Entries are dropped automatically when
# a score is garbage-collected.
_INV_CACHES: WeakKeyDictionary[BICScore, _InvCache] = WeakKeyDictionary()


def _inv_cache(score_obj: BICScore) -> _InvCache:
    """Return the inv-cov cache of ``score_obj``, created on first use."""
    cache = _INV_CACHES.get(score_obj)
    if cache is None:
        cache = _InvCache()
        _INV_CACHES[score_obj] = cache
    return cache


def _add_error_bounds(
    n: int,
    g: float,
    proxy: float,
    b_1: float,
    dR_floor: float,
    diag_range: tuple[float, float],
    s: int,
    t_U: np.ndarray,
    R_uU: np.ndarray,
    R_v_new: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Bound, per batched add, how far its score is from the exact score and from a full score.

    ``M`` is ``inv(A)`` for ``A = cov[S, S]``, solved column by column after an LU factorization:
    column ``j`` solves ``(A + dA_j) x_j = e_j`` with ``|dA_j| <= g |A|``, where
    ``g = 4 (|S| + 2) eps`` covers the factorization, the solves, and each later dot product. A
    product ``x^T M y`` is then within ``g |M x|^T |A| |M| |y|`` of ``x^T inv(A) y``, at most
    ``g proxy ||M x||_1 ||y||_inf`` with ``proxy = ||A||_inf ||M||_inf``, the condition number the
    inverse cache keeps. No covariance exceeds the largest variance ``c`` in size, so with
    ``b = M c_v``:

    * ``R(v | S) = cov[v, v] - c_v^T M c_v`` is within ``dR_v = g proxy ||b||_1 c + dR_floor``,
      where ``dR_floor`` holds the rounding of the residuals and of the scores, here and in the
      full score the result is compared with;
    * ``e_vu = cov[v, u] - c_v^T M c_u`` is within ``de = g proxy ||b||_1 c``;
    * ``R(u | S)`` is within ``dR_u = g proxy sqrt(|S| ||M||_inf c) c``, the same for every
      candidate, as ``||M c_u||_1^2 <= |S| ||M||_2 c_u^T M c_u <= |S| ||M||_inf c`` and
      ``||M||_inf <= proxy`` divided by the smallest variance.

    These errors move ``t = e_vu^2 / R(u | S)`` by at most
    ``dt = 2 sqrt(t / R(u | S)) de + t dR_u / R(u | S)``, as ``|e_vu| / R(u | S) =
    sqrt(t / R(u | S))``. The score is ``n log R(v | S + u)`` plus the penalty, with
    ``R(v | S + u) = R(v | S) - t``, so to first order it is within ``n (dR_v + dt) / R(v | S + u)``
    of both. Returns the bounds and ``dR_u``; a move is in the first-order regime when ``dR_u`` is
    below half of its ``R(u | S)`` and its bound below ``n / 2``.

    References
    ----------
    N. J. Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002,
    Sections 3.1, 9.3, and 14.3.
    """
    var_min, var_max = diag_range
    de = g * proxy * b_1 * var_max
    dR_u = (
        g * proxy * math.sqrt(s * proxy / var_min * var_max) * var_max if var_min > 0 else math.inf
    )
    q = t_U / R_uU
    bound = np.sqrt(q)
    bound *= 2 * de
    bound += q * dR_u
    bound += de + dR_floor
    bound *= n
    bound /= R_v_new
    return bound, dR_u


def _remove_error_bounds(
    n: int,
    g: float,
    proxy: float,
    b_1: float,
    dR_floor: float,
    diag_range: tuple[float, float],
    R_v_old: float,
    b: np.ndarray,
    m: np.ndarray,
    t: np.ndarray,
    R_v_new: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Bound, per member of ``S``, how far the score of removing it is from the exact score and
    from a full score.

    With the norms of :func:`_add_error_bounds`, removing position ``j`` reads ``b_j = (M c_v)_j``
    within ``db = g proxy ||b||_1`` and ``m = M_jj`` within ``dm = g proxy max |M|``, at most
    ``g proxy^2 / (smallest variance)``, and raises ``R(v | S)`` by ``t = b_j^2 / m``, which moves
    by at most ``dt = (2 |b_j| db + t dm) / m``. ``R(v | S)`` is within ``dR_v``, so the score is
    within ``n (2 dR_v / R(v | S) + dt / R(v | S - u))`` of the exact score and of a full score.
    Returns the bounds, ``dm`` and ``dR_v``; the moves are in the first-order regime where ``dm``
    is below half of ``m``, ``dR_v`` below half of ``R(v | S)``, and the bound below ``n / 2``.
    """
    var_min, var_max = diag_range
    db = g * proxy * b_1
    dR_v = db * var_max + dR_floor
    dm = g * proxy * proxy / var_min if var_min > 0 else math.inf
    bound = np.abs(b)
    bound *= 2 * db
    bound += t * dm
    bound /= m
    bound /= R_v_new
    bound += 2 * dR_v / R_v_old
    bound *= n
    return bound, dm, dR_v


def _penalty_size(score_obj: BICScore, s: int) -> float:
    """An upper bound on the penalty of a family one move away from a base of ``s`` parents.

    It is the linear penalty of ``s + 1`` parents, plus, when ``ebic_gamma`` is not 0, the largest
    extended-BIC term, that of half the other variables.
    """
    penalty = score_obj.log_n * (s + 1) * score_obj.penalty_discount
    if score_obj.ebic_gamma:
        n_vars = score_obj.cov.shape[0]
        penalty += ebic_penalty((n_vars - 1) // 2, n_vars, score_obj.ebic_gamma)
    return penalty


def _score_rounding(n: int, penalty: float, var_v: float) -> float:
    """The rounding of ``n log R + penalty``, here and in a full score, as a residual error per unit
    of ``cov[v, v]``. A kept residual lies in ``(1e-12 cov[v, v], cov[v, v]]``, so the score is at
    most ``n (|log cov[v, v]| + 28) + penalty`` in size; a few units in its last place cover both
    computations."""
    return 1.01 * 4 * _EPS * (n * (abs(math.log(var_v)) + 28.0) + penalty) / n


def score_many_with_base(
    score_obj: BICScore,
    i: int,
    base_parents: Sequence[int],
    deltas: Sequence[tuple[str, int]],
    *,
    cond_threshold: float = _COND_THRESHOLD,
    bounds: list[float] | None = None,
) -> list[float]:
    """Return absolute scores ``[score(i, base +/- delta_k)]`` for a batch of single-variable moves.

    ``deltas`` is a sequence of ``("add", u)`` / ``("remove", u)`` tuples; results are returned in
    delta order. Each score is ``n log R(i | base +/- u)`` plus the penalty, with the residual
    updated from ``R(i | base)`` by one rank-one Schur step: adds over the shared
    ``inv(cov[base, base])`` and ``R(u | base)``, removes over its diagonal. Every score starts
    from the base's residual, never from another batched score. Candidates
    masked out by the safety guards, moves not a valid single add/remove of ``base``, and updates
    outside the first-order regime of their error bound fall back to a full ``score`` call.

    ``bounds`` receives, per move, a bound on how far the score can be from
    :meth:`BICScore.score` of the same parent set (:func:`_add_error_bounds`,
    :func:`_remove_error_bounds`), and 0 for a full score. A search that compares these scores
    with a tolerance decides as full scores would wherever the bounds leave the comparison's side
    unchanged, and scores the rest in full. The inv-cov cache (capped at 200k entries) is keyed to
    ``score_obj`` and created on first use; :class:`BICScore` is unchanged.
    """
    k = len(deltas)
    results: list[float | None] = [None] * k
    out_bounds = [0.0] * k
    if not deltas:
        if bounds is not None:
            bounds[:] = out_bounds
        return []

    cov = score_obj.cov
    n = score_obj.n
    log_n = score_obj.log_n
    lam = score_obj.penalty_discount
    gamma = score_obj.ebic_gamma
    n_vars = cov.shape[0]
    cov_diag = score_obj._cov_diag
    inv_cov_cache = _inv_cache(score_obj)

    i = int(i)
    S_base_sorted: tuple[int, ...] = tuple(sorted(int(x) for x in base_parents))
    s = len(S_base_sorted)
    S_set = set(S_base_sorted)
    ops, moves = zip(*deltas)
    n_add = ops.count("add")
    add_idx: Sequence[int] = ()
    add_us: Sequence[int] = ()
    remove_idx: Sequence[int] = ()
    remove_us: Sequence[int] = ()
    if n_add == k and i not in moves and S_set.isdisjoint(moves):
        add_idx, add_us = range(k), moves
    elif n_add == 0 and ops.count("remove") == k and S_set.issuperset(moves):
        remove_idx, remove_us = range(k), moves
    else:
        add_idx, add_us, remove_idx, remove_us = [], [], [], []
        for idx, (op, u) in enumerate(deltas):
            if op == "add":
                if u != i and u not in S_set:
                    add_idx.append(idx)
                    add_us.append(u)
            elif op == "remove":
                if u in S_set:
                    remove_idx.append(idx)
                    remove_us.append(u)
            else:
                raise ValueError(f"unknown delta op: {op!r} (expected 'add' or 'remove')")

    # R(v | base), the residual every move updates; None leaves every move to a full score.
    var_v = float(cov_diag[i])
    floor = ZERO_RESIDUAL * var_v
    R_v_old: float | None = var_v
    if S_base_sorted:
        S_list = list(S_base_sorted)
        M, ill = _get_inv_with_cond(inv_cov_cache, S_base_sorted, cov, S_list, cond_threshold)
        if S_base_sorted in inv_cov_cache:
            inv_cov_cache.move_to_end(S_base_sorted)
            _evict_inv(inv_cov_cache)
        if ill or M is None:
            R_v_old = None
        else:
            proxy = inv_cov_cache[S_base_sorted][1]  # ||A||_inf ||M||_inf
            rows = cov[S_list]
            cs_v = rows[:, i]
            b = cs_v @ M
            R_v_old = float(var_v - b @ cs_v)
            g = 4 * (s + 2) * _EPS
            b_1 = float(np.abs(b).sum())
    if R_v_old is not None and not R_v_old > floor:
        R_v_old = None
    if R_v_old is not None:
        rounding = _score_rounding(n, _penalty_size(score_obj, s), var_v)
    diag_range = inv_cov_cache.diag_range
    if diag_range is None:
        diag_range = inv_cov_cache.diag_range = (float(cov_diag.min()), float(cov_diag.max()))

    def _keep(
        idxs: Sequence[int], values: np.ndarray, bound: np.ndarray, ok: np.ndarray | None
    ) -> None:
        """Store the scores and bounds that ``ok`` keeps (every one for ``None``); a guarded-out
        move stays ``None``."""
        value_list = values.tolist()
        bound_list = bound.tolist()
        if ok is None and len(idxs) == k:
            results[:] = value_list
            out_bounds[:] = bound_list
            return
        for pos in range(len(idxs)) if ok is None else np.flatnonzero(ok).tolist():
            results[idxs[pos]] = value_list[pos]
            out_bounds[idxs[pos]] = bound_list[pos]

    if add_us and R_v_old is not None and s + 1 <= n - 2:
        us_arr = np.asarray(add_us, dtype=np.intp)
        if not S_base_sorted:
            e_vU = cov[i, us_arr]
            R_uU = cov_diag[us_arr]
        else:
            cs_U = rows[:, us_arr]
            e_vU = cov[i, us_arr] - b @ cs_U
            R_uU = cov_diag[us_arr] - np.einsum("ij,ij->j", cs_U, M @ cs_U)
        penalty = log_n * (s + 1) * lam
        if gamma:
            penalty += ebic_penalty(s + 1, n_vars, gamma)
        # The first-order bounds and the zero floor need every residual safely positive; a batch
        # that has one that is not takes the guarded path, where masks drop each such move.
        R_u_min = float(R_uU.min())
        safe = R_u_min > 0.0
        if safe:
            t_U = e_vU * (e_vU / R_uU)
            R_v_new = R_v_old - t_U
            safe = float(R_v_new.min()) > 2 * floor
        with np.errstate(all="ignore") if not safe else contextlib.nullcontext():
            if not safe:
                t_U = e_vU * (e_vU / R_uU)
                R_v_new = R_v_old - t_U
            values = np.log(R_v_new)
            values *= n
            values += penalty
            if not S_base_sorted:
                # One rounding each for the quotient, the product, and the difference.
                bound = (n * (4 * _EPS + rounding) * var_v) / R_v_new
                dR_u = 0.0
            else:
                bound, dR_u = _add_error_bounds(
                    n, g, proxy, b_1, (g + rounding) * var_v, diag_range, s, t_U, R_uU, R_v_new
                )
            # The bound below n / 2 keeps the residual's error below half of it, so a residual
            # above twice the floor stays above the floor.
            if safe and dR_u < 0.5 * R_u_min and float(bound.max()) < 0.5 * n:
                ok = None
            else:
                ok = (R_uU > 2 * dR_u) & (bound < 0.5 * n) & (R_v_new > 2 * floor)
        _keep(add_idx, values, bound, ok)

    if remove_us and R_v_old is not None:
        if s == 1:
            for idx in remove_idx:
                results[idx] = score_obj.score(i, [])
        else:
            assert M is not None  # a base with a residual has an inverse
            idx_of = {p: pos for pos, p in enumerate(S_base_sorted)}
            pos_arr = np.asarray([idx_of[u] for u in remove_us], dtype=np.intp)
            m_all = np.diag(M)
            penalty = log_n * (s - 1) * lam
            if gamma:
                penalty += ebic_penalty(s - 1, n_vars, gamma)
            safe = float(m_all.min()) > 0.0 and R_v_old > 2 * floor
            with np.errstate(all="ignore") if not safe else contextlib.nullcontext():
                t_all = b * (b / m_all)
                R_all = R_v_old + t_all
                bound_all, dm, dR_v = _remove_error_bounds(
                    n,
                    g,
                    proxy,
                    b_1,
                    g * var_v + rounding * R_v_old / 2,
                    diag_range,
                    R_v_old,
                    b,
                    m_all,
                    t_all,
                    R_all,
                )
                R_v_new = R_all[pos_arr]
                bound = bound_all[pos_arr]
                values = np.log(R_v_new)
                values *= n
                values += penalty
                if (
                    safe
                    and dm < 0.5 * float(m_all.min())
                    and dR_v < 0.5 * R_v_old
                    and float(bound.max()) < 0.5 * n
                ):
                    ok = None
                else:
                    ok = (m_all[pos_arr] > 2 * dm) & (bound < 0.5 * n) & (R_v_new > 2 * floor)
                    ok &= dR_v < 0.5 * R_v_old
            _keep(remove_idx, values, bound, ok)

    if None in results:
        for idx, value in enumerate(results):
            if value is None:
                op, u = deltas[idx]
                u = int(u)
                parents = (
                    sorted([*S_base_sorted, u])
                    if op == "add"
                    else [x for x in S_base_sorted if x != u]
                )
                results[idx] = score_obj.score(i, parents)
    if bounds is not None:
        bounds[:] = out_bounds
    return cast("list[float]", results)  # every move now has a score


def move_bound(
    score_obj: BICScore,
    i: int,
    base_parents: Sequence[int],
    op: str,
    u: int,
    *,
    cond_threshold: float = _COND_THRESHOLD,
) -> float:
    """A tighter bound on one move's score from :func:`score_many_with_base`: componentwise.

    The batch bounds every move in norms, which costs a few vector operations; this bound follows
    the same first-order analysis without the norms, at the cost of the base's ``|M| |A|``
    products, for a search that needs one comparison settled. ``M`` is ``inv(A)`` for
    ``A = cov[S, S]``, whose column ``j`` solves ``(A + dA_j) x_j = e_j`` with
    ``|dA_j| <= g |A|``, ``g = 4 (|S| + 2) eps``, so ``x^T M y`` is within
    ``g |M x|^T |A| |M| |y|`` of ``x^T inv(A) y``. With ``b = M c_v`` and
    ``w = g |M|^T |A| |b|``, ``R(v | S)`` is within ``w^T |c_v|`` plus the rounding terms of
    :func:`score_many_with_base`. An add's ``e_vu`` is within ``w^T |c_u|`` and its ``R(u | S)``
    within ``g |M c_u|^T |A| |M| |c_u|``; a removal's ``b_j`` is within ``w_j`` and its ``M_jj``
    within ``g |M_{:, j}|^T |A| |M_{:, j}|``. These enter the score as in
    :func:`_add_error_bounds` and :func:`_remove_error_bounds`. Returns ``inf`` for a move the batch
    scores in full and outside the first-order regime.

    References
    ----------
    N. J. Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002,
    Sections 3.1, 9.3, and 14.3.
    """
    cov = score_obj.cov
    n = score_obj.n
    cov_diag = score_obj._cov_diag
    i, u = int(i), int(u)
    S = tuple(sorted(int(x) for x in base_parents))
    s = len(S)
    var_v = float(cov_diag[i])
    floor = ZERO_RESIDUAL * var_v
    if not var_v > 0.0:
        return math.inf
    rounding = _score_rounding(n, _penalty_size(score_obj, s), var_v)
    with np.errstate(all="ignore"):
        if not S:
            if op != "add" or u == i:
                return math.inf
            e = float(cov[i, u])
            R_new = var_v - e * (e / float(cov_diag[u]))
            bound = n * (4 * _EPS + rounding) * var_v / R_new
        else:
            S_list = list(S)
            M, ill = _get_inv_with_cond(_inv_cache(score_obj), S, cov, S_list, cond_threshold)
            if ill or M is None:
                return math.inf
            g = 4 * (s + 2) * _EPS
            abs_A = np.abs(cov[np.ix_(S_list, S_list)])
            abs_M = np.abs(M)
            c_v = cov[S_list, i]
            b = c_v @ M
            R_old = float(var_v - b @ c_v)
            w = g * (abs_M.T @ (abs_A @ np.abs(b)))
            dR_v = float(w @ np.abs(c_v))
            if op == "add" and u != i and u not in S:
                c_u = cov[S_list, u]
                beta = M @ c_u
                e = float(cov[i, u] - b @ c_u)
                R_u = float(cov_diag[u] - c_u @ beta)
                t = e * (e / R_u)
                R_new = R_old - t
                de = float(w @ np.abs(c_u))
                dR_u = g * float(np.abs(beta) @ (abs_A @ (abs_M @ np.abs(c_u))))
                if not dR_u < 0.5 * R_u:
                    return math.inf
                dt = (2 * abs(e) * de + t * dR_u) / R_u
                bound = n * (dR_v + (g + rounding) * var_v + dt) / R_new
            elif op == "remove" and u in S and s > 1:
                j = S.index(u)
                m = float(M[j, j])
                col = abs_M[:, j]
                dm = g * float(col @ (abs_A @ col))
                dR_v += g * var_v + rounding * R_old / 2
                t = float(b[j]) * (float(b[j]) / m)
                R_new = R_old + t
                if not (m > 0.0 and dm < 0.5 * m and dR_v < 0.5 * R_old):
                    return math.inf
                dt = (2 * abs(float(b[j])) * float(w[j]) + t * dm) / m
                bound = n * (2 * dR_v / R_old + dt / R_new)
            else:
                return math.inf
    if not (bound < 0.5 * n and R_new > 2 * floor):
        return math.inf
    return float(bound)


def _delta_two_scores(
    score_obj: Score, i: int, S_base_sorted: tuple[int, ...], u: int, op: str
) -> float:
    """Raw delta ``score(i, base +/- u) - score(i, base)`` from two memoized full scores.

    The exact fallback for :func:`delta_many_with_base`: whenever the Schur fast path is unavailable
    (ill-conditioned base, or a candidate masked by the ``R > 0`` safety guards) the delta is served
    by two :meth:`BICScore.score` calls, which are memoized and pure, so the fallback delta is
    deterministic: the difference of two full local scores, or ``inf`` when either family is
    unavailable.
    """
    u = int(u)
    if op == "add":
        S_new: list[int] = sorted([*S_base_sorted, u])
    elif op == "remove":
        S_new = [x for x in S_base_sorted if x != u]
    else:
        raise ValueError(f"unknown delta op: {op!r} (expected 'add' or 'remove')")
    return _difference(score_obj.score(i, S_new), score_obj.score(int(i), list(S_base_sorted)))


def _add_bounds(
    n: int,
    penalty: float,
    M: np.ndarray,
    cov_SS_diag: np.ndarray,
    cs_v: np.ndarray,
    cs_U: np.ndarray,
    R_v_old: float,
    R_uU: np.ndarray,
    e_vU: np.ndarray,
    floor: float,
) -> np.ndarray:
    """Per-add bounds on how far a batched Schur add delta moves when another batch scores the add.

    Every batch over one base reads the same ``M = inv(cov[S, S])``, ``cov``, and ``R(v | S)``.
    Only ``M @ cs_U``, ``cs_v @ beta_U``, and the ``R_u`` einsum round differently. Each entry of
    them is a ``k = |S|``-term dot product within ``gamma_k |x|^T |M| |y|`` of the exact one in any
    summation order. With ``d = sqrt(diag cov[S, S])``, ``|x|^T |M| |y|`` is at most
    ``||D M D||_F ||x / d|| ||y / d||``, which does not depend on the scale of the columns. Two
    batches therefore differ by at most ``rho`` in ``R_u`` and ``eta`` in ``e_vu``. The largest
    rise of ``t = e_vu^2 / R_u`` over those intervals bounds how far ``R(v | S + u) = R(v | S) - t``
    moves either way, which bounds the change in its log; rounding of the logs adds a few units in
    the last place.

    Where the intervals leave the sign of ``R_u`` or the position of ``R(v | S + u)`` against
    ``floor`` open, another batch can take the other side of a guard, and the bound is ``inf``.
    Where both batches certainly fall back to two full scores, it is 0.

    References
    ----------
    N. J. Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002,
    Section 3.1.
    """
    # ||D M D||_F^2 = diag (M * M) diag and ||x / d||^2 = (x * x) / diag. Per batch,
    # 3 (k + 2) eps ||D M D||_F covers gamma_k (2 + gamma_k) ||D M D||_F with room to spare; two
    # batches double it.
    inv_diag = 1.0 / cov_SS_diag
    c = 6 * (M.shape[0] + 2) * _EPS * math.sqrt(float(cov_SS_diag @ (M * M) @ cov_SS_diag))
    w_u2 = inv_diag @ (cs_U * cs_U)
    w_v = math.sqrt(float(inv_diag @ (cs_v * cs_v)))
    with np.errstate(all="ignore"):
        e_abs = np.abs(e_vU)
        rho = c * w_u2 + 2 * _EPS * np.abs(R_uU)
        eta = c * w_v * np.sqrt(w_u2) + 2 * _EPS * e_abs
        ru_lo = R_uU - rho
        t = e_vU * (e_vU / R_uU)  # the scale-free form the delta itself uses
        R_new = R_v_old - t
        t_hi = (e_abs + eta) * ((e_abs + eta) / ru_lo) * (1 + 4 * _EPS)
        # t can rise further than it can fall, so this covers R(v | S + u) moving either way.
        move = t_hi - t + 2 * _EPS * (R_v_old + t_hi)
        ulp = 8 * _EPS * (n * (np.abs(np.log(R_new)) + abs(math.log(R_v_old))) + penalty)
        bound = 2 * (ulp - n * np.log1p(-move / R_new))
        certain = (ru_lo > 0.0) & (R_new - move > floor)
        if certain.all():
            return bound
        bound[~certain] = np.inf
        bound[(R_uU + rho <= 0.0) | ((ru_lo > 0.0) & (R_new + move <= floor))] = 0.0
    return bound


def delta_many_with_base(
    score_obj: BICScore,
    i: int,
    base_parents: Sequence[int],
    deltas: Sequence[tuple[str, int]],
    *,
    cond_threshold: float = _COND_THRESHOLD,
    bounds: list[float] | None = None,
) -> list[float]:
    """Return RAW deltas ``[score(i, base +/- u_k) - score(i, base)]`` for single-variable moves.

    The PURE companion to :func:`score_many_with_base`. Each result is a raw local-score *diff*
    (not an absolute score) for a single ``("add", u)`` / ``("remove", u)`` move off the shared
    ``base``. Purity is the point, and it comes from what this function does *not* do:

    * it shares only the condition-gated ``inv(cov[base, base])`` cache -- ``inv(cov[S, S])`` is a
      pure function of ``S``, so the cache is safe to share with any other Schur consumer;
    * it recomputes ``R(i | S)`` fresh from that inverse every call (einsum-vectorized adds exactly
      as :func:`score_many_with_base`; the diagonal-of-``M`` identity for removes);
    * it writes back **no** chained residual -- the impurity that makes
      :func:`local_score_bic_delta` cache-history-dependent at the ULP.

    A candidate whose base is ill-conditioned (``cond >= cond_threshold`` or a singular inverse),
    whose add passes ``n - 2`` parents, or whose rank-one update leaves a residual that is zero up
    to rounding, falls back to :func:`_delta_two_scores` (two memoized full scores) -- still exact,
    and ``inf`` when either family is unavailable. Results are returned in ``deltas`` order.

    The batched adds round differently for different sets of moves, so one move's delta depends on
    the batch it is scored in. ``bounds`` receives, per move, a bound on that dependence: the most
    the delta can differ when the move is scored in any other batch over the same base
    (:func:`_add_bounds`). It is ``inf`` when another batch can take the other side of an
    ``R > 0`` guard, and 0 for every delta that does not depend on the batch.

    Parameters
    ----------
    score_obj : BICScore
        The base score supplying ``cov``, ``n``, ``log_n``, and ``penalty_discount``.
    i : int
        Target node whose local score is differenced.
    base_parents : sequence of int
        The shared parent set every move perturbs (an add's ``u`` must be outside it; a remove's
        ``u`` inside it).
    deltas : sequence of (str, int)
        ``("add", u)`` / ``("remove", u)`` moves.
    cond_threshold : float
        Conditioning-gate threshold routing near-singular bases to the two-full-score fallback.
    bounds : list of float, optional
        Filled with one batch-dependence bound per move, in ``deltas`` order.
    """
    if bounds is not None:
        bounds[:] = [0.0] * len(deltas)
    if not deltas:
        return []

    cov = score_obj.cov
    n = score_obj.n
    log_n = score_obj.log_n
    lam = score_obj.penalty_discount
    gamma = score_obj.ebic_gamma
    n_vars = cov.shape[0]
    cov_diag = score_obj._cov_diag
    inv_cov_cache = _inv_cache(score_obj)

    i = int(i)
    S_base_sorted: tuple[int, ...] = tuple(sorted(int(x) for x in base_parents))
    floor = ZERO_RESIDUAL * float(cov_diag[i])

    def _get_obs_inv(S_sorted: tuple[int, ...], S_list: list[int]):
        M, ill = _get_inv_with_cond(inv_cov_cache, S_sorted, cov, S_list, cond_threshold)
        if S_sorted in inv_cov_cache:
            inv_cov_cache.move_to_end(S_sorted)
            _evict_inv(inv_cov_cache)
        return M, ill

    adds: list[tuple[int, int]] = []
    removes: list[tuple[int, int]] = []
    for idx, (op, u) in enumerate(deltas):
        if op == "add":
            adds.append((idx, int(u)))
        elif op == "remove":
            removes.append((idx, int(u)))
        else:
            raise ValueError(f"unknown delta op: {op!r} (expected 'add' or 'remove')")

    # R(i | base): the shared "subtract" side, recomputed fresh from the cached inverse (pure).
    # ``M_old`` is None for the empty base; ``R_v_old`` is None when the base is ill-conditioned.
    M_old: np.ndarray | None = None
    if not S_base_sorted:
        R_v_old: float | None = float(cov_diag[i])
    else:
        S_list = list(S_base_sorted)
        M_old, ill_base = _get_obs_inv(S_base_sorted, S_list)
        if ill_base:
            R_v_old = None
        else:
            cs_v = cov[S_list, i]
            R_v_old = float(cov_diag[i] - cs_v @ (M_old @ cs_v))
            if not (R_v_old > floor) or not np.isfinite(R_v_old):
                # A residual that is zero up to rounding marks an unavailable base, or an inverse
                # too inaccurate to trust (cond slipped under the gate); every move falls back.
                R_v_old = None

    results: list[float] = [0.0] * len(deltas)

    if adds:
        add_deltas: dict[int, float] = {}
        in_base = set(S_base_sorted)
        in_base.add(i)
        add_us = sorted({u for _, u in adds if u not in in_base})
        if add_us and R_v_old is not None and len(S_base_sorted) + 1 <= n - 2:
            penalty = log_n * lam
            if gamma:
                penalty += ebic_step(len(S_base_sorted), n_vars, gamma)
            if not S_base_sorted:
                diag = cov_diag.tolist()
                row = cov[i].tolist()
                log_old = _log(R_v_old)
                for u in add_us:
                    R_u_old = diag[u]
                    if R_u_old <= 0.0:
                        continue
                    e_vu = row[u]
                    R_v_new = R_v_old - e_vu * (e_vu / R_u_old)
                    if R_v_new <= floor:
                        continue
                    add_deltas[u] = float(n * (_log(R_v_new) - log_old) + penalty)
            else:
                assert M_old is not None
                S_list = list(S_base_sorted)
                cs_v = cov[S_list, i]
                us_arr = np.asarray(add_us, dtype=np.intp)
                cs_U = cov[S_list][:, us_arr]
                beta_U = M_old @ cs_U
                e_vU = cov[i, us_arr] - cs_v @ beta_U
                R_uU = cov_diag[us_arr] - np.einsum("ij,ij->j", cs_U, beta_U)
                safe_mask = R_uU > 0.0
                safe_R_u = np.where(safe_mask, R_uU, 1.0)
                R_v_new_arr = R_v_old - e_vU * (e_vU / safe_R_u)
                safe_mask = safe_mask & (R_v_new_arr > floor)
                if np.any(safe_mask):
                    deltas_arr = (
                        n * (_log(np.where(safe_mask, R_v_new_arr, 1.0)) - _log(R_v_old)) + penalty
                    )
                    add_deltas = {
                        u: d
                        for u, safe, d in zip(
                            add_us, safe_mask.tolist(), deltas_arr.tolist(), strict=True
                        )
                        if safe
                    }
                if bounds is not None:
                    bound_of = _add_bounds(
                        n,
                        abs(penalty),
                        M_old,
                        cov_diag[S_list],
                        cs_v,
                        cs_U,
                        R_v_old,
                        R_uU,
                        e_vU,
                        floor,
                    )
                    if len(deltas) == len(add_us) and [u for _, u in adds] == add_us:
                        bounds[:] = bound_of.tolist()  # one add per source, in source order
                    else:
                        add_bounds = dict(zip(add_us, bound_of.tolist(), strict=True))
                        for idx, u in adds:
                            bounds[idx] = add_bounds.get(u, 0.0)
        for idx, u in adds:
            d = add_deltas.get(u)
            if d is None:
                d = _delta_two_scores(score_obj, i, S_base_sorted, u, "add")
            results[idx] = d

    if removes:
        remove_deltas: dict[int, float] = {}
        remove_us = sorted({u for _, u in removes if u in S_base_sorted})
        if remove_us and R_v_old is not None and M_old is not None:
            idx_of = {p: pos for pos, p in enumerate(S_base_sorted)}
            Mdiag = np.diag(M_old)
            S_list = list(S_base_sorted)
            cs_v = cov[S_list, i]
            log_Rv_old = float(_log(R_v_old))
            penalty = log_n * lam
            if gamma:
                penalty += ebic_step(len(S_base_sorted) - 1, n_vars, gamma)
            for u in remove_us:
                u_idx = idx_of[u]
                M_uu = float(Mdiag[u_idx])
                if M_uu <= 0.0 or not np.isfinite(M_uu):
                    continue
                R_u_new = 1.0 / M_uu
                M_col = M_old[:, u_idx]
                num = float(cs_v @ M_col) - float(cs_v[u_idx]) * M_uu
                e_vu = float(cov[i, u]) + num / M_uu
                R_v_new = R_v_old + e_vu * (e_vu / R_u_new)
                if R_v_new <= floor or not np.isfinite(R_v_new):
                    continue
                remove_deltas[u] = float(n * (_log(R_v_new) - log_Rv_old) - penalty)
        for idx, u in removes:
            d = remove_deltas.get(u)
            if d is None:
                d = _delta_two_scores(score_obj, i, S_base_sorted, u, "remove")
            results[idx] = d

    return results


def add_deltas(
    score_obj: BICScore,
    i: int,
    base_parents: Iterable[int],
    *,
    cond_threshold: float = _COND_THRESHOLD,
) -> np.ndarray:
    """Return ``score(i, base + u) - score(i, base)`` for every variable ``u``, as one array.

    The vector form of :func:`delta_many_with_base`'s ``add`` moves: ``O(|base|^2 d)`` NumPy work
    instead of a Python loop over the candidates. It shares the conditioning-gated inverse cache and
    falls back to two full scores where the Schur path is unavailable. Entries for ``i`` itself, for
    the members of ``base``, and for adds to an unavailable family or past ``n - 2`` parents are
    ``inf``.

    Parameters
    ----------
    score_obj : BICScore
        The base score supplying ``cov``, ``n``, ``log_n``, and ``penalty_discount``.
    i : int
        Target node whose local score is differenced.
    base_parents : sequence of int
        The parent set every candidate is added to.
    cond_threshold : float
        Conditioning-gate threshold routing a near-singular base to the two-full-score fallback.
    """
    cov = score_obj.cov
    i = int(i)
    S_sorted = tuple(sorted(base_parents))
    S_list = list(S_sorted)
    if len(S_list) + 1 > score_obj.n - 2:
        return np.full(cov.shape[0], np.inf)
    if not S_list:
        e = cov[i]
        R_u = score_obj._cov_diag
    else:
        inv_cov_cache = _inv_cache(score_obj)
        M, ill = _get_inv_with_cond(inv_cov_cache, S_sorted, cov, S_list, cond_threshold)
        inv_cov_cache.move_to_end(S_sorted)
        _evict_inv(inv_cov_cache)
        if ill:
            out = np.full(cov.shape[0], np.inf)
            for u in range(cov.shape[0]):
                if u != i and u not in S_sorted:
                    out[u] = _delta_two_scores(score_obj, i, S_sorted, u, "add")
            return out
        cs = cov[S_list]
        beta = M @ cs
        e = cov[i] - cov[i, S_list] @ beta
        R_u = score_obj._cov_diag - np.einsum("ij,ij->j", cs, beta)
    R_v = float(e[i])  # e_ii is R(i | base)
    penalty = score_obj.log_n * score_obj.penalty_discount
    if score_obj.ebic_gamma:
        penalty += ebic_step(len(S_list), cov.shape[0], score_obj.ebic_gamma)
    with np.errstate(divide="ignore", invalid="ignore"):
        R_v_new = R_v - e * (e / R_u)  # e * e underflows on tiny-variance data
        out = score_obj.n * (_log(R_v_new) - _log(R_v)) + penalty
    # Rounding-level residuals and a nonpositive R_u go to the full scores.
    floor = ZERO_RESIDUAL * float(score_obj._cov_diag[i])
    schur = (R_u > 0.0) & (R_v_new > floor) & np.isfinite(out) & (R_v > floor)
    exact = np.flatnonzero(~schur).tolist()
    out[i] = np.inf
    out[S_list] = np.inf
    for u in exact:
        if u != i and u not in S_sorted:
            out[u] = _delta_two_scores(score_obj, i, S_sorted, u, "add")
    return out


def remove_deltas(
    score_obj: BICScore,
    i: int,
    base_parents: Iterable[int],
    *,
    cond_threshold: float = _COND_THRESHOLD,
) -> np.ndarray:
    """Return ``score(i, base - u) - score(i, base)`` for every ``u`` in ``base``, as one array.

    The removal counterpart of :func:`add_deltas`. With ``M = inv(cov[base, base])`` and
    ``b = M cov[base, i]``, dropping position ``j`` raises the residual by ``b_j^2 / M_jj``; one
    inverse serves every removal. A near-singular base, a base residual that is zero up to rounding,
    or a residual that is not positive and finite triggers two full scores. Entries outside ``base``
    are ``inf``.

    Parameters
    ----------
    score_obj : BICScore
        The base score supplying ``cov``, ``n``, ``log_n``, and ``penalty_discount``.
    i : int
        Target node whose local score is differenced.
    base_parents : sequence of int
        The parent set each member is removed from.
    cond_threshold : float
        Conditioning-gate threshold routing a near-singular base to the two-full-score fallback.
    """
    cov = score_obj.cov
    i = int(i)
    S_sorted = tuple(sorted(base_parents))
    S_list = list(S_sorted)
    out = np.full(cov.shape[0], np.inf)
    if not S_list:
        return out
    inv_cov_cache = _inv_cache(score_obj)
    M, _ = _get_inv_with_cond(inv_cov_cache, S_sorted, cov, S_list, cond_threshold)
    inv_cov_cache.move_to_end(S_sorted)
    _evict_inv(inv_cov_cache)
    if M is None:  # ill-conditioned
        for u in S_list:
            out[u] = _delta_two_scores(score_obj, i, S_sorted, u, "remove")
        return out
    cs_i = cov[S_list, i]
    b = M @ cs_i
    R_v = float(cov[i, i] - cs_i @ b)
    M_diag = np.diag(M)
    penalty = score_obj.log_n * score_obj.penalty_discount
    if score_obj.ebic_gamma:
        penalty += ebic_step(len(S_list) - 1, cov.shape[0], score_obj.ebic_gamma)
    with np.errstate(divide="ignore", invalid="ignore"):
        R_v_new = R_v + b * b / M_diag
        deltas = score_obj.n * (_log(R_v_new) - _log(R_v)) - penalty
    # A finite delta implies positive R(i | base - u); M_jj > 0 keeps the sign.
    floor = ZERO_RESIDUAL * float(score_obj._cov_diag[i])
    exact = ~((M_diag > 0.0) & np.isfinite(deltas) & (R_v > floor))
    for j, u in enumerate(S_list):
        out[u] = _delta_two_scores(score_obj, i, S_sorted, u, "remove") if exact[j] else deltas[j]
    return out


class DeltaBICScore:
    """Stateful companion to :class:`~andrey.core.score.BICScore` for the Schur delta path.

    Wraps a :class:`BICScore` and owns the caches that amortize repeated moves: the inv-cov cache
    that every surface shares, and the residual cache of the scalar :meth:`delta`. It exposes
    :meth:`delta` (scalar) and :meth:`score_many_with_base` (batched) for HC/GST. All BIC
    constants (covariance, sample count, ``log_n``, ``penalty_discount``, ``ebic_gamma``) are read
    from the wrapped score. The batched scores come with a bound on their distance from
    :meth:`BICScore.score` (:func:`score_many_with_base`); the scalar :meth:`delta` routes only
    near-singular parent sets to full scores.

    Parameters
    ----------
    score : BICScore
        The base score whose covariance and constants drive the delta arithmetic.
    cond_threshold : float
        Conditioning-gate threshold on ``||cov[S,S]||_inf * ||inv||_inf``.
    """

    def __init__(self, score: BICScore, *, cond_threshold: float = _COND_THRESHOLD) -> None:
        self.score_obj = score
        self.cond_threshold = float(cond_threshold)
        self.inv_cov_cache: _InvCache = _InvCache()
        self.residual_cache: OrderedDict = OrderedDict()
        # Route the batched surfaces' inv-cov cache to this companion's cache.
        _INV_CACHES[score] = self.inv_cov_cache

    def delta(self, v: int, S_old_sorted: tuple[int, ...], u: int, op: str) -> float:
        """Return ``score(v, S_new) - score(v, S_old)`` for a single ``add``/``remove`` of ``u``."""
        return local_score_bic_delta(
            self.score_obj.cov,
            self.score_obj.n,
            v,
            S_old_sorted,
            u,
            op,
            penalty_discount=self.score_obj.penalty_discount,
            log_n=self.score_obj.log_n,
            ebic_gamma=self.score_obj.ebic_gamma,
            inv_cov_cache=self.inv_cov_cache,
            residual_cache=self.residual_cache,
            cond_threshold=self.cond_threshold,
        )

    def score_many_with_base(
        self,
        i: int,
        base_parents: Sequence[int],
        deltas: Sequence[tuple[str, int]],
        *,
        bounds: list[float] | None = None,
    ) -> list[float]:
        """Return absolute scores ``[score(i, base +/- delta_k)]`` for a batch of single moves.

        ``bounds``, when given, receives each score's error bound (:func:`score_many_with_base`).
        """
        return score_many_with_base(
            self.score_obj,
            i,
            base_parents,
            deltas,
            cond_threshold=self.cond_threshold,
            bounds=bounds,
        )

    def move_bound(self, i: int, base_parents: Sequence[int], op: str, u: int) -> float:
        """Return the componentwise bound of one batched move's score (:func:`move_bound`)."""
        return move_bound(
            self.score_obj, i, base_parents, op, u, cond_threshold=self.cond_threshold
        )

    def delta_many_with_base(
        self,
        i: int,
        base_parents: Sequence[int],
        deltas: Sequence[tuple[str, int]],
        *,
        bounds: list[float] | None = None,
    ) -> list[float]:
        """Return RAW deltas ``[score(i, base +/- delta_k) - score(i, base)]`` for single moves.

        ``bounds``, when given, receives each delta's batch-dependence bound
        (:func:`delta_many_with_base`).
        """
        return delta_many_with_base(
            self.score_obj,
            i,
            base_parents,
            deltas,
            cond_threshold=self.cond_threshold,
            bounds=bounds,
        )


class FullScoreDelta:
    """Move deltas from two full local scores, with the surface of :class:`DeltaBICScore`.

    Every delta is ``score(v, S +/- u) - score(v, S)`` from two memoized full scores; the batched
    methods score all of a call's parent sets in one ``score_many`` call.
    GES and hill climbing use it on scores with no Schur update, such as
    :class:`~andrey.core.score.BDeuScore` and the mixed-data scores.

    Parameters
    ----------
    score : BICScore, BDeuScore, DGScore, or CGScore
        The local score; HC also reads its sample count, ``score.n``.
    """

    def __init__(self, score: LocalScore) -> None:
        self.score_obj = score

    def delta(self, v: int, S_old_sorted: tuple[int, ...], u: int, op: str) -> float:
        """Return ``score(v, S_new) - score(v, S_old)`` for a single ``add``/``remove`` of ``u``."""
        return _delta_two_scores(self.score_obj, int(v), tuple(S_old_sorted), u, op)

    def delta_many_with_base(
        self,
        i: int,
        base_parents: Sequence[int],
        deltas: Sequence[tuple[str, int]],
        *,
        bounds: list[float] | None = None,
    ) -> list[float]:
        """Return RAW deltas ``[score(i, base +/- u_k) - score(i, base)]`` for single moves.

        ``bounds``, when given, receives zeros: memoized full scores do not depend on the batch.
        """
        base = sorted(int(p) for p in base_parents)
        if bounds is not None:
            bounds[:] = [0.0] * len(deltas)
        scores = self.score_obj.score_many(int(i), [base, *_moved(base, deltas)])
        return [float(s - scores[0]) for s in scores[1:]]

    def score_many_with_base(
        self,
        i: int,
        base_parents: Sequence[int],
        deltas: Sequence[tuple[str, int]],
        *,
        bounds: list[float] | None = None,
    ) -> list[float]:
        """Return absolute scores ``[score(i, base +/- u_k)]`` for single moves.

        ``bounds``, when given, receives zeros: every score is a full score.
        """
        if bounds is not None:
            bounds[:] = [0.0] * len(deltas)
        base = [int(p) for p in base_parents]
        return self.score_obj.score_many(int(i), _moved(base, deltas))

    def move_bound(self, i: int, base_parents: Sequence[int], op: str, u: int) -> float:
        """Return 0: every score is a full score."""
        return 0.0


def full_add_deltas(
    score_obj: LocalScore, i: int, base_parents: Iterable[int], d: int
) -> np.ndarray:
    """:func:`add_deltas` for a score with no rank-one update, from two full local scores.

    ``score(i, base + u) - score(i, base)`` for each of the ``d`` variables ``u``, with every parent
    set scored in one ``score_many`` call (:meth:`FullScoreDelta.delta_many_with_base`); ``inf`` for
    ``i`` and the members of ``base``.
    """
    base = sorted(int(p) for p in base_parents)
    taken = {int(i), *base}
    others = [u for u in range(d) if u not in taken]
    out = np.full(d, np.inf)
    if others:
        moves = [("add", u) for u in others]
        out[others] = FullScoreDelta(score_obj).delta_many_with_base(i, base, moves)
    return out


def _moved(base: list[int], deltas: Sequence[tuple[str, int]]) -> list[list[int]]:
    """The parent sets ``base`` plus or minus ``u`` for each ``(op, u)`` in ``deltas``."""
    sets = []
    for op, u in deltas:
        if op not in ("add", "remove"):
            raise ValueError(f"unknown delta op: {op!r} (expected 'add' or 'remove')")
        sets.append([*base, int(u)] if op == "add" else [p for p in base if p != u])
    return sets
