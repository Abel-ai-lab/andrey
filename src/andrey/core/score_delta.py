"""Schur-complement delta-BIC for one edge move in linear-Gaussian search.

Adding or removing one parent ``u`` of ``v`` costs O(|S|^2) by a rank-one update, instead of
O(|S|^3) for two :func:`~andrey.core.score.local_score_bic` calls. With
``R(v | S) = cov[v,v] - cov[v,S] inv(cov[S,S]) cov[S,v]`` and
``e_vu = cov[v,u] - cov[v,S] inv(cov[S,S]) cov[S,u]``:

* add ``u``: ``R(v | S + u) = R(v | S) - e_vu^2 / R(u | S)``
* drop ``u``: ``R(v | S - u) = R(v | S) + e_vu^2 / R(u | S_new)``

The delta is ``n * (log R_new - log R_old) +/- log_n * lambda`` (``+`` for an add). A near-singular
``cov[S,S]`` falls back to two full scores (:func:`_get_inv_with_cond`), so the fast path stays
within ``1e-9`` of the exact difference. :func:`local_score_bic_delta` scores one move,
:func:`score_many_with_base` many, and :class:`DeltaBICScore` keeps the caches a search reuses.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np

from .score import BICScore, local_score_bic

# Conditioning threshold for the Schur fast path. The Schur identity loses ~log10(cond) digits;
# at cond=1e10 the residual keeps ~6 digits in float64, well above HC's 1e-9 strict-improvement
# floor. Above this (or on an exactly-singular inverse) the delta is routed to two full scores.
_COND_THRESHOLD = 1e10
_ILL = "__ill__"

# Bounds on the batched instance-level caches (least-recently-used eviction).
_INV_CACHE_CAP = 200_000
_RESIDUAL_CACHE_CAP = 1_000_000
_DERIVED_SCORE_CACHE_CAP = 1_000_000

# Byte ceiling on the cached dense ``inv(cov[S, S])`` matrices. At large ``d`` an inverse is
# ``|S|^2 * 8`` bytes, so the entry-count cap alone can pin many gigabytes; this bounds memory in
# the dense large-d regime (the delta-GES path). ``inv(cov[S, S])`` is a pure function of ``S``, so
# byte-eviction only forces a recompute, never a different value.
_INV_CACHE_BYTE_CAP = 2 * 1024**3  # 2 GiB

_log = np.log


class _InvCache(OrderedDict):
    """LRU cache of ``inv(cov[S, S])`` that also tracks the total bytes of its stored matrices.

    Used wherever the batched Schur surface amortizes inversions (:func:`score_many_with_base`,
    :func:`delta_many_with_base`). The ``_ILL`` sentinel is a string of negligible size and is not
    counted; only real ndarray inverses contribute to :attr:`nbytes`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.nbytes = 0


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
    if isinstance(cache, _InvCache) and isinstance(evicted, np.ndarray):
        cache.nbytes -= evicted.nbytes


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
    true infinity-norm row sums (an upper bound on the condition number). Ill-conditioned or exactly
    singular sets cache a ``"__ill__"`` sentinel at ``S_sorted`` so repeated candidate moves on the
    same set skip re-inversion.
    """
    M = inv_cov_cache.get(S_sorted)
    if M is None:
        XX = cov[S_list][:, S_list]
        try:
            M_candidate = np.linalg.inv(XX)
        except np.linalg.LinAlgError:
            inv_cov_cache[S_sorted] = _ILL
            return None, True
        XX_norm = float(np.linalg.norm(XX, np.inf))
        M_norm = float(np.linalg.norm(M_candidate, np.inf))
        if XX_norm * M_norm > cond_threshold:
            inv_cov_cache[S_sorted] = _ILL
            return None, True
        inv_cov_cache[S_sorted] = M_candidate
        if isinstance(inv_cov_cache, _InvCache):
            inv_cov_cache.nbytes += M_candidate.nbytes
        return M_candidate, False
    if isinstance(M, str) and M == _ILL:
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
    lambda_value: float,
    log_n: float,
) -> float:
    """Fallback delta from two full :func:`local_score_bic` evaluations (exact, O(|S|^3))."""
    if op == "add":
        S_new: list[int] = sorted([*S_old, u])
    elif op == "remove":
        S_new = [x for x in S_old if x != u]
    else:
        raise ValueError(f"unknown op: {op!r} (expected 'add' or 'remove')")
    s_old = local_score_bic(cov, n, v, list(S_old), lambda_value=lambda_value, log_n=log_n)
    s_new = local_score_bic(cov, n, v, S_new, lambda_value=lambda_value, log_n=log_n)
    return float(s_new - s_old)


def local_score_bic_delta(
    cov: np.ndarray,
    n: int,
    v: int,
    S_old_sorted: tuple[int, ...],
    u: int,
    op: str,
    *,
    lambda_value: float,
    log_n: float,
    inv_cov_cache: dict[Any, Any],
    residual_cache: dict[Any, float],
    cond_threshold: float = _COND_THRESHOLD,
) -> float:
    """Return ``score(v, S_new) - score(v, S_old)`` for a single ``add``/``remove`` of ``u``.

    Computes the delta in O(|S|^2) via the rank-one residual update, given a cached
    ``inv(cov[S_old, S_old])``. Both caches are mutated as side effects: ``inv_cov_cache`` is keyed
    by the sorted-tuple parent set, ``residual_cache`` by ``(target, sorted_S_tuple)``; new
    residuals are written back keyed to ``S_new``. Near-singular ``cov[S, S]`` (conditioning proxy
    above ``cond_threshold``, or a ``LinAlgError``) and defensive ``R(u) <= 0`` / ``R(v_new) <= 0``
    checks route to the two-full-score fallback.

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
    lambda_value : float
        Penalty discount on the BIC complexity term.
    log_n : float
        Precomputed ``log(n)``.
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
                    lambda_value=lambda_value,
                    log_n=log_n,
                )
            cs_v = cov[S_list, v]
            R_v_old = float(cov[v, v] - cs_v @ (M_old @ cs_v))
        residual_cache[key_v_old] = R_v_old

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
                    lambda_value=lambda_value,
                    log_n=log_n,
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
                lambda_value=lambda_value,
                log_n=log_n,
            )
        R_v_new = R_v_old - (e_vu * e_vu) / R_u_old
        if R_v_new <= 0.0:
            return _delta_via_full_score(
                cov,
                n,
                v,
                list(S_old_sorted),
                u,
                op,
                lambda_value=lambda_value,
                log_n=log_n,
            )
        delta = n * (_log(R_v_new) - _log(R_v_old)) + log_n * lambda_value
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
                    lambda_value=lambda_value,
                    log_n=log_n,
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
                lambda_value=lambda_value,
                log_n=log_n,
            )
        R_v_new = R_v_old + (e_vu * e_vu) / R_u_new
        if R_v_new <= 0.0:
            return _delta_via_full_score(
                cov,
                n,
                v,
                list(S_old_sorted),
                u,
                op,
                lambda_value=lambda_value,
                log_n=log_n,
            )
        delta = n * (_log(R_v_new) - _log(R_v_old)) - log_n * lambda_value
        residual_cache[(v, S_new_sorted)] = float(R_v_new)
        return float(delta)

    raise ValueError(f"unknown op: {op!r} (expected 'add' or 'remove')")


def _bounded_get(cache: OrderedDict, key: Any) -> Any:
    """Return ``cache[key]`` and mark it most-recently-used, or ``None`` when absent."""
    if key not in cache:
        return None
    cache.move_to_end(key)
    return cache[key]


def _bounded_put(cache: OrderedDict, key: Any, value: Any, cap: int | None) -> None:
    """Insert ``key -> value`` as most-recently-used, evicting the oldest past ``cap``."""
    if cap == 0:
        return
    cache[key] = value
    cache.move_to_end(key)
    if cap is None:
        return
    while len(cache) > cap:
        cache.popitem(last=False)


# Instance-level Schur caches keyed by the score object, so :func:`score_many_with_base` amortizes
# inversions, residuals, and derived scores across calls without mutating :class:`BICScore`.
# Entries are dropped automatically when a score is garbage-collected.
_DELTA_CACHES: WeakKeyDictionary[BICScore, tuple[OrderedDict, OrderedDict, OrderedDict]] = (
    WeakKeyDictionary()
)


def _delta_caches(score_obj: BICScore) -> tuple[OrderedDict, OrderedDict, OrderedDict]:
    """Return ``(inv_cov_cache, residual_cache, derived_score_cache)``, created on first use."""
    caches = _DELTA_CACHES.get(score_obj)
    if caches is None:
        caches = (_InvCache(), OrderedDict(), OrderedDict())
        _DELTA_CACHES[score_obj] = caches
    return caches


def score_many_with_base(
    score_obj: BICScore,
    i: int,
    base_parents: Sequence[int],
    deltas: Sequence[tuple[str, int]],
    *,
    cond_threshold: float = _COND_THRESHOLD,
) -> list[float]:
    """Return absolute scores ``[score(i, base +/- delta_k)]`` for a batch of single-variable moves.

    ``deltas`` is a sequence of ``("add", u)`` / ``("remove", u)`` tuples; results are returned in
    delta order. Shares one ``inv(cov[base, base])`` across the whole batch. Adds are vectorized
    over candidate ``u`` via ``einsum`` with a ``safe_mask``; removes use the diagonal of ``M_old``
    and ``R(u | S_new) = 1 / M_uu``. Candidates masked out by the safety guards, or moves not a
    valid single add/remove of ``base``, fall back to a full ``score`` call. Instance-level
    inv-cov and residual caches (capped at 200k / 1M entries) are keyed to ``score_obj`` and created
    on first use. Derived absolute scores are stored in a separate cache keyed to ``score_obj``;
    :class:`BICScore` is unchanged. A later call whose base equals an earlier derived set reuses
    that score. These Schur-derived values are within 1e-9 of a fresh recompute on well-conditioned
    inputs.
    """
    if not deltas:
        return []

    cov = score_obj.cov
    n = score_obj.n
    log_n = score_obj.log_n
    lam = score_obj.lambda_value
    cov_diag = score_obj._cov_diag
    inv_cov_cache, residual_cache, derived_score_cache = _delta_caches(score_obj)

    i = int(i)
    S_base_sorted: tuple[int, ...] = tuple(sorted(int(x) for x in base_parents))
    base_score = _bounded_get(derived_score_cache, (i, S_base_sorted))
    if base_score is None:
        base_score = score_obj.score(i, S_base_sorted)

    derived_parent_sets: list[list[int]] = []
    for op, u in deltas:
        u_int = int(u)
        if op == "add":
            derived_parent_sets.append(sorted([*S_base_sorted, u_int]))
        elif op == "remove":
            derived_parent_sets.append([x for x in S_base_sorted if x != u_int])
        else:
            raise ValueError(f"unknown delta op: {op!r} (expected 'add' or 'remove')")

    def _get_obs_inv(S_sorted: tuple[int, ...], S_list: list[int]):
        M, ill = _get_inv_with_cond(inv_cov_cache, S_sorted, cov, S_list, cond_threshold)
        if S_sorted in inv_cov_cache:
            inv_cov_cache.move_to_end(S_sorted)
            _evict_inv(inv_cov_cache)
        return M, ill

    results: list[float] = [0.0] * len(deltas)
    adds: list[tuple[int, int]] = []
    removes: list[tuple[int, int]] = []
    for idx, (op, u) in enumerate(deltas):
        if op == "add":
            adds.append((idx, int(u)))
        else:
            removes.append((idx, int(u)))

    # R(v | base): the shared "subtract" side of every delta in the batch.
    key_v_old = (i, S_base_sorted)
    R_v_old = _bounded_get(residual_cache, key_v_old)
    if R_v_old is None:
        if not S_base_sorted:
            R_v_old = float(cov_diag[i])
        else:
            S_list = list(S_base_sorted)
            M_old, ill = _get_obs_inv(S_base_sorted, S_list)
            if not ill:
                cs_v = cov[S_list, i]
                R_v_old = float(cov_diag[i] - cs_v @ (M_old @ cs_v))
        if R_v_old is not None:
            _bounded_put(residual_cache, key_v_old, R_v_old, _RESIDUAL_CACHE_CAP)

    if adds:
        add_us = sorted({u for _, u in adds if u != i and u not in S_base_sorted})
        add_scores: dict[int, float] = {}
        if add_us and R_v_old is not None:
            if not S_base_sorted:
                for u in add_us:
                    key_u_old = (u, S_base_sorted)
                    R_u_old = _bounded_get(residual_cache, key_u_old)
                    if R_u_old is None:
                        R_u_old = float(cov_diag[u])
                        _bounded_put(residual_cache, key_u_old, R_u_old, _RESIDUAL_CACHE_CAP)
                    if R_u_old <= 0.0:
                        continue
                    e_vu = float(cov[i, u])
                    R_v_new = R_v_old - (e_vu * e_vu) / R_u_old
                    if R_v_new <= 0.0:
                        continue
                    add_scores[u] = float(
                        base_score + n * (_log(R_v_new) - _log(R_v_old)) + log_n * lam
                    )
            else:
                S_list = list(S_base_sorted)
                M_old, ill = _get_obs_inv(S_base_sorted, S_list)
                if not ill:
                    cs_v = cov[S_list, i]
                    us_arr = np.asarray(add_us, dtype=np.intp)
                    cs_U = cov[S_list][:, us_arr]
                    beta_U = M_old @ cs_U
                    e_vU = cov[i, us_arr] - cs_v @ beta_U
                    R_uU = cov_diag[us_arr] - np.einsum("ij,ij->j", cs_U, beta_U)
                    safe_mask = R_uU > 0.0
                    safe_R_u = np.where(safe_mask, R_uU, 1.0)
                    R_v_new_arr = R_v_old - (e_vU * e_vU) / safe_R_u
                    safe_mask = safe_mask & (R_v_new_arr > 0.0)
                    if np.any(safe_mask):
                        deltas_arr = (
                            n * (_log(np.where(safe_mask, R_v_new_arr, 1.0)) - _log(R_v_old))
                            + log_n * lam
                        )
                        for pos, u in enumerate(add_us):
                            if safe_mask[pos]:
                                add_scores[u] = float(base_score + deltas_arr[pos])
        for idx, u in adds:
            score_val = add_scores.get(u)
            if score_val is None:
                score_val = score_obj.score(i, derived_parent_sets[idx])
            results[idx] = score_val

    if removes:
        remove_us = sorted({u for _, u in removes if u in S_base_sorted})
        remove_scores: dict[int, float] = {}
        if remove_us and R_v_old is not None:
            if len(S_base_sorted) == 1:
                only = S_base_sorted[0]
                if only in remove_us:
                    remove_scores[only] = score_obj.score(i, [])
            elif S_base_sorted:
                S_list = list(S_base_sorted)
                M_old, ill = _get_obs_inv(S_base_sorted, S_list)
                if not ill:
                    idx_of = {p: pos for pos, p in enumerate(S_base_sorted)}
                    Mdiag = np.diag(M_old)
                    cs_v = cov[S_list, i]
                    log_Rv_old = float(_log(R_v_old))
                    for u in remove_us:
                        u_idx = idx_of[u]
                        M_uu = float(Mdiag[u_idx])
                        if M_uu <= 0.0 or not np.isfinite(M_uu):
                            continue
                        R_u_new = 1.0 / M_uu
                        M_col = M_old[:, u_idx]
                        num = float(cs_v @ M_col) - float(cs_v[u_idx]) * M_uu
                        e_vu = float(cov[i, u]) + num / M_uu
                        R_v_new = R_v_old + (e_vu * e_vu) / R_u_new
                        if R_v_new <= 0.0 or not np.isfinite(R_v_new):
                            continue
                        remove_scores[u] = float(
                            base_score + n * (_log(R_v_new) - log_Rv_old) - log_n * lam
                        )
        for idx, u in removes:
            score_val = remove_scores.get(u)
            if score_val is None:
                score_val = score_obj.score(i, derived_parent_sets[idx])
            results[idx] = score_val

    # Backfill each derived set's absolute score so a later call anchored on it hits the cache.
    for idx in range(len(deltas)):
        _bounded_put(
            derived_score_cache,
            (i, tuple(derived_parent_sets[idx])),
            results[idx],
            _DERIVED_SCORE_CACHE_CAP,
        )
    return results


def _delta_two_scores(
    score_obj: BICScore, i: int, S_base_sorted: tuple[int, ...], u: int, op: str
) -> float:
    """Raw delta ``score(i, base +/- u) - score(i, base)`` from two memoized full scores.

    The exact fallback for :func:`delta_many_with_base`: whenever the Schur fast path is unavailable
    (ill-conditioned base, or a candidate masked by the ``R > 0`` safety guards) the delta is served
    by two :meth:`BICScore.score` calls, which are memoized and pure, so the fallback delta is
    deterministic and identical to differencing two full local scores.
    """
    u = int(u)
    if op == "add":
        S_new: list[int] = sorted([*S_base_sorted, u])
    elif op == "remove":
        S_new = [x for x in S_base_sorted if x != u]
    else:
        raise ValueError(f"unknown delta op: {op!r} (expected 'add' or 'remove')")
    return float(score_obj.score(i, S_new) - score_obj.score(int(i), list(S_base_sorted)))


def delta_many_with_base(
    score_obj: BICScore,
    i: int,
    base_parents: Sequence[int],
    deltas: Sequence[tuple[str, int]],
    *,
    cond_threshold: float = _COND_THRESHOLD,
) -> list[float]:
    """Return RAW deltas ``[score(i, base +/- u_k) - score(i, base)]`` for single-variable moves.

    The PURE companion to :func:`score_many_with_base`. Each result is a raw local-score *diff*
    (not an absolute score) for a single ``("add", u)`` / ``("remove", u)`` move off the shared
    ``base``. Purity is the point, and it comes from what this function does *not* do:

    * it shares only the condition-gated ``inv(cov[base, base])`` cache -- ``inv(cov[S, S])`` is a
      pure function of ``S``, so the cache is safe to share with any other Schur consumer;
    * it recomputes ``R(i | S)`` fresh from that inverse every call (einsum-vectorized adds exactly
      as :func:`score_many_with_base`; the diagonal-of-``M`` identity for removes);
    * it writes back **no** chained residual and **no** derived score -- the two impurities that
      make :func:`score_many_with_base` cache-history-dependent at the ULP.

    A candidate whose base is ill-conditioned (``cond >= cond_threshold`` or a singular inverse), or
    whose rank-one update trips a ``R <= 0`` safety guard, falls back to :func:`_delta_two_scores`
    (two memoized full scores) -- still exact. Results are returned in ``deltas`` order.

    Parameters
    ----------
    score_obj : BICScore
        The base score supplying ``cov``, ``n``, ``log_n``, and ``lambda_value``.
    i : int
        Target node whose local score is differenced.
    base_parents : sequence of int
        The shared parent set every move perturbs (an add's ``u`` must be outside it; a remove's
        ``u`` inside it).
    deltas : sequence of (str, int)
        ``("add", u)`` / ``("remove", u)`` moves.
    cond_threshold : float
        Conditioning-gate threshold routing near-singular bases to the two-full-score fallback.
    """
    if not deltas:
        return []

    cov = score_obj.cov
    n = score_obj.n
    log_n = score_obj.log_n
    lam = score_obj.lambda_value
    cov_diag = score_obj._cov_diag
    inv_cov_cache, _residual_cache, _derived_score_cache = _delta_caches(score_obj)

    i = int(i)
    S_base_sorted: tuple[int, ...] = tuple(sorted(int(x) for x in base_parents))

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
            if not (R_v_old > 0.0) or not np.isfinite(R_v_old):
                # A non-positive fresh residual means the inverse is too inaccurate to trust (cond
                # slipped under the gate); treat the base as ill so every move falls back exactly.
                R_v_old = None

    results: list[float] = [0.0] * len(deltas)

    if adds:
        add_deltas: dict[int, float] = {}
        add_us = sorted({u for _, u in adds if u != i and u not in S_base_sorted})
        if add_us and R_v_old is not None:
            if not S_base_sorted:
                for u in add_us:
                    R_u_old = float(cov_diag[u])
                    if R_u_old <= 0.0:
                        continue
                    e_vu = float(cov[i, u])
                    R_v_new = R_v_old - (e_vu * e_vu) / R_u_old
                    if R_v_new <= 0.0:
                        continue
                    add_deltas[u] = float(n * (_log(R_v_new) - _log(R_v_old)) + log_n * lam)
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
                R_v_new_arr = R_v_old - (e_vU * e_vU) / safe_R_u
                safe_mask = safe_mask & (R_v_new_arr > 0.0)
                if np.any(safe_mask):
                    deltas_arr = (
                        n * (_log(np.where(safe_mask, R_v_new_arr, 1.0)) - _log(R_v_old))
                        + log_n * lam
                    )
                    for pos, u in enumerate(add_us):
                        if safe_mask[pos]:
                            add_deltas[u] = float(deltas_arr[pos])
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
            for u in remove_us:
                u_idx = idx_of[u]
                M_uu = float(Mdiag[u_idx])
                if M_uu <= 0.0 or not np.isfinite(M_uu):
                    continue
                R_u_new = 1.0 / M_uu
                M_col = M_old[:, u_idx]
                num = float(cs_v @ M_col) - float(cs_v[u_idx]) * M_uu
                e_vu = float(cov[i, u]) + num / M_uu
                R_v_new = R_v_old + (e_vu * e_vu) / R_u_new
                if R_v_new <= 0.0 or not np.isfinite(R_v_new):
                    continue
                remove_deltas[u] = float(n * (_log(R_v_new) - log_Rv_old) - log_n * lam)
        for idx, u in removes:
            d = remove_deltas.get(u)
            if d is None:
                d = _delta_two_scores(score_obj, i, S_base_sorted, u, "remove")
            results[idx] = d

    return results


class DeltaBICScore:
    """Stateful companion to :class:`~andrey.core.score.BICScore` for the Schur delta path.

    Wraps a :class:`BICScore` and owns the inv-cov and residual caches that amortize repeated moves,
    exposing :meth:`delta` (scalar) and :meth:`score_many_with_base` (batched) for HC/GST. All BIC
    constants (covariance, sample count, ``log_n``, ``lambda_value``) are read from the wrapped
    score, so scalar and batched paths agree with :meth:`BICScore.score` within the
    conditioning-gated ``1e-9`` contract.

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
        self.derived_score_cache: OrderedDict = OrderedDict()
        # Route the batched surface's instance-level caches to this companion's caches.
        _DELTA_CACHES[score] = (self.inv_cov_cache, self.residual_cache, self.derived_score_cache)

    def delta(self, v: int, S_old_sorted: tuple[int, ...], u: int, op: str) -> float:
        """Return ``score(v, S_new) - score(v, S_old)`` for a single ``add``/``remove`` of ``u``."""
        return local_score_bic_delta(
            self.score_obj.cov,
            self.score_obj.n,
            v,
            S_old_sorted,
            u,
            op,
            lambda_value=self.score_obj.lambda_value,
            log_n=self.score_obj.log_n,
            inv_cov_cache=self.inv_cov_cache,
            residual_cache=self.residual_cache,
            cond_threshold=self.cond_threshold,
        )

    def score_many_with_base(
        self,
        i: int,
        base_parents: Sequence[int],
        deltas: Sequence[tuple[str, int]],
    ) -> list[float]:
        """Return absolute scores ``[score(i, base +/- delta_k)]`` for a batch of single moves."""
        return score_many_with_base(
            self.score_obj, i, base_parents, deltas, cond_threshold=self.cond_threshold
        )

    def delta_many_with_base(
        self,
        i: int,
        base_parents: Sequence[int],
        deltas: Sequence[tuple[str, int]],
    ) -> list[float]:
        """Return RAW deltas ``[score(i, base +/- delta_k) - score(i, base)]`` for single moves."""
        return delta_many_with_base(
            self.score_obj, i, base_parents, deltas, cond_threshold=self.cond_threshold
        )
