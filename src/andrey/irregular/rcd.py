"""RCD: repetitive causal discovery of a linear non-Gaussian model with latent confounders.

RCD repeats three stages until no ancestor set grows. Ancestors: in each variable set of up to
``max_explanatory_num + 1`` members whose residuals on their common ancestors stay non-Gaussian
(Shapiro-Wilk) and correlated (Pearson), a member whose residual is independent (HSIC) of the
others is a sink, and the others become its ancestors. Parents: an ancestor is a parent when the
two residuals, each taken on the other ancestors, stay correlated. Confounders: a non-parent pair
whose residuals stay correlated shares a latent common cause.

The estimated ``B`` (``x_i = sum_j B[i, j] x_j``) holds the least-squares parent coefficients and
``NaN`` on confounded pairs, which the PAG adapter turns into ``<->``. A fit's residuals, p-values,
and HSIC grams depend only on column indices, so :class:`_FitCaches` computes each once.

References
----------
Maeda, Shimizu. "RCD: Repetitive causal discovery of linear non-Gaussian acyclic models with latent
confounders." AISTATS 2020, PMLR 108:735-745.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.stats import pearsonr, shapiro
from sklearn.linear_model import LinearRegression

from andrey.core.independence import gamma_gram, hsic_gamma_from_grams

# scipy's Pearson p-value machinery imported directly, to skip the general test wrapper's per-call
# input validation. Older scipy lacks these private symbols, so fall back to its public beta law.
try:
    from scipy.stats._stats_py import _get_pvalue, _SimpleBeta
except ImportError:  # pragma: no cover - exercised only on older scipy
    from scipy.stats import beta as _scipy_beta

    class _SimpleBeta:  # type: ignore[no-redef]
        def __init__(self, a: float, b: float, loc: float = 0, scale: float = 1) -> None:
            self._dist = _scipy_beta(a, b, loc=loc, scale=scale)

        def cdf(self, x: float) -> float:
            return self._dist.cdf(x)

        def sf(self, x: float) -> float:
            return self._dist.sf(x)

    def _get_pvalue(statistic, distribution, alternative, symmetric=True, xp=None):  # type: ignore[no-redef]
        if alternative == "two-sided":
            return 2 * min(distribution.cdf(statistic), distribution.sf(statistic))
        if alternative == "less":
            return distribution.cdf(statistic)
        return distribution.sf(statistic)


# Memo tables keyed by column-index signatures, valid only while the data matrix stays fixed:
# a residual/coefficient pair per ``(endog, exog-order)``, a Shapiro-Wilk p-value per
# ``(column, conditioning-set)``, and a gamma-HSIC gram per column.
_ResidMemo = dict[tuple[int, tuple[int, ...]], tuple[np.ndarray, np.ndarray]]
_ShapiroMemo = dict[tuple[int, tuple[int, ...]], float]
_GramMemo = dict[int, tuple[np.ndarray, float]]


@dataclass(frozen=True)
class RcdResult:
    """Fitted RCD weights and ancestor sets.

    ``adjacency_matrix_`` is the ``(d, d)`` weight matrix (``B[i, j]`` weights edge ``j -> i``) with
    ``NaN`` on confounded pairs; ``ancestors_list_[i]`` is the recovered ancestor set of variable
    ``i``.
    """

    adjacency_matrix_: np.ndarray
    ancestors_list_: list[set[int]]


@dataclass
class _FitCaches:
    """Per-fit memo tables reused across the subset sweep and the parent/confounder stages.

    Every entry is a pure function of column-index signatures over the fixed data matrix, so a hit
    returns exactly what a recomputation would. ``resid_coef`` and ``column_gram`` are keyed only
    for the canonical data matrix; residual matrices built for a non-empty conditioning set are
    transient and cached per variable set instead.
    """

    resid_coef: _ResidMemo = field(default_factory=dict)
    shapiro_p: _ShapiroMemo = field(default_factory=dict)
    column_gram: _GramMemo = field(default_factory=dict)


# ---- ordinary-least-squares residuals and the elementary tests ----------------------------------


def _resid_and_coef(
    X: np.ndarray, endog: int, exog: list[int], memo: _ResidMemo | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary-least-squares residual and coefficients of ``X[:, endog]`` on ``X[:, exog]``.

    An intercept is fitted and discarded; the returned coefficients align with ``exog``. When
    ``memo`` is given (the matrix is the fixed data), the ``(endog, exog)`` result is stored and
    reused; the key keeps the ``exog`` order so a cached coefficient vector aligns with its request.
    """
    key = (endog, tuple(exog))
    if memo is not None:
        cached = memo.get(key)
        if cached is not None:
            return cached
    regressor = LinearRegression()
    regressor.fit(X[:, exog], X[:, endog])
    result = (X[:, endog] - regressor.predict(X[:, exog]), regressor.coef_)
    if memo is not None:
        memo[key] = result
    return result


def _residual_matrix(
    X: np.ndarray, U: tuple[int, ...], common: set[int], memo: _ResidMemo | None = None
) -> np.ndarray:
    """Residualize each variable in ``U`` on the common-ancestor set ``common``.

    With no common ancestors the data pass through unchanged; otherwise only the ``U`` columns of
    the returned matrix are meaningful (the rest stay zero and are never read).
    """
    if len(common) == 0:
        return X
    Y = np.zeros_like(X)
    exog = list(common)
    for xj in U:
        Y[:, xj], _ = _resid_and_coef(X, xj, exog, memo)
    return Y


def _is_non_gaussian(
    Y: np.ndarray,
    U: tuple[int, ...],
    shapiro_alpha: float,
    memo: _ShapiroMemo,
    common_sig: tuple[int, ...],
) -> bool:
    """True when every ``U`` column fails Shapiro-Wilk Gaussianity at ``shapiro_alpha``.

    Each column's p-value is memoized on ``(column, common_sig)`` -- the conditioning set that fixes
    the residualized column -- so a column recurring across variable sets is tested once.
    """
    for xj in U:
        key = (xj, common_sig)
        p_value = memo.get(key)
        if p_value is None:
            p_value = shapiro(Y[:, xj])[1]
            memo[key] = p_value
        if p_value > shapiro_alpha:
            return False
    return True


def _pearson_pvalue_fast(a: np.ndarray, b: np.ndarray) -> float | None:
    """Two-sided Pearson p-value of ``a`` and ``b`` via the beta tail, or ``None`` for edge inputs.

    Reproduces the value :func:`scipy.stats.pearsonr` returns while skipping its general-purpose
    input wrapper: the correlation is formed with the same numerically stable normalization and the
    p-value from the same symmetric beta law. ``None`` signals a degenerate or unsupported input
    (not 1-D, mismatched or under-length, complex, or a near-constant column) for the caller to
    resolve with the full test.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim != 1 or b.ndim != 1 or a.shape[0] != b.shape[0] or a.shape[0] < 2:
        return None
    dtype = np.result_type(a.dtype, b.dtype)
    if np.issubdtype(dtype, np.integer):
        dtype = np.asarray(1.0).dtype
    if np.issubdtype(dtype, np.complexfloating):
        return None
    a = a.astype(dtype, copy=False)
    b = b.astype(dtype, copy=False)
    n = a.shape[0]
    if np.all(a == a[0]) or np.all(b == b[0]):
        return None

    threshold = np.finfo(dtype).eps ** 0.75
    a_mean = np.mean(a, keepdims=True)
    b_mean = np.mean(b, keepdims=True)
    a_centered = a - a_mean
    b_centered = b - b_mean
    a_max = np.max(np.abs(a_centered), keepdims=True)
    b_max = np.max(np.abs(b_centered), keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm_a = a_max * np.linalg.norm(a_centered / a_max, axis=0, keepdims=True)
        norm_b = b_max * np.linalg.norm(b_centered / b_max, axis=0, keepdims=True)
    if np.any(norm_a < threshold * np.abs(a_mean)) or np.any(norm_b < threshold * np.abs(b_mean)):
        return None

    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.sum(a_centered / norm_a * b_centered / norm_b)
    one = np.asarray(1, dtype=dtype)
    r = np.asarray(np.clip(r, -one, one))
    if n == 2:
        p_value = np.where(np.asarray(np.isnan(r)), np.nan * one, one)
    else:
        beta_shape = n / 2 - 1
        dist = _SimpleBeta(beta_shape, beta_shape, loc=-1, scale=2)
        p_value = _get_pvalue(r, dist, "two-sided", xp=np)
    if getattr(p_value, "ndim", 0) == 0:
        return float(p_value[()])
    return None


def _is_correlated(a: np.ndarray, b: np.ndarray, cor_alpha: float) -> bool:
    """True when the Pearson correlation of ``a`` and ``b`` is significant at ``cor_alpha``.

    The p-value comes from the direct beta-tail computation; when that is undefined, non-finite, or
    lands within ``1e-12`` of ``cor_alpha`` the full :func:`scipy.stats.pearsonr` resolves the
    decision, so the significance verdict is identical to the reference test.
    """
    p = _pearson_pvalue_fast(a, b)
    if p is None or not np.isfinite(p) or abs(float(p) - cor_alpha) <= 1e-12:
        p = pearsonr(a, b)[1]
    return p < cor_alpha


def _column_gram(Y: np.ndarray, xj: int, n: int, memo: _GramMemo) -> tuple[np.ndarray, float]:
    """Return ``Y[:, xj]``'s cached gamma-HSIC gram, computing it once per column in ``memo``."""
    gram = memo.get(xj)
    if gram is None:
        gram = gamma_gram(Y[:, xj].reshape(n, 1))
        memo[xj] = gram
    return gram


def _is_independent_of_residual(
    Y: np.ndarray,
    xi: int,
    exog: list[int],
    ind_alpha: float,
    resid_memo: _ResidMemo | None,
    gram_memo: _GramMemo,
) -> bool:
    """True when ``Y[:, xi]`` on ``exog`` leaves a residual HSIC-independent of each column.

    The residual comes from a single multiple regression (the default ``MLHSICR=False`` path uses
    ordinary least squares, no HSIC-minimizing refit). Each explanatory column's gamma-HSIC gram is
    reused from ``gram_memo``; only the fresh residual's gram is built per call.
    """
    n = Y.shape[0]
    residual, _ = _resid_and_coef(Y, xi, exog, resid_memo)
    resid_gram = gamma_gram(residual.reshape(n, 1))
    for xj in exog:
        _, p_value = hsic_gamma_from_grams(resid_gram, _column_gram(Y, xj, n, gram_memo), n)
        if not p_value > ind_alpha:
            return False
    return True


def _exists_ancestor_in_U(M: list[set[int]], xi: int, exog: list[int]) -> bool:
    """True when ``xi`` already ancestors an ``exog`` member, or ``exog`` are all its ancestors.

    Either case rules ``xi`` out as a fresh sink for the current variable set.
    """
    for xj in exog:
        if xi in M[xj]:
            return True
    return set(exog) <= M[xi]


# ---- the three RCD stages -----------------------------------------------------------------------


def extract_ancestors(
    X: np.ndarray,
    caches: _FitCaches,
    *,
    max_explanatory_num: int,
    cor_alpha: float,
    ind_alpha: float,
    shapiro_alpha: float,
) -> list[set[int]]:
    """Recover each variable's ancestor set by repeatedly naming the unique sink of a variable set.

    Sweeps variable sets ``U`` of growing size (2 up to ``max_explanatory_num + 1``), residualizing
    on common ancestors and testing non-Gaussianity, mutual correlation, and residual independence;
    a set with exactly one independent (sink) member hands its other members to that sink's ancestor
    set. The sweep restarts whenever an ancestor set grows and stops once a full pass adds nothing.
    """
    d = X.shape[1]
    M: list[set[int]] = [set() for _ in range(d)]
    seen_common: dict[tuple[int, ...], set[int]] = {}
    width = 1

    while True:
        changed = False
        for U in combinations(range(d), width + 1):
            common = set.intersection(*(M[xj] for xj in U))
            if U in seen_common and common == seen_common[U]:
                continue

            common_sig = tuple(sorted(common))
            Y = _residual_matrix(X, U, common, caches.resid_coef)
            if not _is_non_gaussian(Y, U, shapiro_alpha, caches.shapiro_p, common_sig):
                continue

            pairs = combinations(U, 2)
            if not all(_is_correlated(Y[:, xi], Y[:, xj], cor_alpha) for xi, xj in pairs):
                continue

            # A residual matrix built on a non-empty conditioning set is transient; its column grams
            # (and the sink residuals) are cached per variable set, while an untouched (``Y is X``)
            # matrix reuses the fit-level caches keyed by column index.
            on_data = Y is X
            resid_memo = caches.resid_coef if on_data else None
            gram_memo = caches.column_gram if on_data else {}

            U_set = set(U)
            sinks = []
            for xi in U:
                exog = list(U_set - {xi})
                if _exists_ancestor_in_U(M, xi, exog):
                    continue
                if _is_independent_of_residual(Y, xi, exog, ind_alpha, resid_memo, gram_memo):
                    sinks.append(xi)

            if len(sinks) == 1:
                sink = sinks[0]
                new_ancestors = U_set - {sink}
                if not new_ancestors <= M[sink]:
                    M[sink] |= new_ancestors
                    changed = True

            seen_common[U] = common

        if changed:
            width = 1
        elif width < max_explanatory_num:
            width += 1
        else:
            break

    return M


def extract_parents(
    X: np.ndarray, M: list[set[int]], caches: _FitCaches, *, cor_alpha: float
) -> list[set[int]]:
    """Keep an ancestor as a direct parent when the two partial residuals stay correlated.

    For candidate ``x_j`` of ``x_i``: residualize ``x_i`` on its other ancestors and ``x_j`` on the
    ancestors they share, then test the residuals for Pearson correlation.
    """
    d = X.shape[1]
    P: list[set[int]] = [set() for _ in range(d)]
    memo = caches.resid_coef
    for xi in range(d):
        for xj in M[xi]:
            others = M[xi] - {xj}
            zi = _resid_and_coef(X, xi, list(others), memo)[0] if others else X[:, xi]
            shared = M[xi] & M[xj]
            wj = _resid_and_coef(X, xj, list(shared), memo)[0] if shared else X[:, xj]
            if _is_correlated(wj, zi, cor_alpha):
                P[xi].add(xj)
    return P


def extract_confounded_pairs(
    X: np.ndarray, P: list[set[int]], caches: _FitCaches, *, cor_alpha: float
) -> list[set[int]]:
    """Flag each non-parent pair whose parent-residuals stay correlated as latently confounded.

    A variable is first residualized on its recovered parents; two such residuals that remain
    Pearson-correlated point to an unobserved common cause.
    """
    d = X.shape[1]
    C: list[set[int]] = [set() for _ in range(d)]
    memo = caches.resid_coef
    residual_to_parents: list[np.ndarray | None] = [None] * d

    def resid(idx: int) -> np.ndarray:
        cached = residual_to_parents[idx]
        if cached is None:
            cached = _resid_and_coef(X, idx, list(P[idx]), memo)[0] if P[idx] else X[:, idx]
            residual_to_parents[idx] = cached
        return cached

    for i, j in combinations(range(d), 2):
        if (i in P[j]) or (j in P[i]):
            continue
        if _is_correlated(resid(i), resid(j), cor_alpha):
            C[i].add(j)
            C[j].add(i)
    return C


def estimate_adjacency(
    X: np.ndarray, P: list[set[int]], C: list[set[int]], caches: _FitCaches
) -> np.ndarray:
    """Assemble ``B``: least-squares parent coefficients on finite cells, ``NaN`` where confounded.

    ``B[i, j]`` weights edge ``j -> i``; a ``NaN`` cell marks the shared-confounder sentinel.
    """
    d = X.shape[1]
    B = np.zeros((d, d), dtype=np.float64)
    for xi in range(d):
        parents = sorted(P[xi])
        if not parents:
            continue
        _, coef = _resid_and_coef(X, xi, parents, caches.resid_coef)
        for pos, xj in enumerate(parents):
            B[xi, xj] = coef[pos]
    for xi in range(d):
        for xj in sorted(C[xi]):
            B[xi, xj] = np.nan
    return B


def fit_rcd(
    X: np.ndarray,
    *,
    max_explanatory_num: int = 2,
    cor_alpha: float = 0.01,
    ind_alpha: float = 0.01,
    shapiro_alpha: float = 0.01,
) -> RcdResult:
    """Fit RCD to ``X`` (rows are samples, columns variables); return the confounder-aware result.

    Runs ancestor extraction, parent extraction, and confounder detection, then assembles the
    adjacency matrix (``NaN`` on confounded pairs). ``max_explanatory_num`` caps the explanatory-set
    size; ``cor_alpha`` / ``ind_alpha`` / ``shapiro_alpha`` are the Pearson, HSIC, and Shapiro-Wilk
    significance levels. Deterministic on its default settings.
    """
    X = np.asarray(X, dtype=np.float64)
    caches = _FitCaches()
    M = extract_ancestors(
        X,
        caches,
        max_explanatory_num=max_explanatory_num,
        cor_alpha=cor_alpha,
        ind_alpha=ind_alpha,
        shapiro_alpha=shapiro_alpha,
    )
    P = extract_parents(X, M, caches, cor_alpha=cor_alpha)
    C = extract_confounded_pairs(X, P, caches, cor_alpha=cor_alpha)
    return RcdResult(adjacency_matrix_=estimate_adjacency(X, P, C, caches), ancestors_list_=M)
